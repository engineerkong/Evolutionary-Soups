"""es_test.py — Evaluate every ES checkpoint on the test set.

Loads every gating checkpoint passed via --gating_paths and runs each one
through the full test (or train) loader to compute its mean reward vector.
Also prints a λ-selection table that, for each preference on the simplex,
shows which checkpoint maximises linear utility (optionally on normalised
rewards from --norm_rewards).

`gating_paths` accepts either:
  - a dir directly containing gating_network.pt, or
  - a dir containing ind_* sub-dirs each with gating_network.pt
    (e.g. gen_0030/  →  gen_0030/ind_000/, gen_0030/ind_001/, …).

Per-sample (prompt / response / reward) rows are also dumped to a per-rank CSV.

TO RUN:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./scripts/es/es_test.py \
    --expert_model_paths ./models/ppo/assistant_ppo_harmless_2701/batch_832/ \
                         ./models/ppo/assistant_ppo_helpful_2701/batch_832/ \
    --gating_paths ./models/es/es_gating_0101/gen_0020/ \
    --run_name 'es_test_0101' 2>&1 | tee ./logs/es_test_0101.log
"""

import csv
import datetime
import json
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from torch.utils.data import DataLoader

from es_architecture import GatingNetwork, MoEForCausalLM, SimpleGatingNetwork, SimpleMoEForCausalLM
from es_utils import (
    Instructions, Instructions_summary, REWARD_PATHS, RewardModels,
    build_dataset_beaver_eval, build_dataset_beaver_ppo,
    build_dataset_eval, build_dataset_ppo,
    build_dataset_summary_eval, build_dataset_summary_ppo,
    generate_and_score, get_simplex_samples,
    load_gating_network, load_main_tokenizer, load_simple_gating_network,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    base_model_name:    str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths: List[str] = field(default_factory=list)
    gating_paths:       List[str] = field(default_factory=list)
    reward_names:       str       = ''          # auto-selected from dataset_name if empty
    dataset_name:       str       = 'Anthropic/hh-rlhf'
    use_train_split:    bool      = False   # if True, use train split + build_dataset_ppo (matches training)
    eval_prompts:       int       = 0
    batch_size:         int       = 128
    do_sample:          bool      = False
    num_continuations:  int       = 1
    pref_step:          float     = 0.1   # simplex step for λ-selection table
    norm_rewards:       str       = ''   # path to reward_norm.json; if set, normalize rewards for utility comparison
    gpu_id:             int       = -1
    save_directory:     str       = './results/es/'
    run_name:           str       = 'es_test'
    seed:               int       = 8888


# ---------------------------------------------------------------------------
# Gating path resolution
# ---------------------------------------------------------------------------

def _resolve_gating_paths(paths: List[str]) -> List[str]:
    resolved = []
    for p in paths:
        p = p.rstrip('/')
        if os.path.exists(os.path.join(p, 'gating_network.pt')):
            resolved.append(p)
        elif os.path.isdir(p):
            subdirs = sorted([
                os.path.join(p, d) for d in os.listdir(p)
                if os.path.isdir(os.path.join(p, d))
                and os.path.exists(os.path.join(p, d, 'gating_network.pt'))
            ])
            if subdirs:
                resolved.extend(subdirs)
            else:
                print(f'[WARN] no gating_network.pt found under {p}', flush=True)
        else:
            print(f'[WARN] path not found: {p}', flush=True)
    return resolved


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_configs(configs, loader, tokenizer, reward_models, instructions,
                 generation_kwargs, gpu_id, num_continuations,
                 results_dir='', rank=0, reward_names=None,
                 samples_csv_path=None):
    partial_path = os.path.join(results_dir, f'rank{rank}.jsonl') if results_dir else None

    done = {}
    if partial_path and os.path.exists(partial_path):
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec['config_idx']] = (rec['label'], rec['rewards'])
                except json.JSONDecodeError:
                    pass
        if done:
            print(f'  rank{rank}: resuming — {len(done)} config(s) already done', flush=True)

    if partial_path:
        os.makedirs(results_dir, exist_ok=True)
        partial_file = open(partial_path, 'a')

    # ---- per-sample CSV (input / output / rewards) ----
    samples_file = samples_writer_csv = None
    if samples_csv_path is not None:
        os.makedirs(os.path.dirname(samples_csv_path) or '.', exist_ok=True)
        new_file = not os.path.exists(samples_csv_path)
        samples_file = open(samples_csv_path, 'a', newline='', encoding='utf-8')
        samples_writer_csv = csv.writer(samples_file)
        if new_file:
            reward_cols = [f'reward_{n}' for n in (reward_names or [])]
            samples_writer_csv.writerow(
                ['config_idx', 'label', 'batch_idx', 'continuation',
                 'prompt_idx_in_batch', 'prompt', 'response'] + reward_cols)
            samples_file.flush()

    results = []
    for config_idx, label, model in configs:
        if config_idx in done:
            print(f'  rank{rank} skip (cached): {label}', flush=True)
            results.append((config_idx, label, done[config_idx][1]))
            continue

        model.eval()
        batch_rewards = []
        for batch_idx, batch in enumerate(loader):
            # Per-config sink: closes over (config_idx, label, batch_idx) so
            # generate_and_score only sees a (cont, prompt, ...) signature.
            if samples_writer_csv is not None:
                def _writer(cont_idx, prompt_idx, prompt, response, rewards,
                            _ci=config_idx, _lbl=label, _bi=batch_idx):
                    samples_writer_csv.writerow(
                        [_ci, _lbl, _bi, cont_idx, prompt_idx, prompt, response]
                        + rewards)
            else:
                _writer = None

            r = generate_and_score(model, batch['input_ids'], batch['attention_mask'],
                                   tokenizer, reward_models, instructions,
                                   generation_kwargs, gpu_id, num_continuations,
                                   sample_writer=_writer)
            batch_rewards.append(np.asarray(r).tolist())   # store plain list (JSON-serialisable)

            if samples_file is not None:
                samples_file.flush()    # checkpoint after each batch
        mean_r = np.mean(batch_rewards, axis=0).tolist()
        print(f'  rank{rank} done: {label}  {mean_r}', flush=True)
        results.append((config_idx, label, mean_r))

        if partial_path:
            record = {'config_idx': config_idx, 'label': label, 'rewards': mean_r,
                      'batch_rewards': batch_rewards,
                      'reward_names': reward_names or [], 'rank': rank}
            partial_file.write(json.dumps(record) + '\n')
            partial_file.flush()

    if partial_path:
        partial_file.close()
    if samples_file is not None:
        samples_file.close()
    return results


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

args: Args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
torch.manual_seed(args.seed)
np.random.seed(args.seed)

output_dir = os.path.join(args.save_directory, args.run_name)
os.makedirs(output_dir, exist_ok=True)

if 'RANK' in os.environ:
    torch.distributed.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=240))
accelerator = Accelerator()
gpu_id      = (args.gpu_id if args.gpu_id >= 0 else accelerator.local_process_index)
rank        = accelerator.process_index
world_size  = accelerator.num_processes
is_main     = accelerator.is_main_process
device      = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in args.reward_names.split(',')]

tokenizer              = load_main_tokenizer(args.expert_model_paths[0])
tokenizer.padding_side = 'left'

_rm_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(_rm_paths, _rm_paths, gpu_id)

_max_new_tokens = 128 if args.dataset_name in {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K'} else 48
generation_kwargs = dict(max_new_tokens=_max_new_tokens, do_sample=args.do_sample)
if args.do_sample:
    generation_kwargs.update(top_k=0, top_p=0.9, temperature=0.7)

if args.use_train_split:
    if args.dataset_name == 'Anthropic/hh-rlhf':
        ds = build_dataset_ppo(
            args.dataset_name, tokenizer,
            reward_models.rm_tokenizers[0], split='train', size=args.eval_prompts if args.eval_prompts > 0 else None)
        instructions = Instructions()
    elif args.dataset_name == 'openai/summarize_from_feedback':
        ds = build_dataset_summary_ppo(
            args.dataset_name, tokenizer,
            reward_models.rm_tokenizers[0], split='train', size=args.eval_prompts if args.eval_prompts > 0 else None)
        instructions = Instructions_summary()
    elif args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
        ds = build_dataset_beaver_ppo(
            args.dataset_name, tokenizer,
            reward_models.rm_tokenizers[0], split='train', size=args.eval_prompts if args.eval_prompts > 0 else None)
        instructions = Instructions()
    else:
        raise ValueError(f'Unsupported dataset_name: {args.dataset_name!r}')
else:
    if args.dataset_name == 'Anthropic/hh-rlhf':
        ds = build_dataset_eval(
            args.dataset_name, tokenizer,
            reward_models.rm_tokenizers, split='test', size=args.eval_prompts if args.eval_prompts > 0 else None)
        instructions = Instructions()
    elif args.dataset_name == 'openai/summarize_from_feedback':
        ds = build_dataset_summary_eval(
            args.dataset_name, tokenizer,
            reward_models.rm_tokenizers, split='test', size=args.eval_prompts if args.eval_prompts > 0 else None)
        instructions = Instructions_summary()
    elif args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
        ds = build_dataset_beaver_eval(
            args.dataset_name, tokenizer,
            reward_models.rm_tokenizers, split='test', size=args.eval_prompts if args.eval_prompts > 0 else None)
        instructions = Instructions()
    else:
        raise ValueError(f'Unsupported dataset_name: {args.dataset_name!r}')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in ds.column_names:
        ds = ds.remove_columns(key)

ds = ds.with_format("numpy")

if is_main:
    print(f'  {len(ds)} prompts after filtering', flush=True)

loader = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                    drop_last=False)

if is_main:
    print(f'\nLoading {len(args.expert_model_paths)} expert models ...', flush=True)
experts = []
for i, path in enumerate(args.expert_model_paths):
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    base.resize_token_embeddings(len(tokenizer))
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    experts.append(m)
    if is_main:
        print(f'  [{i}] {path}', flush=True)

n              = len(experts)
lm_hidden_size = experts[0].config.hidden_size

# ---------------------------------------------------------------------------
# Build config list — every ES GatingNetwork checkpoint evaluated ONCE.
# At inference: select the best individual via argmax dot(λ, reward_i).
# ---------------------------------------------------------------------------

all_configs = []

if args.gating_paths:
    resolved = _resolve_gating_paths(args.gating_paths)
    if is_main:
        print(f'\nFound {len(resolved)} gating checkpoint(s) ...', flush=True)

    for ckpt_path in resolved:
        # Detect checkpoint type from config to choose correct architecture
        cfg_file = os.path.join(ckpt_path, 'gating_config.json')
        ckpt_type = 'GatingNetwork'
        if os.path.exists(cfg_file):
            with open(cfg_file) as _f:
                ckpt_type = json.load(_f).get('type', 'GatingNetwork')

        if ckpt_type == 'SimpleGatingNetwork':
            gating = load_simple_gating_network(ckpt_path, lm_hidden_size=lm_hidden_size,
                                                num_experts=n, device=device)
        else:
            gating = load_gating_network(ckpt_path, lm_hidden_size=lm_hidden_size,
                                         num_experts=n, device=device)
        if gating is None:
            if is_main:
                print(f'  [SKIP] {ckpt_path}', flush=True)
            continue
        ckpt_name = os.path.basename(ckpt_path)
        label     = f'MoE ES [{ckpt_name}]'
        if is_main:
            alphas    = gating.alpha_floats()
            alpha_str = (f'fixed={gating.fixed_alpha}' if gating.fixed_alpha is not None
                         else f'learned  min={min(alphas):.3f} max={max(alphas):.3f} mean={sum(alphas)/len(alphas):.3f}')
            print(f'  loaded {ckpt_name} ({ckpt_type})  α={alpha_str}', flush=True)

        if ckpt_type == 'SimpleGatingNetwork':
            moe_model = SimpleMoEForCausalLM(experts, gating).to(device)
        else:
            moe_model = MoEForCausalLM(experts, gating).to(device)
        all_configs.append((label, moe_model))


# ---------------------------------------------------------------------------
# Shard configs across ranks and evaluate
# ---------------------------------------------------------------------------

my_configs = [(idx, label, model)
              for idx, (label, model) in enumerate(all_configs)
              if idx % world_size == rank]

if is_main:
    print(f'\nEvaluating {len(all_configs)} configs over {len(ds)} prompts '
          f'({len(loader)} batches × {args.num_continuations} continuation(s))', flush=True)
    print(f'  {world_size} rank(s), ~{len(all_configs)//world_size + 1} configs per rank\n',
          flush=True)

samples_csv_path = os.path.join(output_dir, f'samples_rank{rank}.csv')
my_results = eval_configs(my_configs, loader, tokenizer, reward_models, instructions,
                          generation_kwargs, gpu_id, args.num_continuations,
                          results_dir=output_dir, rank=rank,
                          reward_names=reward_names,
                          samples_csv_path=samples_csv_path)

# ---------------------------------------------------------------------------
# Gather and print results
# ---------------------------------------------------------------------------

if world_size > 1:
    all_results = [None] * world_size
    torch.distributed.all_gather_object(all_results, my_results)
else:
    all_results = [my_results]

if is_main:
    flat = sorted([item for sublist in all_results for item in sublist],
                  key=lambda x: x[0])

    out_path = os.path.join(output_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump([{'config_idx': idx, 'label': label, 'rewards': mean_r,
                    'reward_names': reward_names}
                   for idx, label, mean_r in flat], f, indent=2)
    print(f'\nResults saved → {out_path}', flush=True)

    # ── Load normalisation stats if requested ──────────────────────────────
    r_min = r_max = r_range = None
    if args.norm_rewards:
        with open(args.norm_rewards) as f:
            norm = json.load(f)
        r_min   = np.array(norm['r_min'],  dtype=np.float64)
        r_max   = np.array(norm['r_max'],  dtype=np.float64)
        r_range = np.where(r_max > r_min, r_max - r_min, 1.0)
        print(f'\nReward normalisation loaded from {args.norm_rewards}')
        for n, lo, hi in zip(reward_names, r_min, r_max):
            print(f'  {n}: [{lo:.4f}, {hi:.4f}]')

    def maybe_norm(r):
        """Return normalised reward array if norm_rewards is set, else raw."""
        if r_min is None:
            return np.array(r)
        return (np.array(r) - r_min) / r_range

    col = 52
    print(f'\n{"Config":<{col}}  ' + '  '.join(f'{rn:>10}' for rn in reward_names))
    print('-' * (col + 14 * len(reward_names)))
    for _, label, mean_r in flat:
        print(f'{label:<{col}}  ' + '  '.join(f'{v:>10.4f}' for v in mean_r))

    # ── λ-selection table: for each preference, which ES individual wins? ──
    # Selection uses normalised rewards when norm_rewards is set so that
    # λ weights are applied on a consistent [0,1] scale (matching oracle).
    es_results = [(label, mean_r) for _, label, mean_r in flat
                  if label.startswith('MoE ES')]
    if es_results:
        rewards_arr      = np.array([r for _, r in es_results])       # (K, M) raw
        rewards_arr_norm = np.array([maybe_norm(r) for _, r in es_results])  # (K, M) normed
        labels_arr       = [l for l, _ in es_results]

        norm_tag = ' [normalised]' if r_min is not None else ''
        print(f'\n{"λ":<30}  {"best individual":<40}  {"utility"+norm_tag:>14}  ' +
              '  '.join(f'{rn:>10}' for rn in reward_names))
        print('-' * (30 + 44 + 18 + 14 * len(reward_names)))
        for lam in get_simplex_samples(len(reward_names), step=args.pref_step):
            lam_arr = np.array(lam)
            scores  = rewards_arr_norm @ lam_arr   # always use normed for selection
            best_i  = int(np.argmax(scores))
            utility = float(scores[best_i])
            print(f'{str(lam):<30}  {labels_arr[best_i]:<40}  {utility:>14.4f}  ' +
                  '  '.join(f'{v:>10.4f}' for v in rewards_arr[best_i]))

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
