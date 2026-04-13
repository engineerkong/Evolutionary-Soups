"""nsgaii_test.py — Evaluate reward scores for NSGA-II trained GatingNetwork checkpoints.

Two modes:
  1. Fixed-coefficient MoE baselines (sanity check).
  2. NSGA-II trained GatingNetwork checkpoints (--gating_paths).
     Each individual checkpoint is evaluated ONCE to obtain its reward vector.
     At inference, select the best individual for a given preference λ by:
         best_i = argmax_i  dot(λ, reward_i)

gating_paths accepts:
  - A dir directly containing gating_network.pt
  - A dir containing ind_* sub-dirs each with gating_network.pt
    (e.g. gen_0030/ which contains gen_0030/ind_000/, gen_0030/ind_001/, …)

TO RUN:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./scripts/momoe/nsgaii_test.py \
    --sft_model_name ./models/sft/assistant_sft/model/ \
    --expert_model_paths ./models/ppo/assistant_ppo_harmless_2701/batch_832/ \
                         ./models/ppo/assistant_ppo_helpful_2701/batch_832/ \
    --gating_paths ./models/nsgaii/nsgaii_gating_0101/gen_0020/ \
    --run_name 'nsgaii_test_0101' 2>&1 | tee ./logs/nsgaii_test_0101.log
"""

import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from torch.utils.data import DataLoader

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_ppo, build_dataset_summary_ppo,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
    get_clean_data, load_main_tokenizer,
)
from nsgaii_architecture import GatingNetwork, MoEForCausalLM
from nsgaii_utils import REWARD_PATHS, load_gating_network, get_simplex_samples


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    sft_model_name:     str       = './models/sft/assistant_sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    gating_paths:       List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    exp_type:           str       = 'assistant'
    use_train_split:    bool      = False   # if True, use train split + build_dataset_ppo (matches training)
    eval_prompts:       int       = 0
    batch_size:         int       = 128
    max_new_tokens:     int       = 128
    do_sample:          bool      = False
    num_continuations:  int       = 1
    pref_step:          float     = 0.1   # simplex step for λ-selection table
    gpu_id:             int       = -1
    save_directory:     str       = './results/nsgaii/'
    run_name:           str       = 'nsgaii_test'
    seed:               int       = 42


# ---------------------------------------------------------------------------
# Fixed-coefficient gating (baseline)
# ---------------------------------------------------------------------------

class FixedGating(torch.nn.Module):
    def __init__(self, coeffs: List[float]):
        super().__init__()
        self.register_buffer('_c', torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._c.unsqueeze(0).expand(hidden_states.shape[0], -1)


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

def generate_and_score(model, input_ids, attention_mask, tokenizer,
                       reward_models, instructions, generation_kwargs,
                       gpu_id, num_continuations):
    device      = f'cuda:{gpu_id}'
    accumulated = None

    for _ in range(num_continuations):
        outputs         = model.generate(input_ids.to(device),
                                         attention_mask=attention_mask.to(device),
                                         **generation_kwargs)
        responses       = tokenizer.batch_decode(outputs.cpu())
        prompts_decoded = tokenizer.batch_decode(input_ids.cpu())
        del outputs

        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(instructions.get_input(r), instructions.get_response(r))
                 for r in responses_clean]
        scores = reward_models.get_reward_model_scores(
            pairs, normalize_rewards=False, round_digits=None)

        n_prompts, n_rewards = len(prompts_clean), len(scores)
        if accumulated is None:
            accumulated = [[[] for _ in range(n_rewards)] for _ in range(n_prompts)]
        for p in range(n_prompts):
            for k in range(n_rewards):
                accumulated[p][k].append(scores[k][p])
        torch.cuda.empty_cache()

    per_prompt = np.array([[np.mean(accumulated[p][k]) for k in range(n_rewards)]
                           for p in range(n_prompts)])
    return per_prompt.mean(axis=0).tolist()


def eval_configs(configs, loader, tokenizer, reward_models, instructions,
                 generation_kwargs, gpu_id, num_continuations,
                 results_dir='', rank=0, reward_names=None):
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

    results = []
    for config_idx, label, model in configs:
        if config_idx in done:
            print(f'  rank{rank} skip (cached): {label}', flush=True)
            results.append((config_idx, label, done[config_idx][1]))
            continue

        model.eval()
        batch_rewards = []
        for batch in loader:
            r = generate_and_score(model, batch['input_ids'], batch['attention_mask'],
                                   tokenizer, reward_models, instructions,
                                   generation_kwargs, gpu_id, num_continuations)
            batch_rewards.append(r)
        mean_r = np.mean(batch_rewards, axis=0).tolist()
        print(f'  rank{rank} done: {label}  {mean_r}', flush=True)
        results.append((config_idx, label, mean_r))

        if partial_path:
            record = {'config_idx': config_idx, 'label': label, 'rewards': mean_r,
                      'reward_names': reward_names or [], 'rank': rank}
            partial_file.write(json.dumps(record) + '\n')
            partial_file.flush()

    if partial_path:
        partial_file.close()
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

tokenizer              = load_main_tokenizer(args.sft_model_name)
tokenizer.padding_side = 'left'

reward_models = RewardModels(
    [REWARD_PATHS[n] for n in reward_names],
    [REWARD_PATHS[n] for n in reward_names],
    gpu_id,
)

generation_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample)
if args.do_sample:
    generation_kwargs.update(top_k=0, top_p=0.9, temperature=0.7)

if args.use_train_split:
    # Match nsgaii.py training exactly: same build_dataset_ppo, same filters (input_ids ≤ 256)
    train_split = f'train[:{args.eval_prompts}]' if args.eval_prompts > 0 else 'train'
    if args.exp_type == 'assistant':
        ds = build_dataset_ppo(
            'Anthropic/hh-rlhf', tokenizer,
            reward_models.rm_tokenizers[0], split=train_split)
        instructions = Instructions()
    else:
        ds = build_dataset_summary_ppo(
            'openai/summarize_from_feedback', tokenizer,
            reward_models.rm_tokenizers[0], split=train_split)
        instructions = Instructions_summary()
else:
    eval_split = f'test[:{args.eval_prompts}]' if args.eval_prompts > 0 else 'test'
    if args.exp_type == 'assistant':
        ds = build_dataset_eval_ppo(
            'Anthropic/hh-rlhf', tokenizer,
            reward_models.rm_tokenizers, split=eval_split)
        instructions = Instructions()
    else:
        ds = build_dataset_summary_eval_ppo(
            'openai/summarize_from_feedback', tokenizer,
            reward_models.rm_tokenizers, split=eval_split)
        instructions = Instructions_summary()

for key in ['text', 'prompt', 'response', 'query']:
    if key in ds.column_names:
        ds = ds.remove_columns(key)

if is_main:
    print(f'  {len(ds)} prompts after filtering', flush=True)

loader = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                    drop_last=False)

if is_main:
    print(f'\nLoading {len(args.expert_model_paths)} expert models ...', flush=True)
experts = []
for i, path in enumerate(args.expert_model_paths):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device)
    m.resize_token_embeddings(len(tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    experts.append(m)
    if is_main:
        print(f'  [{i}] {path}', flush=True)

n              = len(experts)
lm_hidden_size = experts[0].config.hidden_size

# ---------------------------------------------------------------------------
# Build config list
# ---------------------------------------------------------------------------

# all_configs = [(f'expert[{i}] standalone', experts[i]) for i in range(n)]
all_configs = []

# 1. Fixed-coefficient baselines
simplex = get_simplex_samples(len(reward_names), step=0.1)
for coeffs in simplex:
    all_configs.append((f'MoE fixed {coeffs}',
                        MoEForCausalLM(experts, FixedGating(coeffs)).to(device)))

# 2. NSGA-II GatingNetwork checkpoints — each evaluated ONCE.
#    At inference: select best individual via argmax dot(λ, reward_i).
nsgaii_start_idx = len(all_configs)   # index of first NSGA-II config

if args.gating_paths:
    resolved = _resolve_gating_paths(args.gating_paths)
    if is_main:
        print(f'\nFound {len(resolved)} gating checkpoint(s) ...', flush=True)

    for ckpt_path in resolved:
        gating = load_gating_network(ckpt_path, lm_hidden_size=lm_hidden_size,
                                     num_experts=n, device=device)
        if gating is None:
            if is_main:
                print(f'  [SKIP] {ckpt_path}', flush=True)
            continue
        ckpt_name = os.path.basename(ckpt_path)
        label     = f'MoE NSGAII [{ckpt_name}]'
        all_configs.append((label, MoEForCausalLM(experts, gating).to(device)))

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

my_results = eval_configs(my_configs, loader, tokenizer, reward_models, instructions,
                          generation_kwargs, gpu_id, args.num_continuations,
                          results_dir=output_dir, rank=rank,
                          reward_names=reward_names)

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

    col = 52
    print(f'\n{"Config":<{col}}  ' + '  '.join(f'{rn:>10}' for rn in reward_names))
    print('-' * (col + 14 * len(reward_names)))
    for _, label, mean_r in flat:
        print(f'{label:<{col}}  ' + '  '.join(f'{v:>10.4f}' for v in mean_r))

    # ── λ-selection table: for each preference, which NSGA-II individual wins? ──
    nsgaii_results = [(label, mean_r) for _, label, mean_r in flat
                      if label.startswith('MoE NSGAII')]
    if nsgaii_results:
        rewards_arr = np.array([r for _, r in nsgaii_results])  # (K, M)
        labels_arr  = [l for l, _ in nsgaii_results]

        print(f'\n{"λ":<30}  {"best individual":<40}  ' +
              '  '.join(f'{rn:>10}' for rn in reward_names))
        print('-' * (30 + 44 + 14 * len(reward_names)))
        for lam in get_simplex_samples(len(reward_names), step=args.pref_step):
            scores = rewards_arr @ np.array(lam)
            best_i = int(np.argmax(scores))
            print(f'{str(lam):<30}  {labels_arr[best_i]:<40}  ' +
                  '  '.join(f'{v:>10.4f}' for v in rewards_arr[best_i]))
