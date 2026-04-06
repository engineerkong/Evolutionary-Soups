"""moead_test.py — Evaluate reward scores across three modes:

  1. Pure expert models (standalone)
  2. MoE with fixed merging coefficients (sanity check: [1,0] should match expert[0])
  3. MoE with a trained GatingNetwork from MOEA/D (--gating_paths)

Single-GPU usage:
    python moead_test.py \
        --expert_model_paths ./models/ppo/harmless/batch_832/ \
                             ./models/ppo/helpful/batch_832/ \
        --sft_model_name ./models/sft/model/ \
        --gating_paths ./models/moead/moead_gating_0304/gen_0030/

Multi-GPU usage (configs sharded across ranks):
    torchrun --nproc_per_node=4 moead_test.py \
        --expert_model_paths ./models/ppo/harmless/batch_832/ \
                             ./models/ppo/helpful/batch_832/ \
        --sft_model_name ./models/sft/model/ \
        --gating_paths ./models/moead/moead_gating_0304/gen_0030/

gating_paths accepts:
  - A dir that directly contains gating_network.pt  (e.g. best/)
  - A dir containing sub-dirs with gating_network.pt (e.g. gen_0030/ or final/)
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from torch.utils.data import DataLoader

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import Instructions, get_clean_data, load_main_tokenizer
from moead_architecture import MoEForCausalLM
from moead_utils import REWARD_PATHS, load_gating_network, get_simplex_samples


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    sft_model_name:     str       = './models/sft/assistant_sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    gating_paths:       List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    eval_prompts:       int       = 0
    batch_size:         int       = 64
    max_new_tokens:     int       = 128
    do_sample:          bool      = True
    num_continuations:  int       = 1
    gpu_id:             int       = -1     # used only when not running under torchrun
    save_directory:     str       = './results/'
    run_name:           str       = 'moead_test'
    seed:               int       = 42


# ---------------------------------------------------------------------------
# Fixed-coefficient gating (sanity check helper)
# ---------------------------------------------------------------------------

class FixedGating(torch.nn.Module):
    """Returns a fixed coefficient vector regardless of input hidden states."""
    def __init__(self, coeffs: List[float]):
        super().__init__()
        self.register_buffer('_c', torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._c.unsqueeze(0).expand(hidden_states.shape[0], -1)


# ---------------------------------------------------------------------------
# Gating path resolution
# ---------------------------------------------------------------------------

def _resolve_gating_paths(paths: List[str]) -> List[str]:
    """Expand each entry in paths:
      - If the dir directly contains gating_network.pt → keep as-is.
      - If the dir contains sub-dirs with gating_network.pt → expand to those sub-dirs.
    Returned list is sorted by directory name within each source path.
    """
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
# Evaluation
# ---------------------------------------------------------------------------

def generate_and_score(model, input_ids, attention_mask, tokenizer,
                       reward_models, instructions, generation_kwargs,
                       gpu_id, num_continuations):
    """Generate responses and return mean reward vector over all prompts."""
    device = f'cuda:{gpu_id}'
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
        if hasattr(instructions, 'get_post'):
            scores = reward_models.get_reward_model_scores(
                pairs, instructions.get_post, normalize_rewards=False, round_digits=None)
        else:
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
    return per_prompt.mean(axis=0)


def eval_configs(configs, loader, tokenizer, reward_models, instructions,
                 generation_kwargs, gpu_id, num_continuations,
                 results_dir='', rank=0, reward_names=None):
    """Evaluate a list of (config_idx, label, model) tuples.

    If results_dir is set:
      - Appends each result to {results_dir}/rank{rank}.jsonl immediately after
        the config finishes (crash-safe continuous saving).
      - On startup, loads any existing partial file so already-done configs are
        skipped (resumption).

    Returns list of (config_idx, label, mean_reward_vector).
    """
    partial_path = os.path.join(results_dir, f'rank{rank}.jsonl') if results_dir else None

    # Load already-completed results from a previous (partial) run
    done = {}   # config_idx → (label, mean_r)
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
                    pass   # ignore truncated line from a previous crash
        if done:
            print(f'  rank{rank}: resuming — {len(done)} config(s) already done', flush=True)

    if partial_path:
        os.makedirs(results_dir, exist_ok=True)
        partial_file = open(partial_path, 'a')   # append mode

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

# ---------------------------------------------------------------------------
# Setup (all ranks)
# ---------------------------------------------------------------------------

reward_names = [x.strip() for x in args.reward_names.split(',')]
instructions = Instructions()

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

# Dataset (all ranks build independently — no I/O contention, just CPU work)
split_str = f'test[:{args.eval_prompts}]' if args.eval_prompts > 0 else 'test'
if is_main:
    print(f'Loading Anthropic/hh-rlhf {split_str} ...', flush=True)
ds = load_dataset('Anthropic/hh-rlhf', split=split_str)

def _tokenize(sample):
    parts = sample['chosen'].split('\n\nAssistant:')
    sample['prompt']    = '\n\nAssistant:'.join(parts[:-1]) + ' \n\nAssistant:'
    sample['input_ids'] = tokenizer.encode(sample['prompt'])
    return sample

ds = ds.map(_tokenize, batched=False, num_proc=4)
ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
ds = ds.remove_columns([c for c in ds.column_names if c != 'input_ids'])
ds.set_format(type='torch')
if is_main:
    print(f'  {len(ds)} prompts after filtering', flush=True)

loader = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                    drop_last=False)

# Experts (each rank loads onto its own GPU)
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

n = len(experts)

# ---------------------------------------------------------------------------
# Build full config list (on all ranks — gating nets are small)
# ---------------------------------------------------------------------------

# 1. Expert baselines
# all_configs = [(f'expert[{i}] standalone', experts[i]) for i in range(n)]
all_configs = []

# 2. Fixed-coefficient sanity checks
simplex = get_simplex_samples(len(reward_names), step=0.1)
for coeffs in simplex:
    all_configs.append((f'MoE fixed {coeffs}',
                        MoEForCausalLM(experts, FixedGating(coeffs)).to(device)))

# 3. Trained gating networks from MOEA/D
if args.gating_paths:
    resolved = _resolve_gating_paths(args.gating_paths)
    lm_hidden_size = experts[0].config.hidden_size
    if is_main:
        print(f'\nFound {len(resolved)} gating checkpoint(s) ...', flush=True)
    for path in resolved:
        gating = load_gating_network(path, lm_hidden_size=lm_hidden_size,
                                     num_experts=n, device=device)
        if gating is None:
            if is_main:
                print(f'  [SKIP] {path}', flush=True)
            continue
        lambda_file = os.path.join(path, 'lambda.json')
        if os.path.exists(lambda_file):
            meta  = json.load(open(lambda_file))
            label = (f'MoE MOEAD λ={np.round(meta["lambda"], 2).tolist()} '
                     f'fit={meta.get("fitness", float("nan")):.4f}')
        else:
            label = f'MoE MOEAD [{os.path.basename(path)}]'
        all_configs.append((label, MoEForCausalLM(experts, gating).to(device)))

# ---------------------------------------------------------------------------
# Shard configs across ranks and evaluate
# ---------------------------------------------------------------------------

# Each rank evaluates configs at positions rank, rank+world_size, rank+2*world_size, ...
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
# Gather results to rank 0 and print
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

    col = 44
    print(f'\n{"Config":<{col}}  ' + '  '.join(f'{rn:>10}' for rn in reward_names))
    print('-' * (col + 14 * len(reward_names)))
    for _, label, mean_r in flat:
        print(f'{label:<{col}}  ' + '  '.join(f'{v:>10.4f}' for v in mean_r))
