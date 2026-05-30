"""es_select.py — Pick ES individuals per preference and evaluate on test.

For each preference λ over the simplex:
  1. Load all (gating_network, train_fitness) pairs from --es_gating_dir
  2. Pick the best individual by utility (linear or Tchebyshev w/ z* from es_meta.json)
  3. Evaluate the selected individual on the test dataset

Preferences that select the same individual share a single test-set evaluation.

TO RUN:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./evolutionary/es_select.py \\
    --expert_model_paths './models/ppo/ppo_beaver_reward_2204/best_model' \\
                         './models/ppo/ppo_beaver_cost_2204/best_model' \\
    --es_gating_dir      './models/es/es_beaver_0105/gen_0035' \\
    --reward_names       'beaver_reward,beaver_cost' \\
    --dataset_name       'PKU-Alignment/PKU-SafeRLHF-10K' \\
    --utility            'tchebyshev' \\
    --run_name           'es_select_0105' \\
    2>&1 | tee ./logs/es_select_0105.log
"""

import datetime
import gc
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from torch.utils.data import DataLoader

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from es_architecture import MoEForCausalLM
from es_utils import (
    Instructions, Instructions_summary, REWARD_PATHS, RewardModels,
    build_dataset_beaver_eval, build_dataset_eval, build_dataset_summary_eval,
    generate_and_score, get_simplex_samples, load_gating_network,
    load_main_tokenizer,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    base_model_name:    str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'beaver_reward,beaver_cost'
    dataset_name:       str       = 'PKU-Alignment/PKU-SafeRLHF-10K'
    eval_prompts:       int       = 0
    batch_size:         int       = 128
    num_continuations:  int       = 1
    pref_step:          float     = 0.1
    es_gating_dir:      str       = ''     # dir with ind_XXX/ subdirs (each has fitness.json + gating_network.pt)
    es_meta_path:       str       = ''     # explicit path to es_meta.json (auto-discovered if blank)
    utility:            str       = 'linear'   # 'linear' | 'tchebyshev'
    gpu_id:             int       = -1
    save_directory:     str       = './results/es_select/'
    run_name:           str       = 'es_select'
    seed:               int       = 8888


# ---------------------------------------------------------------------------
# Pool loading + utility selection
# ---------------------------------------------------------------------------

def _load_pool(gating_dir: str):
    """Return (ind_dirs, fitnesses) for all ind_XXX subdirs with fitness.json."""
    ind_dirs, fitnesses = [], []
    for ind_dir in sorted(glob.glob(os.path.join(gating_dir, 'ind_*'))):
        fp = os.path.join(ind_dir, 'fitness.json')
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        mf = data.get('mean_fitness') or data.get('fitness')
        if mf is not None:
            ind_dirs.append(ind_dir)
            fitnesses.append(np.array(mf, dtype=np.float32))
    return ind_dirs, (np.stack(fitnesses) if fitnesses else None)


def _find_es_meta(gating_dir: str, override: str) -> str:
    """Locate es_meta.json: explicit override first, then walk up from gating_dir."""
    if override:
        return override if os.path.isfile(override) else ''
    cur = os.path.abspath(gating_dir)
    for _ in range(4):
        cand = os.path.join(cur, 'es_meta.json')
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return ''


def _select_individual(fitnesses: np.ndarray, lam: np.ndarray,
                       utility: str, z_star: np.ndarray = None) -> int:
    """Return index into fitnesses of the best individual for preference λ."""
    if utility == 'linear':
        return int(np.argmax(fitnesses @ lam))
    if utility == 'tchebyshev':
        # min_i max_m λ_m * (z*_m - r_i_m)  — closest to ideal in worst dimension
        gaps   = np.maximum(z_star[None, :] - fitnesses, 0.0)
        scores = (lam[None, :] * gaps).max(axis=1)
        return int(np.argmin(scores))
    raise ValueError(f"utility must be 'linear' or 'tchebyshev', got {utility!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_individual(ind_dir, experts, loader, tokenizer, reward_models,
                    instructions, generation_kwargs, gpu_id, num_continuations,
                    cache_path=None, rank=0, label=''):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f'  rank{rank} cached: {label} {cached["rewards"]}', flush=True)
        return cached['rewards']

    lm_hidden_size = experts[0].config.hidden_size
    device         = f'cuda:{gpu_id}'
    gating = load_gating_network(ind_dir, lm_hidden_size=lm_hidden_size,
                                 num_experts=len(experts), device=device)
    if gating is None:
        raise FileNotFoundError(f'No gating_network.pt under {ind_dir}')
    moe = MoEForCausalLM(experts, gating).to(device)
    moe.eval()

    batch_rewards = []
    for batch in loader:
        r = generate_and_score(moe, batch['input_ids'], batch['attention_mask'],
                                tokenizer, reward_models, instructions,
                                generation_kwargs, gpu_id, num_continuations)
        batch_rewards.append(r)
    mean_r = np.mean(batch_rewards, axis=0).tolist()
    print(f'  rank{rank} done: {label} {mean_r}', flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'label': label, 'rewards': mean_r}, f)

    del moe, gating
    gc.collect(); torch.cuda.empty_cache()
    return mean_r


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

args: Args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert args.utility in ('linear', 'tchebyshev'), \
    f"utility must be 'linear' or 'tchebyshev', got {args.utility!r}"
assert args.es_gating_dir, 'es_gating_dir is required'

output_dir = os.path.join(args.save_directory, args.run_name)
os.makedirs(output_dir, exist_ok=True)

if 'RANK' in os.environ:
    torch.distributed.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=240))
accelerator = Accelerator()
gpu_id     = args.gpu_id if args.gpu_id >= 0 else accelerator.local_process_index
rank       = accelerator.process_index
world_size = accelerator.num_processes
is_main    = accelerator.is_main_process
device     = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in args.reward_names.split(',')]
M = len(reward_names)

tokenizer              = load_main_tokenizer(args.expert_model_paths[0])
tokenizer.padding_side = 'left'

_rm_paths     = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(_rm_paths, _rm_paths, gpu_id)

_max_new_tokens = 128 if args.dataset_name in {
    'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K'} else 48
generation_kwargs = dict(max_new_tokens=_max_new_tokens, do_sample=False)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    ds = build_dataset_beaver_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions()
elif args.dataset_name == 'Anthropic/hh-rlhf':
    ds = build_dataset_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions()
elif args.dataset_name == 'openai/summarize_from_feedback':
    ds = build_dataset_summary_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions_summary()
else:
    raise ValueError(f'Unsupported dataset: {args.dataset_name!r}')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in ds.column_names:
        ds = ds.remove_columns(key)
ds = ds.with_format('numpy')
loader = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                    drop_last=False)
if is_main:
    print(f'{len(ds)} test prompts', flush=True)

# ---------------------------------------------------------------------------
# Load pool + z_star (for tchebyshev)
# ---------------------------------------------------------------------------

ind_dirs, fitnesses = _load_pool(args.es_gating_dir)
if not ind_dirs:
    raise FileNotFoundError(f'No ind_XXX/fitness.json under {args.es_gating_dir}')

z_star = None
if args.utility == 'tchebyshev':
    meta_path = _find_es_meta(args.es_gating_dir, args.es_meta_path)
    if not meta_path:
        raise FileNotFoundError(
            'es_meta.json not found near es_gating_dir; pass --es_meta_path explicitly')
    with open(meta_path) as f:
        meta = json.load(f)
    z_star = np.array(meta['z_star'], dtype=np.float32)
    if is_main:
        print(f'Loaded z_star={np.round(z_star, 4)} from {meta_path}', flush=True)

if is_main:
    print(f'Loaded {len(ind_dirs)} individuals from {args.es_gating_dir}', flush=True)

# ---------------------------------------------------------------------------
# Select individuals per preference (deterministic across ranks)
# ---------------------------------------------------------------------------

preferences = get_simplex_samples(M, step=args.pref_step)
n_prefs     = len(preferences)

selections: List[int] = [
    _select_individual(fitnesses, np.array(lam, dtype=np.float32),
                       args.utility, z_star)
    for lam in preferences
]

# Dedupe: which preferences picked each unique individual?
ind_to_prefs: dict = {}
for pi, ii in enumerate(selections):
    ind_to_prefs.setdefault(ii, []).append(pi)
unique_inds = sorted(ind_to_prefs.keys())

if is_main:
    print(f'\n{len(unique_inds)} unique winners across {n_prefs} preferences '
          f'(utility={args.utility})', flush=True)

# ---------------------------------------------------------------------------
# Load expert models
# ---------------------------------------------------------------------------

if is_main:
    print(f'\nLoading {len(args.expert_model_paths)} expert models …', flush=True)
experts = []
for i, path in enumerate(args.expert_model_paths):
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.resize_token_embeddings(len(tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    experts.append(m)
    if is_main:
        print(f'  [{i}] {path}', flush=True)

# ---------------------------------------------------------------------------
# Shard unique individuals across ranks and evaluate
# ---------------------------------------------------------------------------

my_inds    = [ui for j, ui in enumerate(unique_inds) if j % world_size == rank]
my_results = {}   # {ind_idx_into_pool: test_rewards}

for ii in my_inds:
    ind_dir  = ind_dirs[ii]
    ind_name = os.path.basename(ind_dir)
    cache_p  = os.path.join(output_dir, 'cache', f'ind_{ii:03d}.json')
    rewards  = eval_individual(ind_dir, experts, loader, tokenizer, reward_models,
                                instructions, generation_kwargs, gpu_id,
                                args.num_continuations,
                                cache_path=cache_p, rank=rank,
                                label=f'ind={ind_name}')
    my_results[ii] = rewards

# ---------------------------------------------------------------------------
# Gather + write results
# ---------------------------------------------------------------------------

if world_size > 1:
    all_results = [None] * world_size
    torch.distributed.all_gather_object(all_results, my_results)
else:
    all_results = [my_results]

if is_main:
    merged_inds: dict = {}
    for r in all_results:
        merged_inds.update(r)

    per_preference = []
    for pi, ii in enumerate(selections):
        per_preference.append({
            'pref':          list(preferences[pi]),
            'ind':           os.path.basename(ind_dirs[ii]),
            'ind_dir':       ind_dirs[ii],
            'train_fitness': fitnesses[ii].tolist(),
            'test_rewards':  merged_inds[ii],
        })

    out_path = os.path.join(output_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'utility':        args.utility,
            'reward_names':   reward_names,
            'z_star':         z_star.tolist() if z_star is not None else None,
            'per_preference': per_preference,
        }, f, indent=2)
    print(f'\nResults saved → {out_path}', flush=True)

    rn_width = 10
    print('\n' + '=' * 80)
    print(f'utility={args.utility}')
    header = (f'{"λ":<24}  {"ind":<14}  ' +
              '  '.join(f'{rn:>{rn_width}}' for rn in reward_names))
    print(header)
    print('-' * len(header))
    for row in per_preference:
        lam_str = str(row['pref'])
        print(f'{lam_str:<24}  {row["ind"]:<14}  ' +
              '  '.join(f'{v:>{rn_width}.4f}' for v in row['test_rewards']))

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
