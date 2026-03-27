"""Test trained GatingNetwork against ground truth from gating_dataset.csv.

Weight-space metrics : MAE between predicted and optimal weights.
Reward-space metrics : predicted / optimal / naive rewards estimated via RBF
                       surrogate built from collected_rewards.csv.

Works for any number of experts and both block_mode='uniform' | 'custom'.
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from scipy.interpolate import RBFInterpolator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_main_tokenizer
from new_architecture_old import GatingNetwork, GatingDataset, get_prompt_hidden
from new_utils_old import load_base_model, load_gating_network


@dataclass
class ScriptArguments:
    sft_model_name:     str       = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    checkpoint_path:    str       = ''
    rewards_csv:        str       = './data/new/new_assistant/collected_rewards.csv'
    dataset_csv:        str       = './data/new/new_assistant/gating_dataset.csv'
    reward_names:       str       = 'harmless,helpful'
    block_mode:         str       = 'uniform'   # 'uniform' | 'custom'
    save_directory:     str       = './results/new/'
    wandb_name:         str       = 'new_assistant_test'
    hidden_dim:         int       = 256
    batch_size:         int       = 128
    num_samples:        int       = 0   # 0 = use all


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
device       = accelerator.device
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_experts  = len(reward_names)

# ── Load tokenizer and frozen expert models ──────────────────────────────────
tokenizer     = load_main_tokenizer(script_args.sft_model_name)
expert_models = []
for path in script_args.expert_model_paths:
    m = load_base_model(path, target_device=device)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    expert_models.append(m)

with torch.no_grad():
    _dummy = tokenizer('hello', return_tensors='pt').to(device)
    _out   = expert_models[0](**_dummy, output_hidden_states=True)
    lm_hidden_size = _out.hidden_states[-1].shape[-1]
print(f'lm_hidden_size = {lm_hidden_size}')

# ── Load gating network ──────────────────────────────────────────────────────
gating_net = load_gating_network(
    script_args.checkpoint_path,
    lm_hidden_size=lm_hidden_size,
    num_experts=num_experts,
    block_mode=script_args.block_mode,
    device=device,
)
if gating_net is None:
    raise FileNotFoundError(f'No gating network found at {script_args.checkpoint_path}')
gating_net.eval()
if accelerator.is_main_process:
    print(f'Loaded gating network from {script_args.checkpoint_path}')

# ── Load datasets ────────────────────────────────────────────────────────────
dataset_df = pd.read_csv(script_args.dataset_csv)
rewards_df = pd.read_csv(script_args.rewards_csv)

if script_args.num_samples > 0:
    prompt_ids = dataset_df['prompt_idx'].unique()
    chosen     = np.random.choice(prompt_ids,
                                  size=min(script_args.num_samples, len(prompt_ids)),
                                  replace=False)
    dataset_df = dataset_df[dataset_df['prompt_idx'].isin(chosen)].reset_index(drop=True)
    rewards_df = rewards_df[rewards_df['prompt_idx'].isin(chosen)].reset_index(drop=True)

# Auto-detect reward names from CSV headers
pref_cols = [c for c in dataset_df.columns if c.startswith('pref_')]
detected  = [c[len('pref_'):] for c in pref_cols]
if detected != reward_names:
    print(f'Warning: reward_names arg={reward_names}, detected={detected}. Using detected.')
    reward_names = detected
    num_experts  = len(reward_names)

full_dataset = GatingDataset(dataset_df, reward_names, tokenizer,
                             block_mode=script_args.block_mode)

# Shard across GPUs
if accelerator.num_processes > 1:
    full_size = len(full_dataset)
    per_rank  = full_size // accelerator.num_processes
    start     = accelerator.process_index * per_rank
    end       = start + per_rank if accelerator.process_index < accelerator.num_processes - 1 \
                else full_size
    from torch.utils.data import Subset
    test_dataset = Subset(full_dataset, list(range(start, end)))
else:
    test_dataset = full_dataset

loader = DataLoader(test_dataset, batch_size=script_args.batch_size,
                    shuffle=False, drop_last=False)
if accelerator.is_main_process:
    print(f'Total dataset: {len(full_dataset)} items, this rank: {len(test_dataset)} items')

# ── Precompute per-prompt RBF surrogates from rewards_csv ────────────────────
if script_args.block_mode == 'uniform':
    w_cols = sorted([c for c in rewards_df.columns if c.startswith('w') and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
else:
    e_cols = sorted([c for c in rewards_df.columns if c.endswith('_early')],
                    key=lambda c: int(c[1:c.index('_')]))
    m_cols = sorted([c for c in rewards_df.columns if c.endswith('_mid')],
                    key=lambda c: int(c[1:c.index('_')]))
    l_cols = sorted([c for c in rewards_df.columns if c.endswith('_late')],
                    key=lambda c: int(c[1:c.index('_')]))
    w_cols = e_cols + m_cols + l_cols

r_cols = [f'reward_{n}' for n in reward_names]
print('Building per-prompt RBF surrogates ...')
reward_rbf = {}
for pidx in rewards_df['prompt_idx'].unique():
    sub = rewards_df[rewards_df['prompt_idx'] == pidx]
    X   = sub[w_cols].values.astype(np.float64)
    Y   = sub[r_cols].values.astype(np.float64)
    reward_rbf[int(pidx)] = [
        RBFInterpolator(X, Y[:, k], kernel='linear') for k in range(num_experts)
    ]
print(f'Built RBF for {len(reward_rbf)} prompts.')

# ── Run inference ────────────────────────────────────────────────────────────
n_w     = num_experts * 3 if script_args.block_mode == 'custom' else num_experts
records = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        if accelerator.is_main_process:
            print(f'Processing batch {batch_idx+1}/{len(loader)} ...', end='\r')

        input_ids        = batch['input_ids'].to(device)
        attention_mask   = batch['attention_mask'].to(device)
        preference       = batch['preference'].to(device)
        optimal_weights  = batch['optimal_weights'].to(device)
        prompt_idx_batch = batch['prompt_idx']

        prompt_hidden = get_prompt_hidden(expert_models, input_ids, attention_mask)
        pred_weights  = gating_net(prompt_hidden, preference)

        for b in range(input_ids.shape[0]):
            pref_np = preference[b].cpu().numpy()
            pred_w  = pred_weights[b].cpu().numpy()
            opt_w   = optimal_weights[b].cpu().numpy()
            pidx    = int(prompt_idx_batch[b])
            rbfs    = reward_rbf[pidx]

            # Naive baseline: preference used directly as merging weights
            naive_w = (np.concatenate([pref_np, pref_np, pref_np])
                       if script_args.block_mode == 'custom' else pref_np)

            def eval_r(w):
                return np.array([rbfs[k](w.reshape(1, -1))[0] for k in range(num_experts)])

            pred_r  = eval_r(pred_w)
            opt_r   = eval_r(opt_w)
            naive_r = eval_r(naive_w)

            pred_scalar  = float(pref_np @ pred_r)
            opt_scalar   = float(pref_np @ opt_r)
            naive_scalar = float(pref_np @ naive_r)
            weight_mae   = float(np.abs(pred_w - opt_w).mean())
            pref_str     = '(' + ', '.join(f'{v:.2f}' for v in pref_np) + ')'

            row = {
                'prompt_idx':    pidx,
                'pref_str':      pref_str,
                'weight_mae':    weight_mae,
                'pred_scalar':   pred_scalar,
                'opt_scalar':    opt_scalar,
                'naive_scalar':  naive_scalar,
                'pred_vs_opt':   pred_scalar  - opt_scalar,
                'pred_vs_naive': pred_scalar  - naive_scalar,
            }
            for k, name in enumerate(reward_names):
                row[f'pref_{name}']    = pref_np[k]
                row[f'pred_r_{name}']  = pred_r[k]
                row[f'opt_r_{name}']   = opt_r[k]
                row[f'naive_r_{name}'] = naive_r[k]
            for k in range(n_w):
                row[f'pred_w{k}'] = pred_w[k]
                row[f'opt_w{k}']  = opt_w[k]
            records.append(row)

# ── Gather results from all ranks ─────────────────────────────────────────────
shard_path = os.path.join(output_dir, f'test_results_rank{accelerator.process_index}.csv')
pd.DataFrame(records).to_csv(shard_path, index=False, escapechar='\\')
accelerator.wait_for_everyone()

if not accelerator.is_main_process:
    exit(0)

shards  = [pd.read_csv(os.path.join(output_dir, f'test_results_rank{r}.csv'))
           for r in range(accelerator.num_processes)]
results = pd.concat(shards, ignore_index=True)
for r in range(accelerator.num_processes):
    os.remove(os.path.join(output_dir, f'test_results_rank{r}.csv'))

# ── Print summary ─────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('OVERALL METRICS')
print('='*60)
print(f"Mean weight MAE             : {results['weight_mae'].mean():.4f}")
print(f"Mean pred_scalar            : {results['pred_scalar'].mean():.4f}")
print(f"Mean opt_scalar             : {results['opt_scalar'].mean():.4f}")
print(f"Mean naive_scalar           : {results['naive_scalar'].mean():.4f}")
print(f"pred vs opt   (gap, ↑bad)   : {results['pred_vs_opt'].mean():.4f}")
print(f"pred vs naive (gap, ↑good)  : {results['pred_vs_naive'].mean():.4f}")
pct_beat_naive = (results['pred_vs_naive'] > 0).mean() * 100
print(f"% prompts pred beats naive  : {pct_beat_naive:.1f}%")

print('\n' + '='*60)
print('PER-PREFERENCE BREAKDOWN')
print('='*60)
grp = results.groupby('pref_str').agg(
    weight_mae   =('weight_mae',    'mean'),
    pred_scalar  =('pred_scalar',   'mean'),
    opt_scalar   =('opt_scalar',    'mean'),
    naive_scalar =('naive_scalar',  'mean'),
    pred_vs_opt  =('pred_vs_opt',   'mean'),
    pred_vs_naive=('pred_vs_naive', 'mean'),
).round(4)
print(grp.to_string())

print('\n' + '='*60)
print('PER-REWARD DIMENSION')
print('='*60)
for name in reward_names:
    print(f"\n  {name}:")
    print(f"    mean pred_r : {results[f'pred_r_{name}'].mean():.4f}")
    print(f"    mean opt_r  : {results[f'opt_r_{name}'].mean():.4f}")
    print(f"    mean naive_r: {results[f'naive_r_{name}'].mean():.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
out_csv = os.path.join(output_dir, 'test_results.csv')
results.to_csv(out_csv, index=False, escapechar='\\')
print(f'\nFull results saved to {out_csv}')

grp.to_csv(os.path.join(output_dir, 'test_per_preference_summary.csv'))
print(f'Per-preference summary saved to {output_dir}')