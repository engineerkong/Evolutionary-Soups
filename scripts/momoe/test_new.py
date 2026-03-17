"""Test trained GatingNetwork predictions against ground truth in gating_dataset.csv using interpolation.

Outputs:
  - Per-preference mean predicted t vs mean optimal t
  - Per-prompt scatter: predicted t vs optimal t
  - Reward-space comparison: predicted reward vs optimal reward vs naive (t=pref[0]) reward
  - Summary metrics: MAE on t, MAE on reward, % improvement over naive
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_main_tokenizer
from new_architecture import GatingNetwork, chebyshev_optimal_weights, GatingDataset, get_prompt_hidden, SAMPLE_T_VALUES
from new_utils import load_base_model, load_gating_network


@dataclass
class ScriptArguments:
    sft_model_name: str = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    checkpoint_path: str = ''
    dataset_csv: str = './data/new/new_assistant/gating_dataset.csv'
    rewards_csv: str = './data/new/new_assistant/collected_rewards.csv'
    reward_names: str = 'harmless,helpful'
    save_directory: str = './results/new/'
    wandb_name: str = 'new_assistant_test'
    hidden_dim: int = 256
    batch_size: int = 128
    num_samples: int = 0   # 0 = use all, otherwise subsample for quick test


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
device = accelerator.device
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_experts = len(reward_names)

# ── Load tokenizer and frozen expert models ──────────────────────────────────
tokenizer = load_main_tokenizer(script_args.sft_model_name)
expert_models = []
for path in script_args.expert_model_paths:
    m = load_base_model(path, target_device=device)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    expert_models.append(m)

# Infer hidden size
with torch.no_grad():
    _dummy = tokenizer('hello', return_tensors='pt').to(device)
    _out = expert_models[0](**_dummy, output_hidden_states=True)
    lm_hidden_size = _out.hidden_states[-1].shape[-1]
print(f'lm_hidden_size = {lm_hidden_size}')

# ── Load gating network ──────────────────────────────────────────────────────
gating_net = load_gating_network(
    script_args.checkpoint_path,
    lm_hidden_size=lm_hidden_size,
    num_experts=num_experts,
    device=device,
)
if gating_net is None:
    raise FileNotFoundError(f'No gating network found at {script_args.checkpoint_path}')
gating_net.eval()
if accelerator.is_main_process:
    print(f'Loaded gating network from {script_args.checkpoint_path}')

# ── Load datasets ────────────────────────────────────────────────────────────
dataset_df = pd.read_csv(script_args.dataset_csv)
rewards_df  = pd.read_csv(script_args.rewards_csv)

if script_args.num_samples > 0:
    prompt_ids = dataset_df['prompt_idx'].unique()
    chosen = np.random.choice(prompt_ids,
                              size=min(script_args.num_samples, len(prompt_ids)),
                              replace=False)
    dataset_df = dataset_df[dataset_df['prompt_idx'].isin(chosen)].reset_index(drop=True)
    rewards_df  = rewards_df[rewards_df['prompt_idx'].isin(chosen)].reset_index(drop=True)

print(f'dataset_df columns:', dataset_df.columns.tolist())
print(f'dataset_df head:')
print(dataset_df.head(2).to_string())
print(f'rewards_df columns:', rewards_df.columns.tolist())
pref_cols = [c for c in dataset_df.columns if c.startswith('pref_')]
detected_reward_names = [c[len('pref_'):] for c in pref_cols]
if detected_reward_names != reward_names:
    print(f'Warning: reward_names arg={reward_names}, detected from CSV={detected_reward_names}')
    print('Using detected names from CSV.')
    reward_names = detected_reward_names

full_dataset = GatingDataset(dataset_df, rewards_df, reward_names, tokenizer)

# Shard dataset across GPUs — each rank processes a non-overlapping subset
if accelerator.num_processes > 1:
    full_dataset_size = len(full_dataset)
    per_rank = full_dataset_size // accelerator.num_processes
    start = accelerator.process_index * per_rank
    end   = start + per_rank if accelerator.process_index < accelerator.num_processes - 1 \
            else full_dataset_size
    indices = list(range(start, end))
    from torch.utils.data import Subset
    test_dataset = Subset(full_dataset, indices)
else:
    test_dataset = full_dataset

loader = DataLoader(test_dataset, batch_size=script_args.batch_size,
                    shuffle=False, drop_last=False)

if accelerator.is_main_process:
    print(f'Total dataset: {len(full_dataset)} items, '
          f'this rank: {len(test_dataset)} items')

# ── Build reward lookup for naive baseline ────────────────────────────────────
reward_lookup = {}
for _, row in rewards_df.iterrows():
    idx = int(row['prompt_idx'])
    t   = float(row['t_value'])
    rv  = [float(row[f'reward_{n}']) for n in reward_names]
    reward_lookup.setdefault(idx, {})[t] = rv

# ── Run inference ────────────────────────────────────────────────────────────
records = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        if accelerator.is_main_process:
            print(f'Processing batch {batch_idx+1}/{len(loader)}...', end='\r')

        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        preference     = batch['preference'].to(device)
        optimal_weights= batch['optimal_weights'].to(device)
        reward_matrix  = batch['reward_matrix'].to(device)

        prompt_hidden = get_prompt_hidden(expert_models, input_ids, attention_mask)
        pred_weights  = gating_net(prompt_hidden, preference)

        B = input_ids.shape[0]
        for b in range(B):
            pref_np   = preference[b].cpu().numpy()
            pred_w    = pred_weights[b].cpu().numpy()
            opt_w     = optimal_weights[b].cpu().numpy()
            rmat      = reward_matrix[b].cpu().numpy()

            pred_t  = float(pred_w[0])
            opt_t   = float(opt_w[0])
            naive_t = float(pref_np[0])

            pred_r  = np.array([np.interp(pred_t,  SAMPLE_T_VALUES, rmat[:, k])
                                 for k in range(num_experts)])
            opt_r   = np.array([np.interp(opt_t,   SAMPLE_T_VALUES, rmat[:, k])
                                 for k in range(num_experts)])
            naive_r = np.array([np.interp(naive_t, SAMPLE_T_VALUES, rmat[:, k])
                                 for k in range(num_experts)])

            pred_scalar  = float(pref_np @ pred_r)
            opt_scalar   = float(pref_np @ opt_r)
            naive_scalar = float(pref_np @ naive_r)

            row = {
                'pred_t':        pred_t,
                'opt_t':         opt_t,
                'naive_t':       naive_t,
                't_error':       abs(pred_t - opt_t),
                'pred_scalar':   pred_scalar,
                'opt_scalar':    opt_scalar,
                'naive_scalar':  naive_scalar,
                'pred_vs_opt':   pred_scalar  - opt_scalar,
                'pred_vs_naive': pred_scalar  - naive_scalar,
            }
            for k, name in enumerate(reward_names):
                row[f'pref_{name}']   = pref_np[k]
                row[f'pred_r_{name}'] = pred_r[k]
                row[f'opt_r_{name}']  = opt_r[k]
                row[f'naive_r_{name}']= naive_r[k]
                row[f'pred_w{k}']     = pred_w[k]
                row[f'opt_w{k}']      = opt_w[k]
            records.append(row)

# ── Gather results from all ranks onto rank 0 ────────────────────────────────
# Each rank saves its shard; rank 0 merges after barrier
shard_path = os.path.join(output_dir, f'test_results_rank{accelerator.process_index}.csv')
pd.DataFrame(records).to_csv(shard_path, index=False, escapechar='\\')

accelerator.wait_for_everyone()

if not accelerator.is_main_process:
    exit(0)

# Merge all rank shards
shards = [pd.read_csv(os.path.join(output_dir, f'test_results_rank{r}.csv'))
          for r in range(accelerator.num_processes)]
results = pd.concat(shards, ignore_index=True)
for r in range(accelerator.num_processes):
    os.remove(os.path.join(output_dir, f'test_results_rank{r}.csv'))

# ── Print summary ────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('OVERALL METRICS')
print('='*60)
print(f"Mean |pred_t - opt_t|     : {results['t_error'].mean():.4f}")
print(f"Mean pred_scalar          : {results['pred_scalar'].mean():.4f}")
print(f"Mean opt_scalar           : {results['opt_scalar'].mean():.4f}")
print(f"Mean naive_scalar         : {results['naive_scalar'].mean():.4f}")
print(f"pred vs opt  (gap, ↑bad)  : {results['pred_vs_opt'].mean():.4f}")
print(f"pred vs naive (gap, ↑good): {results['pred_vs_naive'].mean():.4f}")
pct_beat_naive = (results['pred_vs_naive'] > 0).mean() * 100
print(f"% prompts pred beats naive: {pct_beat_naive:.1f}%")

print('\n' + '='*60)
print('PER-PREFERENCE BREAKDOWN')
print('='*60)
pref_col = f'pref_{reward_names[0]}'
grp = results.groupby(pref_col).agg(
    pred_t       =('pred_t',        'mean'),
    opt_t        =('opt_t',         'mean'),
    naive_t      =('naive_t',       'mean'),
    t_error      =('t_error',       'mean'),
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