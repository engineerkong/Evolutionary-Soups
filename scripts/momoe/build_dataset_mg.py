"""Step 2: Read collected_rewards.csv, apply Chebyshev scalarization per prompt per
preference, and produce the supervised training dataset.

Supports both:
  uniform   — one merging weight vector per sample (block_mode='uniform')
  blockwise — three independent weight vectors per sample (block_mode='custom')

Output CSV columns (uniform):
    prompt_idx, prompt_text, pref_0, pref_1, ..., optimal_w0, optimal_w1, ...

Output CSV columns (blockwise):
    prompt_idx, prompt_text, pref_0, pref_1, ...,
    optimal_w0_early, optimal_w1_early, optimal_w0_mid, ..., optimal_w0_late, ...
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import HfArgumentParser

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import sample_preferences_uniform

def chebyshev_optimal_weights(
    reward_matrix,
    preference,
    sample_weights,
    n_interp=1000,
):
    preference    = np.array(preference,    dtype=np.float64)
    reward_matrix = np.array(reward_matrix, dtype=np.float64)
    r_star        = reward_matrix.max(axis=0)

    blockwise = isinstance(sample_weights[0], tuple)
    n_experts = len(sample_weights[0][0]) if blockwise else len(sample_weights[0])
    rng       = np.random.default_rng(seed=42)

    if not blockwise:
        if n_experts == 2:
            # --- 2 experts: exact 1D linear interpolation ---
            observed_t    = np.array([w[0] for w in sample_weights])
            sort_idx      = np.argsort(observed_t)
            observed_t    = observed_t[sort_idx]
            reward_sorted = reward_matrix[sort_idx]

            fine_t       = np.linspace(observed_t[0], observed_t[-1], n_interp)
            fine_rewards = np.stack([
                np.interp(fine_t, observed_t, reward_sorted[:, i])
                for i in range(reward_matrix.shape[1])
            ], axis=1)                                       # (n_interp, num_rewards)

            gaps      = preference * np.abs(fine_rewards - r_star)
            utilities = -gaps.max(axis=1)
            best_idx  = int(np.argmax(utilities))
            optimal_t = float(fine_t[best_idx])
            return [optimal_t, 1.0 - optimal_t]

        else:
            # --- n_experts>2: RBF surrogate over the full simplex ---
            X_observed = np.array(sample_weights)            # (num_observed, n_experts)
            fine_w     = rng.dirichlet(np.ones(n_experts), size=n_interp)
                                                             # (n_interp, n_experts)

    else:
        # --- blockwise: RBF surrogate over flattened (early, mid, late) ---
        X_observed = np.array([
            list(e) + list(m) + list(l)
            for e, m, l in sample_weights
        ])                                                   # (num_observed, n_experts*3)

        fine_early = rng.dirichlet(np.ones(n_experts), size=n_interp)
        fine_mid   = rng.dirichlet(np.ones(n_experts), size=n_interp)
        fine_late  = rng.dirichlet(np.ones(n_experts), size=n_interp)
        fine_w     = np.concatenate([fine_early, fine_mid, fine_late], axis=1)
                                                             # (n_interp, n_experts*3)

    # --- shared RBF path for n_experts>2 uniform and blockwise ---
    from scipy.interpolate import RBFInterpolator
    fine_rewards = np.stack([
        RBFInterpolator(X_observed, reward_matrix[:, i], kernel='linear')(fine_w)
        for i in range(reward_matrix.shape[1])
    ], axis=1)                                               # (n_interp, num_rewards)

    gaps      = preference * np.abs(fine_rewards - r_star)
    utilities = -gaps.max(axis=1)
    best_idx  = int(np.argmax(utilities))

    if not blockwise:
        return fine_w[best_idx].tolist()
    else:
        return (
            fine_early[best_idx].tolist(),
            fine_mid[best_idx].tolist(),
            fine_late[best_idx].tolist(),
        )


@dataclass
class ScriptArguments:
    rewards_csv:      str = './results/new/data/new_assistant/collected_rewards.csv'
    reward_names:     str = 'harmless,helpful'
    num_pref_samples: int = 11
    save_directory:   str = './results/new/'
    wandb_name:       str = 'new_assistant'
    block_mode:       str = 'uniform'   # 'uniform' | 'custom'


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards  = len(reward_names)
num_experts  = num_rewards

# Load collected rewards
df = pd.read_csv(script_args.rewards_csv)
print(f'Loaded {len(df)} rows, {df["prompt_idx"].nunique()} unique prompts')

# Infer weight columns directly from CSV headers
if script_args.block_mode == 'uniform':
    weight_cols = sorted([c for c in df.columns if c.startswith('w') and c[1:].isdigit()],
                         key=lambda c: int(c[1:]))                   # [w0, w1, ...]
else:
    early_cols = sorted([c for c in df.columns if c.endswith('_early')],
                        key=lambda c: int(c[1:c.index('_')]))
    mid_cols   = sorted([c for c in df.columns if c.endswith('_mid')],
                        key=lambda c: int(c[1:c.index('_')]))
    late_cols  = sorted([c for c in df.columns if c.endswith('_late')],
                        key=lambda c: int(c[1:c.index('_')]))

# Sample preferences uniformly on the simplex
preferences = sample_preferences_uniform(num_rewards, script_args.num_pref_samples)
print(f'Using {len(preferences)} preferences:')
for p in preferences:
    print(f'  {[round(x, 2) for x in p]}')

# Build dataset: for each (prompt, preference) pair find Chebyshev optimal weights
rows = []
prompt_ids = df['prompt_idx'].unique()

for prompt_idx in prompt_ids:
    sub = df[df['prompt_idx'] == prompt_idx]

    # reward_matrix: (num_samples, num_rewards)
    reward_matrix = sub[[f'reward_{n}' for n in reward_names]].values
    prompt_text   = sub['prompt_text'].iloc[0]

    # Reconstruct sample_weights in the correct format
    if script_args.block_mode == 'uniform':
        sample_weights = sub[weight_cols].values.tolist()       # List[List[float]]
    else:
        sample_weights = [                                       # List[Tuple[List,List,List]]
            (row[early_cols].tolist(), row[mid_cols].tolist(), row[late_cols].tolist())
            for _, row in sub.iterrows()
        ]

    # Verify we have samples
    if len(sample_weights) == 0:
        print(f'Warning: prompt {prompt_idx} has no samples, skipping')
        continue

    for preference in preferences:
        opt_w = chebyshev_optimal_weights(reward_matrix, preference, sample_weights)

        row = {
            'prompt_idx':  prompt_idx,
            'prompt_text': prompt_text,
        }
        for k, name in enumerate(reward_names):
            row[f'pref_{name}'] = preference[k]

        if script_args.block_mode == 'uniform':
            for k, w in enumerate(opt_w):                       # opt_w is List[float]
                row[f'optimal_w{k}'] = w
        else:
            early_w, mid_w, late_w = opt_w                      # opt_w is Tuple[List,List,List]
            for k, w in enumerate(early_w):
                row[f'optimal_w{k}_early'] = w
            for k, w in enumerate(mid_w):
                row[f'optimal_w{k}_mid'] = w
            for k, w in enumerate(late_w):
                row[f'optimal_w{k}_late'] = w

        rows.append(row)

dataset_df = pd.DataFrame(rows)
out_path   = os.path.join(output_dir, 'gating_dataset.csv')
dataset_df.to_csv(out_path, index=False, escapechar='\\')

print(f'\nDataset saved to {out_path}')
print(f'Total training pairs: {len(dataset_df)}')
print(f'  ({df["prompt_idx"].nunique()} prompts × {len(preferences)} preferences)')
print('\nMean optimal weights per preference:')
opt_cols = ([f'optimal_w{k}' for k in range(num_experts)] if script_args.block_mode == 'uniform'
            else [f'optimal_w{k}_{b}' for b in ('early', 'mid', 'late')
                  for k in range(num_experts)])
pref_col = f'pref_{reward_names[0]}'
print(dataset_df.groupby(pref_col)[opt_cols].mean().round(3))