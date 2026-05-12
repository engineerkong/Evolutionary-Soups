"""Step 2: Read collected_rewards.csv, find the utility-optimal merging weights per
prompt per preference, and produce the supervised training dataset.

Optimal weights are chosen by argmax of linear utility  sum_i(pref_i * reward_i)
evaluated directly on the real collected reward samples (no interpolation).

Output CSV columns:
    prompt_idx, prompt_text, pref_0, pref_1, ..., optimal_w0, optimal_w1, ...
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import HfArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimal_utils import get_simplex_samples, utility_optimal_weights


@dataclass
class ScriptArguments:
    rewards_csv:    str   = './results/optimal/optimal_assistant/collected_rewards.csv'
    reward_names:   str   = 'harmless,helpful'
    simplex_step:   float = 0.1
    save_directory: str   = './results/optimal/'
    wandb_name:     str   = 'optimal_assistant'


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards  = len(reward_names)

# Load collected rewards; drop duplicate header rows caused by re-runs
df = pd.read_csv(script_args.rewards_csv)
df = df[df['prompt_idx'].astype(str) != 'prompt_idx'].reset_index(drop=True)
reward_cols = [f'reward_{n}' for n in reward_names]
df[reward_cols] = df[reward_cols].astype(float)
print(f'Loaded {len(df)} rows, {df["prompt_idx"].nunique()} unique prompts')

weight_cols = sorted([c for c in df.columns if c.startswith('w') and c[1:].isdigit()],
                     key=lambda c: int(c[1:]))   # [w0, w1, ...]

preferences = get_simplex_samples(num_rewards, step=script_args.simplex_step)
print(f'Using {len(preferences)} preferences:')
for p in preferences:
    print(f'  {[round(x, 2) for x in p]}')

# Build dataset: for each (prompt, preference) pair find utility-optimal weights
rows       = []
prompt_ids = df['prompt_idx'].unique()

for prompt_idx in prompt_ids:
    progress = (np.where(prompt_ids == prompt_idx)[0][0] + 1) / len(prompt_ids) * 100
    print(f'Processing prompt {prompt_idx} ({progress:.1f}%)...')
    sub = df[df['prompt_idx'] == prompt_idx]

    reward_matrix  = sub[[f'reward_{n}' for n in reward_names]].values   # (S, R)
    prompt_text    = sub['prompt_text'].iloc[0]
    sample_weights = sub[weight_cols].values.tolist()                     # List[List[float]]

    if len(sample_weights) == 0:
        print(f'Warning: prompt {prompt_idx} has no samples, skipping')
        continue

    for preference in preferences:
        opt_w = utility_optimal_weights(reward_matrix, preference, sample_weights)

        row = {'prompt_idx': prompt_idx, 'prompt_text': prompt_text}
        for k, name in enumerate(reward_names):
            row[f'pref_{name}'] = preference[k]
        for k, w in enumerate(opt_w):
            row[f'optimal_w{k}'] = w
        rows.append(row)

dataset_df = pd.DataFrame(rows)
out_path   = os.path.join(output_dir, 'gating_dataset.csv')
dataset_df.to_csv(out_path, index=False, escapechar='\\')

print(f'\nDataset saved to {out_path}')
print(f'Total training pairs: {len(dataset_df)}')
print(f'  ({df["prompt_idx"].nunique()} prompts × {len(preferences)} preferences)')
print('\nMean optimal weights per preference:')
opt_cols = [f'optimal_w{k}' for k in range(num_rewards)]
pref_col = f'pref_{reward_names[0]}'
print(dataset_df.groupby(pref_col)[opt_cols].mean().round(3))
