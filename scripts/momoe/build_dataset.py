"""Step 2: Read collected_rewards.csv, apply Chebyshev scalarization per prompt per
preference, and produce the supervised training dataset.

Output CSV columns:
    prompt_idx, prompt_text, pref_0, pref_1, ..., optimal_t, optimal_w0, optimal_w1, ...
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
from new_architecture import chebyshev_optimal_weights

SAMPLE_T_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class ScriptArguments:
    rewards_csv: str = './results/new/data/new_assistant/collected_rewards.csv'
    reward_names: str = 'harmless,helpful'
    num_pref_samples: int = 11
    save_directory: str = './results/new/data/'
    wandb_name: str = 'new_assistant'


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards = len(reward_names)
num_experts = num_rewards  # one expert per reward dimension

# Load collected rewards
df = pd.read_csv(script_args.rewards_csv)
print(f'Loaded {len(df)} rows, {df["prompt_idx"].nunique()} unique prompts')

# Sample preferences uniformly on the simplex
preferences = sample_preferences_uniform(num_rewards, script_args.num_pref_samples)
print(f'Using {len(preferences)} preferences:')
for p in preferences:
    print(f'  {[round(x, 2) for x in p]}')

# Build dataset: for each (prompt, preference) pair find Chebyshev optimal t
rows = []
prompt_ids = df['prompt_idx'].unique()

for prompt_idx in prompt_ids:
    sub = df[df['prompt_idx'] == prompt_idx].sort_values('t_value')

    # reward_matrix: (num_t_samples, num_rewards)
    reward_matrix = sub[[f'reward_{n}' for n in reward_names]].values
    t_values = sub['t_value'].values.tolist()
    prompt_text = sub['prompt_text'].iloc[0]

    # Verify we have all expected t values
    if len(t_values) != len(SAMPLE_T_VALUES):
        print(f'Warning: prompt {prompt_idx} has {len(t_values)} t values, skipping')
        continue

    for preference in preferences:
        opt_t, opt_weights = chebyshev_optimal_weights(reward_matrix, preference, t_values)

        row = {
            'prompt_idx': prompt_idx,
            'prompt_text': prompt_text,
            'optimal_t': opt_t,
        }
        for k, name in enumerate(reward_names):
            row[f'pref_{name}'] = preference[k]
        for k in range(num_experts):
            row[f'optimal_w{k}'] = opt_weights[k]

        rows.append(row)

dataset_df = pd.DataFrame(rows)
out_path = os.path.join(output_dir, 'gating_dataset.csv')
dataset_df.to_csv(out_path, index=False, escapechar='\\')

print(f'\nDataset saved to {out_path}')
print(f'Total training pairs: {len(dataset_df)}')
print(f'  ({df["prompt_idx"].nunique()} prompts × {len(preferences)} preferences)')
print('\nOptimal t distribution:')
print(dataset_df['optimal_t'].value_counts().sort_index())
print('\nMean optimal_t per preference:')
pref_col = f'pref_{reward_names[0]}'
print(dataset_df.groupby(pref_col)['optimal_t'].mean().round(3))
