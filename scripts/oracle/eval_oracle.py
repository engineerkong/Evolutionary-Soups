"""Step 3: Read collected_rewards.csv and compute the oracle upper bound.

For each preference λ and each prompt, the oracle picks the best merging
coefficient:  oracle(λ, prompt) = max_w  λ · r(prompt, w)

Reports mean oracle utility vs naive (rewarded soups, w=λ) over all prompts,
showing how much headroom prompt-adaptive merging has.

Usage:
  python scripts/optimal/eval_oracle.py \
      --rewards_csv ./results/optimal/optimal_assistant/collected_rewards.csv \
      --reward_names harmless,helpful
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import HfArgumentParser

from optimal_utils import get_simplex_samples


@dataclass
class ScriptArguments:
    rewards_csv:    str   = './results/optimal/optimal_assistant/collected_rewards.csv'
    reward_names:   str   = 'harmless,helpful'
    simplex_step:   float = 0.1


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_rewards    = len(reward_names)

df = pd.read_csv(script_args.rewards_csv)
print(f'Loaded {len(df)} rows | {df["prompt_idx"].nunique()} prompts')

weight_cols  = sorted([c for c in df.columns if c.startswith('w') and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
reward_cols  = [f'reward_{n}' for n in reward_names]

# Normalize rewards to [0, 1] globally (same as build_dataset.py)
r_min = df[reward_cols].min().values
r_max = df[reward_cols].max().values
r_range = np.where(r_max > r_min, r_max - r_min, 1.0)
df[reward_cols] = (df[reward_cols].values - r_min) / r_range
print(f'Reward normalization: ' + ', '.join(f'{n}=[{lo:.3f},{hi:.3f}]' for n, lo, hi in zip(reward_names, r_min, r_max)))
pref_simplex = get_simplex_samples(n_rewards, step=script_args.simplex_step)

prompt_ids = sorted(df['prompt_idx'].unique())
n_prompts  = len(prompt_ids)

# reward_cube: (n_combos, n_prompts, n_rewards)
combos     = df[weight_cols].drop_duplicates().sort_values(weight_cols).values  # (C, n_experts)
n_combos   = len(combos)

reward_cube = np.zeros((n_combos, n_prompts, n_rewards))
for ci, combo in enumerate(combos):
    mask = np.all(df[weight_cols].values == combo, axis=1)
    sub  = df[mask].set_index('prompt_idx')
    for pi, pid in enumerate(prompt_ids):
        reward_cube[ci, pi, :] = sub.loc[pid, reward_cols].values

print(f'Reward cube: {n_combos} combos × {n_prompts} prompts × {n_rewards} rewards\n')

print(f'{"λ":<25}  {"naive":>8}  {"oracle":>8}  {"gain":>8}  {"gain%":>7}')
print('-' * 62)

for lam in pref_simplex:
    lam_arr = np.array(lam)

    # utilities: (C, N)
    utilities = reward_cube @ lam_arr

    # naive: combo whose weights == λ (exact match on same simplex grid)
    dist      = np.abs(combos - lam_arr).sum(axis=1)
    naive_idx = int(dist.argmin())
    naive_u   = float(utilities[naive_idx].mean())

    # oracle: per-prompt best combo
    oracle_u  = float(utilities.max(axis=0).mean())

    gain     = oracle_u - naive_u
    gain_pct = gain / abs(naive_u) * 100 if naive_u != 0 else float('nan')

    print(f'{str([round(v, 2) for v in lam]):<25}  {naive_u:>8.4f}  '
          f'{oracle_u:>8.4f}  {gain:>+8.4f}  {gain_pct:>6.1f}%')
