"""Step 4: Evaluate the trained GatingNetwork.
Three baselines on the test split per preference:
  - naive: t = round(pref[0], 1), full dataset
  - pred:  per-prompt weights from gating_net (rounded to 1 d.p., grouped by unique weight)
  - opt:   Chebyshev-optimal t from gating_dataset_test.csv (grouped by unique opt_t, full dataset per group)
"""
import gc
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import shutil

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser)
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
    get_clean_data, load_main_tokenizer, save_configs, sample_preferences_uniform,
)
from new_architecture import GatingNetwork, get_prompt_hidden, chebyshev_optimal_weights
from new_utils import load_base_model, load_gating_network, merge_and_save_weights

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}


@dataclass
class ScriptArguments:
    sft_model_name: str = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = field(default='')
    manual_expert_weights: Optional[str] = field(default='0.5,0.5')
    gating_dataset_test: str = field(default='',
        metadata={'help': 'gating_dataset_test.csv for opt_t lookup'})
    rewards_csv_test: str = field(default='',
        metadata={'help': 'collected_rewards.csv from test split for opt reward interpolation'})
    num_pref_samples: int = 10
    reward_names: str = 'harmless,helpful'
    exp_type: str = 'assistant'
    save_directory: str = './results/new/'
    wandb_name: str = 'new_assistant_eval'
    hidden_dim: int = 256


def parse_manual_weights(spec, num_experts):
    w = [float(v.strip()) for v in spec.split(',') if v.strip()]
    assert len(w) == num_experts
    s = sum(w)
    return [v / s for v in w]


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_experts = len(reward_names)
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(reward_model_paths, reward_model_paths, gpu_id)
save_configs({'sft_model_name': script_args.sft_model_name,
              'expert_model_paths': str(script_args.expert_model_paths)}, output_dir)
tokenizer = load_main_tokenizer(script_args.sft_model_name)

# ── Gating network ────────────────────────────────────────────────────────────
if script_args.checkpoint_path:
    _tmp = load_base_model(script_args.expert_model_paths[0], target_device=f'cuda:{gpu_id}')
    with torch.no_grad():
        _out = _tmp(input_ids=tokenizer('hello', return_tensors='pt').input_ids.to(f'cuda:{gpu_id}'),
                    output_hidden_states=True)
        lm_hidden_size = _out.hidden_states[-1].shape[-1]
    del _tmp; gc.collect(); torch.cuda.empty_cache()
    gating_net = load_gating_network(script_args.checkpoint_path, lm_hidden_size=lm_hidden_size,
                                     num_experts=num_experts, device=f'cuda:{gpu_id}')
else:
    gating_net = None

if gating_net is None:
    manual_weights = parse_manual_weights(script_args.manual_expert_weights, num_experts)

expert_models = []
if gating_net is not None:
    for path in script_args.expert_model_paths:
        m = load_base_model(path, target_device=f'cuda:{gpu_id}')
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        expert_models.append(m)

# ── opt_t lookup from gating_dataset_test ────────────────────────────────────
# key: (prompt_text, pref_tuple) -> optimal_t  (pre-computed by build_dataset.py)
opt_lookup = {}
if script_args.gating_dataset_test and os.path.exists(script_args.gating_dataset_test):
    _gdf = pd.read_csv(script_args.gating_dataset_test)
    _pref_cols = [c for c in _gdf.columns if c.startswith('pref_')]
    for _, _row in _gdf.iterrows():
        _key = (str(_row['prompt_text']).strip(),
                tuple(round(float(_row[c]), 4) for c in _pref_cols))
        opt_lookup[_key] = round(float(_row['optimal_t']), 1)
    print(f'Loaded {len(opt_lookup)} opt_t entries from {script_args.gating_dataset_test}')

# ── Test-split reward vectors for opt reward interpolation ──────────────────
reward_lookup = {}
if script_args.rewards_csv_test and os.path.exists(script_args.rewards_csv_test):
    _rdf   = pd.read_csv(script_args.rewards_csv_test)
    _rcols = [c for c in _rdf.columns if c.startswith('reward_')]
    for _, _row in _rdf.iterrows():
        _text = str(_row['prompt_text']).strip()
        _t    = float(_row['t_value'])
        reward_lookup.setdefault(_text, {})[_t] = [float(_row[c]) for c in _rcols]
    print(f'Loaded reward vectors for {len(reward_lookup)} prompts from {script_args.rewards_csv_test}')

def interp_opt_rewards(prompt_text, opt_t):
    rd = reward_lookup.get(prompt_text.strip())
    if rd is None:
        return None
    t_vals = sorted(rd.keys())
    rmat   = np.array([rd[t] for t in t_vals])
    return [float(np.interp(opt_t, t_vals, rmat[:, j])) for j in range(len(reward_names))]

# ── Dataset ───────────────────────────────────────────────────────────────────
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_ppo(
        'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_ppo(
        'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)
print(f'Eval dataset size: {len(valid_dataset)}')
sampled_preferences = sample_preferences_uniform(num_experts, script_args.num_pref_samples)


# ── Helpers ───────────────────────────────────────────────────────────────────
def predict_weights_all(preference):
    """Per-prompt rounded weights over full eval set. Returns List[tuple]."""
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(valid_dataset, batch_size=128, shuffle=False,
                        drop_last=False, collate_fn=collator)
    if gating_net is None:
        return [tuple(round(w, 1) for w in manual_weights)] * len(valid_dataset)
    pref_t = torch.tensor(preference, dtype=torch.float32, device=f'cuda:{gpu_id}')
    all_w = []
    gating_net.eval()
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(f'cuda:{gpu_id}')
            mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            w = gating_net(get_prompt_hidden(expert_models, ids, mask),
                           pref_t.unsqueeze(0).expand(ids.shape[0], -1)).cpu().numpy()
            for row in w:
                rounded = [round(float(v), 1) for v in row]
                s = sum(rounded)
                rounded = [v / s for v in rounded] if s > 0 else list(row)
                all_w.append(tuple(round(v, 1) for v in rounded))
    return all_w


def evaluate_model(temp_save_path, subset_indices=None):
    """Generate + score. Uses full dataset or subset_indices into valid_dataset.
    Uses drop_last=False so no batch is silently discarded."""
    dataset = valid_dataset.select(subset_indices) if subset_indices is not None \
              else valid_dataset
    if len(dataset) == 0:
        return [[] for _ in reward_names], [], []
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=128, drop_last=False, collate_fn=collator)
    model = AutoModelForCausalLM.from_pretrained(
        temp_save_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
    model.resize_token_embeddings(len(tokenizer))
    _acc = Accelerator()
    model, loader = _acc.prepare(model, loader)
    gen_kwargs = {
        'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
        'min_length': -1, 'top_k': 0.0, 'top_p': 0.9, 'do_sample': False,
    }
    tokenizer.padding_side = 'left'
    full_responses, full_prompts = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating', leave=False):
            out = _acc.unwrap_model(model).generate(
                batch['input_ids'], attention_mask=batch['attention_mask'], **gen_kwargs)
            full_responses.extend(out)
            full_prompts.extend(batch['input_ids'])
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts   = tokenizer.batch_decode(full_prompts)
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)
    qr = [(instructions.get_input(r), instructions.get_response(r)) for r in full_responses]
    rewards_list = reward_models.get_reward_model_scores(
        qr, instructions.get_post if hasattr(instructions, 'get_post') else None,
        normalize_rewards=False) if hasattr(instructions, 'get_post') else \
        reward_models.get_reward_model_scores(qr, normalize_rewards=False)
    all_rewards   = [_acc.gather_for_metrics(r) for r in rewards_list]
    all_prompts   = _acc.gather_for_metrics(full_prompts)
    all_responses = _acc.gather_for_metrics(full_responses)
    return all_rewards, all_prompts, all_responses


def run_grouped(prefix, groups, pref_k):
    """Merge+infer for each unique weight group. Returns {orig_idx: result_dict}."""
    results = {}
    for wt, indices in sorted(groups.items()):
        path = os.path.join(output_dir, f'temp_{prefix}_pref{pref_k}_w{"_".join(str(w) for w in wt)}')
        if process_id == 0:
            merge_and_save_weights(script_args.expert_model_paths, list(wt), path)
        accelerator.wait_for_everyone()
        gc.collect(); torch.cuda.empty_cache()
        r_rewards, r_prompts, r_responses = evaluate_model(path, indices)
        gc.collect(); torch.cuda.empty_cache()
        if process_id == 0:
            shutil.rmtree(path, ignore_errors=True)
            for i in range(len(r_prompts)):
                orig = indices[i] if i < len(indices) else indices[-1]
                results[orig] = {
                    'prompt': r_prompts[i], 'response': r_responses[i], 'w': list(wt),
                    **{f'reward_{reward_names[j]}': r_rewards[j][i]
                       for j in range(len(reward_names))},
                }
    return results


# ========== Main eval loop ==========
all_results = []

for k, preference in enumerate(sampled_preferences):
    print(f'\nPref {k+1}/{len(sampled_preferences)}: {[round(p,2) for p in preference]}')
    pref_key = tuple(round(float(p), 4) for p in preference)

    # ── naive ─────────────────────────────────────────────────────────────────
    naive_t = round(float(preference[0]), 1)
    naive_w = (naive_t, round(1.0 - naive_t, 1))
    naive_path = os.path.join(output_dir, f'temp_naive_pref{k}')
    if process_id == 0:
        merge_and_save_weights(script_args.expert_model_paths, list(naive_w), naive_path)
    accelerator.wait_for_everyone()
    gc.collect(); torch.cuda.empty_cache()
    n_rewards, n_prompts, n_responses = evaluate_model(naive_path)
    gc.collect(); torch.cuda.empty_cache()
    if process_id == 0:
        shutil.rmtree(naive_path, ignore_errors=True)
    print(f'  naive t={naive_t}, n={len(n_prompts)}')

    # ── pred ──────────────────────────────────────────────────────────────────
    per_prompt_w = predict_weights_all(preference)
    pred_groups = defaultdict(list)
    for idx, w in enumerate(per_prompt_w):
        pred_groups[w].append(idx)
    print(f'  pred: {len(pred_groups)} group(s): { {str(wt): len(v) for wt,v in sorted(pred_groups.items())} }')
    pred_by_idx = run_grouped('pred', pred_groups, k)

    # ── opt: interpolate rewards from rewards_csv_test (no inference needed) ───
    opt_by_idx = {}
    if opt_lookup and reward_lookup:
        print(f'all lookups available, performing opt reward interpolation...')
        for i, prompt_text in enumerate(n_prompts):
            opt_t = opt_lookup.get((prompt_text.strip(), pref_key), naive_t)
            opt_rewards = interp_opt_rewards(prompt_text, opt_t)
            if opt_rewards is not None:
                opt_by_idx[i] = {
                    'w': [round(opt_t, 1), round(1.0 - opt_t, 1)],
                    **{f'reward_{reward_names[j]}': opt_rewards[j] for j in range(len(reward_names))},
                }
        print(f'  opt: {len(opt_by_idx)}/{len(n_prompts)} prompts resolved via interpolation')

    # ── Assemble ──────────────────────────────────────────────────────────────
    if process_id == 0:
        rows = []
        for i, prompt_text in enumerate(n_prompts):
            pred = pred_by_idx.get(i) or next(
                (v for v in pred_by_idx.values() if v['prompt'] == prompt_text), None)
            opt  = opt_by_idx.get(i)
            row  = {
                'prompt':          prompt_text,
                'naive_t':         naive_t,
                'naive_response':  n_responses[i] if i < len(n_responses) else '',
                'pred_w':          pred['w']        if pred else float('nan'),
                'pred_response':   pred['response'] if pred else '',
                'opt_w':           opt['w']         if opt  else float('nan'),
            }
            for name in reward_names:
                row[f'naive_reward_{name}'] = n_rewards[reward_names.index(name)][i] \
                                              if i < len(n_rewards[0]) else float('nan')
                row[f'pred_reward_{name}']  = pred[f'reward_{name}'] if pred else float('nan')
                row[f'opt_reward_{name}']   = opt[f'reward_{name}']  if opt  else float('nan')
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir,
            f'eval_pref_{"_".join([str(round(p,2)) for p in preference])}.csv'), escapechar='\\')

        summary_row = {'pref_idx': k, 'naive_t': naive_t,
                       **{f'pref_{reward_names[j]}': preference[j] for j in range(num_experts)}}
        for name in reward_names:
            for tag in ['naive', 'pred', 'opt']:
                summary_row[f'mean_{tag}_reward_{name}'] = float(np.nanmean(df[f'{tag}_reward_{name}']))
            print(f'  {name:10s}  '
                  f'naive={summary_row[f"mean_naive_reward_{name}"]:.4f}  '
                  f'pred={summary_row[f"mean_pred_reward_{name}"]:.4f}  '
                  f'opt={summary_row[f"mean_opt_reward_{name}"]:.4f}')
        all_results.append(summary_row)

if process_id == 0:
    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(output_dir, 'eval_summary.csv'), index=False)
    print('\nEvaluation complete. Summary:')
    print(summary.to_string())