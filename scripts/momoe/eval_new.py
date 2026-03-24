"""Step 4: Evaluate the trained GatingNetwork.
Three conditions per preference:
  naive : preference used directly as merging weights, full dataset.
  pred  : per-prompt weights from gating_net (rounded, grouped by unique weight combo).
  opt   : Chebyshev-optimal weights from gating_dataset_test.csv,
          rewards estimated via RBF from rewards_csv_test (no inference needed).

Efficiency: all predictions for all preferences are collected first, then every unique
weight combination is merged and evaluated ONCE. Results are distributed to all
(preference, prompt) pairs that share that combination.

Works for any number of objectives and both block_mode='uniform' | 'custom'.
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
from scipy.interpolate import RBFInterpolator
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
from new_architecture import GatingNetwork, get_prompt_hidden
from new_utils import (load_base_model, load_gating_network,
                       merge_and_save_weights, merge_and_save_weights_blockwise)

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
    sft_model_name:        str           = './models/sft/model/'
    expert_model_paths:    List[str]     = field(default_factory=list)
    checkpoint_path:       Optional[str] = field(default='')
    manual_expert_weights: Optional[str] = field(default=None)
    rewards_csv_test:      str           = field(default='',
        metadata={'help': 'collected_rewards.csv from test split for RBF reward estimation'})
    gating_dataset_test:   str           = field(default='',
        metadata={'help': 'gating_dataset_test.csv for opt_w lookup'})
    num_pref_samples:      int           = 6   # 6 for 2-obj, 27 for 3-obj grid
    reward_names:          str           = 'harmless,helpful'
    block_mode:            str           = 'uniform'   # 'uniform' | 'custom'
    exp_type:              str           = 'assistant'
    save_directory:        str           = './results/new/'
    wandb_name:            str           = 'new_assistant_eval'
    hidden_dim:            int           = 256


def _round_and_renorm(w, decimals=1):
    """Round weights to `decimals` d.p. and renormalise to sum=1."""
    r = [round(float(v), decimals) for v in w]
    s = sum(r)
    return [v / s for v in r] if s > 0 else list(w)


def parse_manual_weights(spec, num_experts):
    w = [float(v.strip()) for v in spec.split(',') if v.strip()]
    assert len(w) == num_experts
    s = sum(w)
    return [v / s for v in w]


# ── Setup ──────────────────────────────────────────────────────────────────────
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id

reward_names       = [x.strip() for x in script_args.reward_names.split(',')]
num_experts        = len(reward_names)
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, gpu_id)
save_configs({'sft_model_name': script_args.sft_model_name,
              'expert_model_paths': str(script_args.expert_model_paths)}, output_dir)
tokenizer = load_main_tokenizer(script_args.sft_model_name)

# ── Gating network ─────────────────────────────────────────────────────────────
if script_args.checkpoint_path:
    _tmp = load_base_model(script_args.expert_model_paths[0], target_device=f'cuda:{gpu_id}')
    with torch.no_grad():
        _out = _tmp(
            input_ids=tokenizer('hello', return_tensors='pt').input_ids.to(f'cuda:{gpu_id}'),
            output_hidden_states=True)
        lm_hidden_size = _out.hidden_states[-1].shape[-1]
    del _tmp; gc.collect(); torch.cuda.empty_cache()
    gating_net = load_gating_network(
        script_args.checkpoint_path, lm_hidden_size=lm_hidden_size,
        num_experts=num_experts, block_mode=script_args.block_mode,
        device=f'cuda:{gpu_id}')
else:
    gating_net = None

if gating_net is None:
    if script_args.manual_expert_weights:
        manual_weights = parse_manual_weights(script_args.manual_expert_weights, num_experts)
    else:
        manual_weights = [1.0 / num_experts] * num_experts

expert_models = []
if gating_net is not None:
    for path in script_args.expert_model_paths:
        m = load_base_model(path, target_device=f'cuda:{gpu_id}')
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        expert_models.append(m)

# ── opt_w lookup from gating_dataset_test ─────────────────────────────────────
# key: (prompt_text, pref_tuple) -> flat np.array of weights
opt_lookup = {}
if script_args.gating_dataset_test and os.path.exists(script_args.gating_dataset_test):
    _gdf       = pd.read_csv(script_args.gating_dataset_test)
    _pref_cols = [c for c in _gdf.columns if c.startswith('pref_')]
    for _, _row in _gdf.iterrows():
        _key = (str(_row['prompt_text']).strip(),
                tuple(round(float(_row[c]), 4) for c in _pref_cols))
        if script_args.block_mode == 'uniform':
            _opt_w = np.array([float(_row[f'optimal_w{k}'])
                               for k in range(num_experts)], dtype=np.float64)
        else:
            _opt_w = np.concatenate([
                [float(_row[f'optimal_w{k}_early']) for k in range(num_experts)],
                [float(_row[f'optimal_w{k}_mid'])   for k in range(num_experts)],
                [float(_row[f'optimal_w{k}_late'])  for k in range(num_experts)],
            ])
        opt_lookup[_key] = _opt_w
    print(f'Loaded {len(opt_lookup)} opt_w entries from {script_args.gating_dataset_test}')

# ── Per-prompt RBF surrogates from rewards_csv_test ───────────────────────────
reward_rbf = {}   # prompt_text -> list of RBFInterpolator (one per reward dim)
if script_args.rewards_csv_test and os.path.exists(script_args.rewards_csv_test):
    _rdf = pd.read_csv(script_args.rewards_csv_test)
    if script_args.block_mode == 'uniform':
        _w_cols = sorted([c for c in _rdf.columns if c.startswith('w') and c[1:].isdigit()],
                         key=lambda c: int(c[1:]))
    else:
        _e = sorted([c for c in _rdf.columns if c.endswith('_early')],
                    key=lambda c: int(c[1:c.index('_')]))
        _m = sorted([c for c in _rdf.columns if c.endswith('_mid')],
                    key=lambda c: int(c[1:c.index('_')]))
        _l = sorted([c for c in _rdf.columns if c.endswith('_late')],
                    key=lambda c: int(c[1:c.index('_')]))
        _w_cols = _e + _m + _l
    _r_cols = [f'reward_{n}' for n in reward_names]
    for _text, _sub in _rdf.groupby('prompt_text'):
        _X = _sub[_w_cols].values.astype(np.float64)
        _Y = _sub[_r_cols].values.astype(np.float64)
        reward_rbf[str(_text).strip()] = [
            RBFInterpolator(_X, _Y[:, k], kernel='linear') for k in range(num_experts)
        ]
    print(f'Built RBF for {len(reward_rbf)} prompts from {script_args.rewards_csv_test}')


def get_opt_reward(prompt_text, opt_w_flat):
    rbfs = reward_rbf.get(prompt_text.strip())
    if rbfs is None:
        return None
    return [float(rbfs[k](opt_w_flat.reshape(1, -1))[0]) for k in range(num_experts)]


# ── Dataset ────────────────────────────────────────────────────────────────────
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
N_DATASET = len(valid_dataset)
sampled_preferences = sample_preferences_uniform(num_experts, script_args.num_pref_samples)


# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_weights_all(preference):
    """Return per-prompt predicted weight tuples for the full eval dataset."""
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(valid_dataset, batch_size=128, shuffle=False,
                          drop_last=False, collate_fn=collator)
    if gating_net is None:
        flat = tuple(_round_and_renorm(manual_weights))
        if script_args.block_mode == 'uniform':
            return [flat] * N_DATASET
        else:
            return [(flat, flat, flat)] * N_DATASET

    pref_t = torch.tensor(preference, dtype=torch.float32, device=f'cuda:{gpu_id}')
    all_w  = []
    gating_net.eval()
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(f'cuda:{gpu_id}')
            mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            w    = gating_net(get_prompt_hidden(expert_models, ids, mask),
                              pref_t.unsqueeze(0).expand(ids.shape[0], -1)).cpu().numpy()
            for row in w:
                if script_args.block_mode == 'uniform':
                    all_w.append(tuple(_round_and_renorm(row)))
                else:
                    n = num_experts
                    early = tuple(_round_and_renorm(row[:n]))
                    mid   = tuple(_round_and_renorm(row[n:2*n]))
                    late  = tuple(_round_and_renorm(row[2*n:]))
                    all_w.append((early, mid, late))
    return all_w


def _merge_for_group(wt, path):
    if script_args.block_mode == 'uniform':
        merge_and_save_weights(script_args.expert_model_paths, list(wt), path)
    else:
        early_w, mid_w, late_w = wt
        merge_and_save_weights_blockwise(
            script_args.expert_model_paths,
            list(early_w), list(mid_w), list(late_w), path)


def _wt_to_str(wt):
    if script_args.block_mode == 'uniform':
        return '_'.join(f'{v:.2f}' for v in wt)
    else:
        e, m, l = wt
        return ('E' + '_'.join(f'{v:.2f}' for v in e) +
                '_M' + '_'.join(f'{v:.2f}' for v in m) +
                '_L' + '_'.join(f'{v:.2f}' for v in l))


def evaluate_model(temp_save_path, subset_indices=None):
    """Generate + score on full dataset or a subset. Returns (rewards, prompts, responses)."""
    dataset = valid_dataset.select(subset_indices) if subset_indices is not None \
              else valid_dataset
    if len(dataset) == 0:
        return [[] for _ in reward_names], [], []
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(dataset, batch_size=128, drop_last=False, collate_fn=collator)
    model    = AutoModelForCausalLM.from_pretrained(
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
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(
            qr, instructions.get_post, normalize_rewards=False)
    else:
        rewards_list = reward_models.get_reward_model_scores(qr, normalize_rewards=False)
    all_rewards   = [_acc.gather_for_metrics(r) for r in rewards_list]
    all_prompts   = _acc.gather_for_metrics(full_prompts)
    all_responses = _acc.gather_for_metrics(full_responses)
    return all_rewards, all_prompts, all_responses


# =============================================================================
# Phase 1 — Predict weights for ALL preferences (gating_net forward passes only)
# =============================================================================
print('\n' + '='*60)
print('Phase 1: Predicting weights for all preferences')
print('='*60)

naive_display_w_by_k = {}   # k -> display tuple (uniform weights)
naive_merge_w_by_k   = {}   # k -> wt key used for merging
pred_groups_by_k     = {}   # k -> {wt: [dataset_indices]}
idx_to_pred_wt       = {}   # k -> {dataset_idx: wt}

for k, preference in enumerate(sampled_preferences):
    print(f'  Pref {k+1}/{len(sampled_preferences)}: {[round(p, 2) for p in preference]}')
    naive_w = tuple(_round_and_renorm(preference))
    naive_display_w_by_k[k] = naive_w
    naive_merge_w_by_k[k]   = naive_w if script_args.block_mode == 'uniform' \
                               else (naive_w, naive_w, naive_w)

    per_prompt_w = predict_weights_all(preference)
    groups = defaultdict(list)
    for idx, w in enumerate(per_prompt_w):
        groups[w].append(idx)
    pred_groups_by_k[k] = dict(groups)

    idx_to_pred_wt[k] = {}
    for wt, indices in groups.items():
        for idx in indices:
            idx_to_pred_wt[k][idx] = wt

    print(f'    naive={[round(v,2) for v in naive_w]} | '
          f'pred: {len(groups)} unique group(s)')

# =============================================================================
# Phase 2 — Collect all unique weight combinations across all preferences
# =============================================================================
print('\n' + '='*60)
print('Phase 2: Building unified weight map')
print('='*60)

# wts that require full-dataset evaluation (naive weights)
wt_is_full = {}                        # wt -> True
# wts that only need a subset (pred weights not used as any naive)
wt_pred_indices = defaultdict(set)    # wt -> set of dataset indices

for k, merge_w in naive_merge_w_by_k.items():
    wt_is_full[merge_w] = True

for k, groups in pred_groups_by_k.items():
    for wt, indices in groups.items():
        if not wt_is_full.get(wt, False):
            wt_pred_indices[wt].update(indices)

all_unique_wts = sorted(
    set(wt_is_full.keys()) | set(wt_pred_indices.keys()), key=_wt_to_str)

n_full = sum(wt_is_full.values())
n_pred = len(wt_pred_indices)
print(f'  {len(all_unique_wts)} unique weight combos '
      f'({n_full} full-dataset + {n_pred} pred-subset)')

# =============================================================================
# Phase 3 — Merge + evaluate ONCE per unique weight combination
# =============================================================================
print('\n' + '='*60)
print('Phase 3: Merging and evaluating each unique weight combination')
print('='*60)

# wt_result_by_idx[wt][dataset_idx] = {prompt, response, reward_X, ...}
wt_result_by_idx = {}
all_prompts_text = None   # populated from first full-dataset evaluation

for wt in all_unique_wts:
    is_full = wt_is_full.get(wt, False)
    subset  = None if is_full else sorted(wt_pred_indices[wt])
    n_valid = N_DATASET if is_full else len(subset)
    print(f'\n  [{_wt_to_str(wt)}]  '
          f'{"full dataset" if is_full else f"{n_valid} prompts"}')

    path = os.path.join(output_dir, f'temp_wt_{_wt_to_str(wt)}')
    if process_id == 0:
        _merge_for_group(wt, path)
    accelerator.wait_for_everyone()
    gc.collect(); torch.cuda.empty_cache()

    r_rewards, r_prompts, r_responses = evaluate_model(path, subset)
    gc.collect(); torch.cuda.empty_cache()

    if process_id == 0:
        shutil.rmtree(path, ignore_errors=True)
        mapping = {}
        for local_i in range(min(n_valid, len(r_prompts))):
            orig_i = local_i if is_full else subset[local_i]
            mapping[orig_i] = {
                'prompt':   r_prompts[local_i],
                'response': r_responses[local_i],
                **{f'reward_{reward_names[j]}': r_rewards[j][local_i]
                   for j in range(num_experts)},
            }
        wt_result_by_idx[wt] = mapping

        if all_prompts_text is None and is_full:
            all_prompts_text = [mapping[i]['prompt']
                                for i in range(min(n_valid, len(r_prompts)))]

# =============================================================================
# Phase 4 — Opt rewards via RBF (no model merging)
# =============================================================================
print('\n' + '='*60)
print('Phase 4: Computing opt rewards via RBF')
print('='*60)

opt_by_k = {}   # k -> {dataset_idx: {reward_X, w}}
if opt_lookup and reward_rbf and all_prompts_text is not None:
    for k, preference in enumerate(sampled_preferences):
        pref_key  = tuple(round(float(p), 4) for p in preference)
        opt_by_k[k] = {}
        for i, prompt_text in enumerate(all_prompts_text):
            opt_w_flat = opt_lookup.get((prompt_text.strip(), pref_key))
            if opt_w_flat is None:
                continue
            opt_r = get_opt_reward(prompt_text, opt_w_flat)
            if opt_r is not None:
                opt_by_k[k][i] = {
                    'w': opt_w_flat.tolist(),
                    **{f'reward_{reward_names[j]}': opt_r[j] for j in range(num_experts)},
                }
    resolved = {k: len(v) for k, v in opt_by_k.items()}
    print(f'  Opt rewards resolved per pref: '
          f'{[f"k{k}:{n}" for k, n in resolved.items()]}')

# =============================================================================
# Phase 5 — Assemble per-preference results and save
# =============================================================================
print('\n' + '='*60)
print('Phase 5: Assembling results')
print('='*60)

all_results = []

if process_id == 0 and all_prompts_text is not None:
    N = len(all_prompts_text)
    for k, preference in enumerate(sampled_preferences):
        naive_display_w = naive_display_w_by_k[k]
        naive_results   = wt_result_by_idx.get(naive_merge_w_by_k[k], {})
        rows = []
        for i in range(N):
            naive_res = naive_results.get(i, {})
            pred_wt   = idx_to_pred_wt[k].get(i)
            pred_res  = wt_result_by_idx.get(pred_wt, {}).get(i, {}) if pred_wt else {}
            opt_res   = opt_by_k.get(k, {}).get(i, {})

            row = {
                'prompt':         naive_res.get('prompt', ''),
                'naive_w':        list(naive_display_w),
                'naive_response': naive_res.get('response', ''),
                'pred_w':         list(pred_wt) if pred_wt else float('nan'),
                'pred_response':  pred_res.get('response', ''),
                'opt_w':          opt_res.get('w', float('nan')),
            }
            for name in reward_names:
                row[f'naive_reward_{name}'] = naive_res.get(f'reward_{name}', float('nan'))
                row[f'pred_reward_{name}']  = pred_res.get(f'reward_{name}', float('nan'))
                row[f'opt_reward_{name}']   = opt_res.get(f'reward_{name}',  float('nan'))
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir,
            f'eval_pref_{"_".join([str(round(p, 2)) for p in preference])}.csv'),
            escapechar='\\', index=False)

        summary_row = {
            'pref_idx': k,
            **{f'pref_{reward_names[j]}': preference[j] for j in range(num_experts)},
        }
        print(f'\nPref {k+1}: {[round(p,2) for p in preference]}')
        for name in reward_names:
            for tag in ['naive', 'pred', 'opt']:
                summary_row[f'mean_{tag}_reward_{name}'] = float(
                    np.nanmean(df[f'{tag}_reward_{name}']))
            print(f'  {name:10s}  '
                  f'naive={summary_row[f"mean_naive_reward_{name}"]:.4f}  '
                  f'pred={summary_row[f"mean_pred_reward_{name}"]:.4f}  '
                  f'opt={summary_row[f"mean_opt_reward_{name}"]:.4f}')
        all_results.append(summary_row)

    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(output_dir, 'eval_summary.csv'), index=False)
    print('\nEvaluation complete. Summary:')
    print(summary.to_string())
