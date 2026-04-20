"""Step 4: Evaluate the trained GatingNetwork.
Three conditions per preference:
  naive : preference used directly as merging weights, full dataset.
  pred  : per-prompt weights from gating_net (rounded, grouped by unique weight combo).
  opt   : utility-optimal weights from gating_dataset_test.csv,
          rewards from actual model inference (no RBF estimation).

Efficiency: all predictions for all preferences are collected first, then every unique
weight combination is merged and evaluated ONCE. Results are distributed to all
(preference, prompt) pairs that share that combination.

Works for any number of objectives and both block_mode='uniform' | 'custom'.
"""
import gc
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import datetime

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, HfArgumentParser)
from trl import set_seed
from peft import PeftModel

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo, build_dataset_ppo, build_dataset_summary_ppo,
    get_clean_data, load_main_tokenizer, save_configs, sample_preferences_uniform,
)
from new_architecture import GatingNetwork, get_prompt_hidden, get_prompt_hidden_from_reward_models
from new_utils import (
    EARLY_FRAC, LATE_FRAC,
    merge_and_save_weights, merge_and_save_weights_blockwise,
    load_lora_adapters, apply_merged_lora, apply_merged_lora_blockwise,
    load_base_model, load_gating_network,
)

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}

# ── Eval Helpers ────────────────────────────────────────────────────────────────────
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

def predict_weights_all(preference):
    """Return per-prompt predicted weight tuples for the full eval dataset."""
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(valid_dataset, batch_size=64, shuffle=False,
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
            if script_args.use_reward_features:
                prompt_texts  = tokenizer.batch_decode(ids, skip_special_tokens=True)
                prompt_hidden = get_prompt_hidden_from_reward_models(
                    feature_models, feature_tokenizers, prompt_texts, f'cuda:{gpu_id}')
            else:
                prompt_hidden = get_prompt_hidden(feature_models, ids, mask)
            w = gating_net(prompt_hidden,
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


def _wt_to_str(wt):
    if script_args.block_mode == 'uniform':
        return '_'.join(f'{v:.2f}' for v in wt)
    else:
        e, m, l = wt
        return ('E' + '_'.join(f'{v:.2f}' for v in e) +
                '_M' + '_'.join(f'{v:.2f}' for v in m) +
                '_L' + '_'.join(f'{v:.2f}' for v in l))


def evaluate_model(subset_indices=None):
    """Generate + score on full dataset or a subset. Returns (rewards, prompts, responses, orig_indices).
    Expects base_model weights to already be patched by _apply_weights_for_group.

    When do_sample=True, runs num_continuations independent stochastic passes and
    averages rewards across them (GRPO / RLOO style). When do_sample=False, greedy
    decoding is deterministic so a single pass is used regardless of num_continuations.
    """
    dataset = valid_dataset.select(subset_indices) if subset_indices is not None \
              else valid_dataset
    if len(dataset) == 0:
        return [[] for _ in reward_names], [], [], []

    # Add original index column so we can recover order after gather
    orig_indices_col = list(range(len(dataset)))
    dataset = dataset.add_column("orig_idx", orig_indices_col)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(dataset, batch_size=64, drop_last=False, collate_fn=collator)
    loader   = accelerator.prepare(loader)

    gen_kwargs = {
        'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
        'min_length': -1,
        'top_k': 0.0,
        'top_p': 0.9,
        'temperature': 0.7,
        'do_sample': script_args.do_sample,
    }
    tokenizer.padding_side = 'left'

    n_continuations = script_args.num_continuations if script_args.do_sample else 1

    # accumulated_rewards[prompt_idx][obj_idx] = list of K reward values
    accumulated_rewards = None
    saved_prompts       = None
    saved_orig_indices  = None

    base_model.eval()
    for cont_idx in range(n_continuations):
        full_responses, full_prompts, full_orig_indices = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader,
                              desc=f'Generating cont={cont_idx+1}/{n_continuations}',
                              leave=False):
                out = accelerator.unwrap_model(base_model).generate(
                    batch['input_ids'], attention_mask=batch['attention_mask'], **gen_kwargs)
                full_responses.extend(out)
                full_prompts.extend(batch['input_ids'])
                full_orig_indices.extend(batch['orig_idx'].tolist())

        full_responses = tokenizer.batch_decode(full_responses)
        full_prompts   = tokenizer.batch_decode(full_prompts)
        full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

        qr = [(instructions.get_input(r), instructions.get_response(r)) for r in full_responses]
        if hasattr(instructions, 'get_post'):
            rewards_list = reward_models.get_reward_model_scores(
                qr, instructions.get_post, normalize_rewards=False)
        else:
            rewards_list = reward_models.get_reward_model_scores(qr, normalize_rewards=False)

        all_rewards      = [accelerator.gather_for_metrics(r) for r in rewards_list]
        all_prompts_g    = accelerator.gather_for_metrics(full_prompts)
        all_responses_g  = accelerator.gather_for_metrics(full_responses)
        all_orig_idx_g   = accelerator.gather_for_metrics(
            torch.tensor(full_orig_indices, dtype=torch.long, device=f'cuda:{gpu_id}')
        ).cpu().tolist()

        n_prompts = len(all_orig_idx_g)
        n_obj     = len(reward_names)
        if accumulated_rewards is None:
            accumulated_rewards = [[[] for _ in range(n_obj)] for _ in range(n_prompts)]
            saved_prompts      = all_prompts_g
            saved_orig_indices = all_orig_idx_g

        for idx in range(n_prompts):
            for k in range(n_obj):
                accumulated_rewards[idx][k].append(all_rewards[k][idx])

        torch.cuda.empty_cache()

    # Average rewards across continuations
    avg_rewards = [
        [float(np.mean(accumulated_rewards[idx][k])) for idx in range(len(saved_orig_indices))]
        for k in range(len(reward_names))
    ]
    # Return last continuation's responses (representative sample)
    return avg_rewards, saved_prompts, all_responses_g, saved_orig_indices

@dataclass
class ScriptArguments:
    sft_model_name:        str           = './models/sft/model/'
    expert_model_paths:    List[str]     = field(default_factory=list)
    checkpoint_path:       Optional[str] = field(default='')
    manual_expert_weights: Optional[str] = field(default=None)
    dataset_csv_test:   str           = field(default='',
        metadata={'help': 'gating_dataset_test.csv for opt_w lookup'})
    num_pref_samples:      int           = 6   # 6/11 for 2-obj, 21/66 for 3-obj grid
    reward_names:          str           = 'harmless,helpful'
    block_mode:            str           = 'uniform'   # 'uniform' | 'custom'
    exp_type:              str           = 'assistant'
    hidden_dim:            int           = 256
    use_reward_features:   bool          = False
    use_lora:              bool          = False  # True → in-memory LoRA swap
    do_sample:             bool          = False  # passed to generate(); True → stochastic, use num_continuations
    num_continuations:     int           = 3      # K passes averaged when do_sample=True; forced to 1 if do_sample=False
    save_directory:        str           = './results/new/'
    wandb_name:            str           = 'new_assistant_eval'


# ── Setup ──────────────────────────────────────────────────────────────────────
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
if 'RANK' in os.environ:
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
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
    if script_args.use_reward_features:
        # Detect lm_hidden_size = sum of each reward model's hidden size (concatenated)
        lm_hidden_size = 0
        with torch.no_grad():
            for _path in reward_model_paths:
                _m   = AutoModelForSequenceClassification.from_pretrained(
                    _path, device_map=f'cuda:{gpu_id}')
                _tok = AutoTokenizer.from_pretrained(_path)
                _d   = _tok('hello', return_tensors='pt').to(f'cuda:{gpu_id}')
                _out = _m(**_d, output_hidden_states=True)
                if (hasattr(_out, 'encoder_last_hidden_state')
                        and _out.encoder_last_hidden_state is not None):
                    lm_hidden_size += _out.encoder_last_hidden_state.shape[-1]
                else:
                    lm_hidden_size += _out.hidden_states[-1].shape[-1]
                del _m; gc.collect(); torch.cuda.empty_cache()
    else:
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

# Load feature models for gating network hidden-state extraction
feature_models      = []
feature_tokenizers  = []
expert_models       = []   # kept for API compatibility (generation uses merged models)
if gating_net is not None:
    if script_args.use_reward_features:
        for _path in reward_model_paths:
            _m = AutoModelForSequenceClassification.from_pretrained(
                _path, device_map=f'cuda:{gpu_id}')
            _m.eval()
            for p in _m.parameters():
                p.requires_grad = False
            _tok = AutoTokenizer.from_pretrained(_path)
            if _tok.pad_token is None:
                _tok.pad_token = _tok.eos_token
                _m.config.pad_token_id = _tok.eos_token_id
            feature_models.append(_m)
            feature_tokenizers.append(_tok)
    else:
        # use_lora=True: expert paths are LoRA adapters → load base + merge.
        # use_lora=False: expert paths are full pre-merged models on disk.
        for path in script_args.expert_model_paths:
            if script_args.use_lora:
                m = load_base_model(script_args.sft_model_name, target_device=f'cuda:{gpu_id}')
                m = PeftModel.from_pretrained(m, path)
                m = m.merge_and_unload()
            else:
                m = load_base_model(path, target_device=f'cuda:{gpu_id}')
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            expert_models.append(m)
        feature_models = expert_models

# ── opt_w lookup from gating_dataset_test ─────────────────────────────────────
# key: (prompt_idx, pref_tuple) -> flat np.array of weights
# prompt_idx from collect_rewards.py equals the position in valid_dataset, so
# dataset index i maps directly to prompt_idx i — no fragile text matching needed.
opt_lookup = {}
if script_args.dataset_csv_test and os.path.exists(script_args.dataset_csv_test):
    _gdf       = pd.read_csv(script_args.dataset_csv_test)
    _pref_cols = [c for c in _gdf.columns if c.startswith('pref_')]
    for _, _row in _gdf.iterrows():
        _key = (int(_row['prompt_idx']),
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
    print(f'Loaded {len(opt_lookup)} opt_w entries from {script_args.dataset_csv_test}')


# ── Dataset ────────────────────────────────────────────────────────────────────
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_ppo(
        'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers, split='test')
    # valid_dataset = build_dataset_ppo(
    #     'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers[0], split='train')  # TODO
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_ppo(
        'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers, split='test')
    # valid_dataset = build_dataset_summary_ppo(
    #     'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers, split='train') # TODO
    instructions = Instructions_summary()
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)
print(f'Eval dataset size: {len(valid_dataset)}')
N_DATASET = len(valid_dataset)
sampled_preferences = sample_preferences_uniform(num_experts, script_args.num_pref_samples)


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
opt_groups_by_k      = {}   # k -> {rounded_wt: [dataset_indices]}
idx_to_opt_wt        = {}   # k -> {dataset_idx: rounded_wt}

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

    # Collect opt weights from opt_lookup for actual inference
    if opt_lookup:
        pref_key   = tuple(round(float(p), 4) for p in preference)
        opt_groups = defaultdict(list)
        for i in range(N_DATASET):
            opt_w_flat = opt_lookup.get((i, pref_key))
            if opt_w_flat is None:
                continue
            n = num_experts
            if script_args.block_mode == 'uniform':
                wt = tuple(_round_and_renorm(opt_w_flat))
            else:
                wt = (tuple(_round_and_renorm(opt_w_flat[:n])),
                      tuple(_round_and_renorm(opt_w_flat[n:2*n])),
                      tuple(_round_and_renorm(opt_w_flat[2*n:])))
            opt_groups[wt].append(i)
        opt_groups_by_k[k] = dict(opt_groups)
        idx_to_opt_wt[k]   = {i: wt
                               for wt, idxs in opt_groups_by_k[k].items()
                               for i in idxs}

    print(f'    naive={[round(v,2) for v in naive_w]} | '
          f'pred: {len(groups)} unique group(s) | '
          f'opt: {len(opt_groups_by_k.get(k, {}))} unique group(s)')

# Phase 1 complete — feature_models are no longer needed (all predictions done).
# Delete them now to free GPU memory before Phase 4 loads base_model per iteration.
if gating_net is not None and not script_args.use_reward_features:
    for _m in feature_models:
        del _m
    feature_models.clear()
    expert_models.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print('Freed feature_models from GPU.')

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

# Also add opt weights so they get merged and evaluated (not just estimated via RBF)
for k, groups in opt_groups_by_k.items():
    for wt, indices in groups.items():
        if not wt_is_full.get(wt, False):
            wt_pred_indices[wt].update(indices)

all_unique_wts = sorted(
    set(wt_is_full.keys()) | set(wt_pred_indices.keys()), key=_wt_to_str)

n_full = sum(wt_is_full.values())
n_pred = len(wt_pred_indices)
print(f'  {len(all_unique_wts)} unique weight combos '
      f'({n_full} full-dataset + {n_pred} pred/opt-subset)')

# =============================================================================
# Phase 3 - model setup for merging and evaluation
# =============================================================================
if script_args.use_lora:
    print(f'[Rank {process_id}] Loading base model ...')
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name, torch_dtype=torch.bfloat16,
        device_map=f'cuda:{gpu_id}')
    base_model.resize_token_embeddings(len(tokenizer))

    # Wrap once with the first expert's LoRA config so the model is PEFT-aware
    base_model = PeftModel.from_pretrained(
        base_model, script_args.expert_model_paths[0], is_trainable=False)

    # Cache all adapter state dicts on CPU — only delta weights, not full models
    adapter_state_dicts = load_lora_adapters(base_model, script_args.expert_model_paths)
    n_layers = base_model.config.num_hidden_layers

    base_model = accelerator.prepare(base_model)
else:
    if process_id == 0:
        for wt in all_unique_wts:
            temp_path = os.path.join(output_dir, f'temp_model_w{_wt_to_str(wt)}')
            if os.path.exists(temp_path):
                print(f'  Skipping (already exists): {temp_path}')
            elif script_args.block_mode == 'uniform':
                merge_and_save_weights(
                    expert_model_paths=script_args.expert_model_paths,
                    weights=list(wt), save_path=temp_path)
            else:
                early_w, mid_w, late_w = wt
                merge_and_save_weights_blockwise(
                    expert_model_paths=script_args.expert_model_paths,
                    early_weights=list(early_w), mid_weights=list(mid_w),
                    late_weights=list(late_w), save_path=temp_path)
        print(f'\n[Rank 0] All {len(all_unique_wts)} models merged and saved to disk.')
    accelerator.wait_for_everyone()     # only barrier in the whole script; disk path only

# =============================================================================
# Phase 4 — Merge + evaluate ONCE per unique weight combination
# =============================================================================
print('\n' + '='*60)
print('Phase 4: Merging and evaluating each unique weight combination')
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

    if script_args.use_lora:
        # ── Hot-swap: interpolate adapters in-place, no disk I/O ─────────
        unwrapped = accelerator.unwrap_model(base_model)
        if script_args.block_mode == 'uniform':
            apply_merged_lora(unwrapped, adapter_state_dicts, list(wt))
        else:
            early_w, mid_w, late_w = wt
            apply_merged_lora_blockwise(
                unwrapped, adapter_state_dicts,
                list(early_w), list(mid_w), list(late_w), n_layers)
        gc.collect(); torch.cuda.empty_cache()
    else:
        # ── Load pre-merged model from disk ───────────────────────────────
        temp_path  = os.path.join(output_dir, f'temp_model_w{_wt_to_str(wt)}')
        base_model = AutoModelForCausalLM.from_pretrained(
            temp_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
        base_model.resize_token_embeddings(len(tokenizer))
        base_model = accelerator.prepare(base_model)

    r_rewards, r_prompts, r_responses, r_orig_indices = evaluate_model(subset)
    gc.collect(); torch.cuda.empty_cache()

    if not script_args.use_lora:
        del base_model
        base_model = None
        gc.collect()
        torch.cuda.empty_cache()

    # ...(lora/disk cleanup unchanged)...

    if process_id == 0:
        mapping = {}
        for local_i, local_orig_i in enumerate(r_orig_indices):
            # local_orig_i is the position within dataset/subset, not the global index
            orig_i = local_orig_i if is_full else subset[local_orig_i]
            if local_i >= len(r_prompts):
                break
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

            # opt: actual merged-model inference at utility-optimal grid weights
            opt_wt  = idx_to_opt_wt.get(k, {}).get(i)
            opt_res = wt_result_by_idx.get(opt_wt, {}).get(i, {}) if opt_wt else {}

            row = {
                'prompt_idx':     i,
                'prompt':         naive_res.get('prompt', ''),
                'naive_w':        list(naive_display_w),
                'naive_response': naive_res.get('response', ''),
                'pred_w':         list(pred_wt) if pred_wt else float('nan'),
                'pred_response':  pred_res.get('response', ''),
                'opt_w':          list(opt_wt) if opt_wt else float('nan'),
                'opt_response':   opt_res.get('response', ''),
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

# # ── Cleanup temp models — disk path only ──────────────────────────────────────
# if not script_args.use_lora and process_id == 0:
#     for wt in all_unique_wts:
#         shutil.rmtree(os.path.join(output_dir, f'temp_model_w{_wt_to_str(wt)}'),
#                       ignore_errors=True)
