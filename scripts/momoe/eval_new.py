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
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, HfArgumentParser)
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
from new_architecture import GatingNetwork, get_prompt_hidden, get_prompt_hidden_from_reward_models
from new_utils import load_base_model, load_gating_network

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}

EARLY_FRAC = 1 / 3
LATE_FRAC  = 1 / 3


# ── In-memory LoRA delta helpers ──────────────────────────────────────────────

def _get_layer_idx(key: str):
    parts = key.split('.')
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _is_head_tensor(key: str) -> bool:
    return any(k in key for k in ('lm_head', 'model.norm'))


def _is_embed_tensor(key: str) -> bool:
    return 'embed_tokens' in key


def _block_weights_fn(key, early_weights, mid_weights, late_weights, early_end, late_start):
    if _is_embed_tensor(key):
        return early_weights
    if _is_head_tensor(key):
        return late_weights
    idx = _get_layer_idx(key)
    if idx is None:
        return mid_weights
    if idx < early_end:
        return early_weights
    if idx >= late_start:
        return late_weights
    return mid_weights


def cache_expert_adapters(expert_model_paths):
    """Load each expert's LoRA A/B matrices and scaling on CPU."""
    import json
    all_adapters = []
    for path in expert_model_paths:
        with open(os.path.join(path, 'adapter_config.json')) as f:
            cfg = json.load(f)
        r       = cfg['r']
        scaling = cfg.get('lora_alpha', r) / r

        sf_path  = os.path.join(path, 'adapter_model.safetensors')
        bin_path = os.path.join(path, 'adapter_model.bin')
        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            raw = {k: v.float().cpu() for k, v in load_file(sf_path).items()}
        elif os.path.exists(bin_path):
            raw = {k: v.float().cpu()
                   for k, v in torch.load(bin_path, map_location='cpu').items()}
        else:
            raise FileNotFoundError(f'No adapter weights found in {path}')

        adapters = {}
        for key, val in raw.items():
            if '.lora_A.weight' not in key:
                continue
            param = (key.replace('base_model.model.', '')
                        .replace('.lora_A.weight', '.weight'))
            b_key = key.replace('lora_A', 'lora_B')
            if b_key in raw:
                adapters[param] = {'A': val, 'B': raw[b_key], 'scaling': scaling}

        all_adapters.append(adapters)
        print(f'  [cache] {len(adapters)} LoRA params <- {path}')
    return all_adapters


def apply_merged_delta(model, expert_adapters, weights, device):
    """delta = Σ w_i * scaling_i * (B_i @ A_i); add in-place, return for restoration.
    Supports both merged and unmerged PEFT base models (.base_layer.weight fallback)."""
    assert abs(sum(weights) - 1.0) < 1e-6, f'weights must sum to 1, got {sum(weights)}'
    named   = dict(model.named_parameters())
    applied = {}
    for param_name in expert_adapters[0]:
        actual = param_name
        if actual not in named:
            fallback = param_name.replace('.weight', '.base_layer.weight')
            if fallback in named:
                actual = fallback
            else:
                continue
        param = named[actual]
        delta = sum(
            weights[i] * expert_adapters[i][param_name]['scaling']
            * (expert_adapters[i][param_name]['B'] @ expert_adapters[i][param_name]['A'])
            for i in range(len(weights))
        ).to(dtype=param.dtype, device=device)
        param.data.add_(delta)
        applied[actual] = delta
    return applied


def apply_merged_delta_blockwise(model, expert_adapters, early_weights, mid_weights,
                                  late_weights, n_layers, device,
                                  early_frac=EARLY_FRAC, late_frac=LATE_FRAC):
    """Blockwise variant: different weight vectors per layer block.
    Supports both merged and unmerged PEFT base models (.base_layer.weight fallback)."""
    early_end  = int(n_layers * early_frac)
    late_start = n_layers - int(n_layers * late_frac)
    named   = dict(model.named_parameters())
    applied = {}
    for param_name in expert_adapters[0]:
        actual = param_name
        if actual not in named:
            fallback = param_name.replace('.weight', '.base_layer.weight')
            if fallback in named:
                actual = fallback
            else:
                continue
        param = named[actual]
        w = _block_weights_fn(param_name, early_weights, mid_weights, late_weights,
                              early_end, late_start)
        delta = sum(
            w[i] * expert_adapters[i][param_name]['scaling']
            * (expert_adapters[i][param_name]['B'] @ expert_adapters[i][param_name]['A'])
            for i in range(len(w))
        ).to(dtype=param.dtype, device=device)
        param.data.add_(delta)
        applied[actual] = delta
    return applied


def restore_model(model, applied):
    """Subtract applied deltas to restore base model weights."""
    named = dict(model.named_parameters())
    for param_name, delta in applied.items():
        if param_name in named:
            named[param_name].data.sub_(delta)
    del applied
    gc.collect()
    torch.cuda.empty_cache()


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
    save_directory:        str           = './results/new/'
    wandb_name:            str           = 'new_assistant_eval'
    hidden_dim:            int           = 256
    use_reward_features:   bool          = True
    use_lora:              bool          = True   # True → in-memory LoRA swap (recommended)


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
            feature_models.append(_m)
            feature_tokenizers.append(_tok)
    else:
        for path in script_args.expert_model_paths:
            m = load_base_model(path, target_device=f'cuda:{gpu_id}')
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            expert_models.append(m)
        feature_models = expert_models

# ── Expert adapters + base model for generation ────────────────────────────────
expert_adapters = None
base_model      = None
n_layers        = None
if script_args.use_lora:
    print(f'[Rank {process_id}] Caching LoRA A/B matrices on CPU ...')
    expert_adapters = cache_expert_adapters(script_args.expert_model_paths)
    print(f'[Rank {process_id}] Loading base model on GPU ...')
    _raw_base = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name, torch_dtype=torch.bfloat16,
        device_map=f'cuda:{gpu_id}')
    _raw_base.resize_token_embeddings(len(tokenizer))

    # Check if sft_model_name is a PeftModel (has adapter_config.json);
    # if so, merge its LoRA into the base weights and unload to get a clean model.
    _adapter_cfg = os.path.join(script_args.sft_model_name, 'adapter_config.json')
    if os.path.exists(_adapter_cfg):
        from peft import PeftModel
        print(f'[Rank {process_id}] SFT model is a PeftModel — merging LoRA into base ...')
        _raw_base = PeftModel.from_pretrained(_raw_base, script_args.sft_model_name)
        _raw_base = _raw_base.merge_and_unload()
        print(f'[Rank {process_id}] merge_and_unload() done — base model is now clean.')

    base_model = _raw_base
    del _raw_base
    n_layers   = base_model.config.num_hidden_layers
    base_model = accelerator.prepare(base_model)

# ── opt_w lookup from gating_dataset_test ─────────────────────────────────────
# key: (prompt_text, pref_tuple) -> flat np.array of weights
opt_lookup = {}
if script_args.dataset_csv_test and os.path.exists(script_args.dataset_csv_test):
    _gdf       = pd.read_csv(script_args.dataset_csv_test)
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
    print(f'Loaded {len(opt_lookup)} opt_w entries from {script_args.dataset_csv_test}')


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


def _apply_weights_for_group(wt):
    """Apply LoRA delta for weight tuple wt to base_model in-place; return applied dict."""
    unwrapped = accelerator.unwrap_model(base_model)
    if script_args.block_mode == 'uniform':
        return apply_merged_delta(
            unwrapped, expert_adapters, list(wt), device=f'cuda:{gpu_id}')
    else:
        early_w, mid_w, late_w = wt
        return apply_merged_delta_blockwise(
            unwrapped, expert_adapters,
            list(early_w), list(mid_w), list(late_w), n_layers,
            device=f'cuda:{gpu_id}')


def _wt_to_str(wt):
    if script_args.block_mode == 'uniform':
        return '_'.join(f'{v:.2f}' for v in wt)
    else:
        e, m, l = wt
        return ('E' + '_'.join(f'{v:.2f}' for v in e) +
                '_M' + '_'.join(f'{v:.2f}' for v in m) +
                '_L' + '_'.join(f'{v:.2f}' for v in l))


def evaluate_model(subset_indices=None):
    """Generate + score on full dataset or a subset. Returns (rewards, prompts, responses).
    Expects base_model weights to already be patched by _apply_weights_for_group."""
    dataset = valid_dataset.select(subset_indices) if subset_indices is not None \
              else valid_dataset
    if len(dataset) == 0:
        return [[] for _ in reward_names], [], []
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(dataset, batch_size=128, drop_last=False, collate_fn=collator)
    loader   = accelerator.prepare(loader)
    gen_kwargs = {
        'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
        'min_length': -1, 'top_k': 0.0, 'top_p': 0.9, 'do_sample': False,
    }
    tokenizer.padding_side = 'left'
    full_responses, full_prompts = [], []
    base_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating', leave=False):
            out = accelerator.unwrap_model(base_model).generate(
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
    all_rewards   = [accelerator.gather_for_metrics(r) for r in rewards_list]
    all_prompts   = accelerator.gather_for_metrics(full_prompts)
    all_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_prompts, all_responses


# =============================================================================
# Pre-decode all prompts (needed to look up opt weights from opt_lookup)
# =============================================================================
all_prompts_decoded = []
if opt_lookup:
    _collator_tmp = DataCollatorWithPadding(tokenizer=tokenizer)
    _loader_tmp   = DataLoader(valid_dataset, batch_size=256, shuffle=False,
                               drop_last=False, collate_fn=_collator_tmp)
    for _b in _loader_tmp:
        all_prompts_decoded.extend(
            tokenizer.batch_decode(_b['input_ids'], skip_special_tokens=True))
    print(f'Pre-decoded {len(all_prompts_decoded)} prompts for opt weight lookup')

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

    # Collect opt weights from opt_lookup for actual inference (not just RBF)
    if all_prompts_decoded and opt_lookup:
        pref_key   = tuple(round(float(p), 4) for p in preference)
        opt_groups = defaultdict(list)
        for i, prompt_text in enumerate(all_prompts_decoded):
            opt_w_flat = opt_lookup.get((prompt_text.strip(), pref_key))
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

    _applied = _apply_weights_for_group(wt)
    gc.collect(); torch.cuda.empty_cache()

    r_rewards, r_prompts, r_responses = evaluate_model(subset)
    gc.collect(); torch.cuda.empty_cache()

    restore_model(accelerator.unwrap_model(base_model), _applied)

    if process_id == 0:
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
# Phase 4 — Assemble per-preference results and save
# =============================================================================
print('\n' + '='*60)
print('Phase 4: Assembling results')
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
