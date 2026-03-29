import gc
import json
import os
import re
import shutil
from itertools import product
from typing import List, Tuple, Union

import numpy as np
import torch
from pymoo.indicators.hv import HV
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from new_architecture import GatingNetwork

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARLY_FRAC = 1 / 3
LATE_FRAC  = 1 / 3

# ---------------------------------------------------------------------------
# Simplex sampling
# ---------------------------------------------------------------------------

def get_simplex_samples(
    n_objectives: int,
    step: float = 0.2,
    block_mode: str = 'uniform',
) -> List[Union[List[float], Tuple[List[float], List[float], List[float]]]]:
    """Generate weight samples on the simplex.

    uniform : returns List[List[float]]
    custom  : returns List[Tuple[List[float], List[float], List[float]]]
    """
    steps = round(1.0 / step)
    vals  = [round(i * step, 8) for i in range(steps + 1)]
    base  = [
        list(combo)
        for combo in product(vals, repeat=n_objectives)
        if abs(sum(combo) - 1.0) < 1e-6
    ]
    if block_mode == 'uniform':
        return base
    if block_mode == 'custom':
        return list(product(base, base, base))
    raise ValueError(f"Unknown block_mode '{block_mode}'. Choose: uniform | custom")

# ---------------------------------------------------------------------------
# Shared layer-classification helpers
# ---------------------------------------------------------------------------

def _get_layer_idx(key: str):
    """Extract layer index from keys like 'model.layers.7.self_attn.q_proj.weight'."""
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


def _block_weights_fn(key, early_weights, mid_weights, late_weights,
                      early_end, late_start):
    """Return the weight vector that should govern tensor `key`."""
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

# ---------------------------------------------------------------------------
# Disk-based merge helpers  (use_lora=False path)
# ---------------------------------------------------------------------------

def merge_and_save_weights(expert_model_paths, weights, save_path):
    """Flat merge — same weight vector applied to every tensor.

    Memory-efficient: streams one expert at a time.  Peak RAM ≈ 2 × model_float32_size
    instead of the naive (2n+1) × model_float32_size.
    """
    n_experts = len(expert_model_paths)
    assert abs(sum(weights) - 1.0) < 1e-6

    merged = None
    model  = None
    for i in range(n_experts):
        print(f'  Expert {i+1}/{n_experts}: {expert_model_paths[i]}')
        model = AutoModelForCausalLM.from_pretrained(
            expert_model_paths[i], torch_dtype=torch.float32, device_map='cpu')
        sd = model.state_dict()
        if merged is None:
            merged = {k: weights[i] * v.clone() for k, v in sd.items()}
        else:
            for k in merged:
                merged[k].add_(weights[i] * sd[k])
        del sd
        if i < n_experts - 1:
            del model
            gc.collect()
        # keep `model` on the last iteration — reused as the save vessel

    model.load_state_dict(merged)
    del merged; gc.collect()
    model.half()
    model.save_pretrained(save_path)
    AutoTokenizer.from_pretrained(expert_model_paths[0]).save_pretrained(save_path)
    del model; gc.collect()
    print(f'  Saved flat-merged model → {save_path}')


def merge_and_save_weights_blockwise(expert_model_paths, early_weights, mid_weights,
                                     late_weights, save_path,
                                     early_frac=EARLY_FRAC, late_frac=LATE_FRAC):
    """Blockwise disk merge — different weight vectors per layer block.

    Memory-efficient: streams one expert at a time.  Peak RAM ≈ 2 × model_float32_size.
    """
    n_experts = len(expert_model_paths)
    for name, w in [('early', early_weights), ('mid', mid_weights), ('late', late_weights)]:
        assert len(w) == n_experts, \
            f'{name}_weights has {len(w)} entries but there are {n_experts} experts'
        assert abs(sum(w) - 1.0) < 1e-6, \
            f'{name}_weights must sum to 1.0, got {sum(w):.6f}'

    # Read layer count from config without loading weights
    cfg      = AutoConfig.from_pretrained(expert_model_paths[0])
    n_layers   = cfg.num_hidden_layers
    early_end  = int(n_layers * early_frac)
    late_start = n_layers - int(n_layers * late_frac)

    print(f'  Layers: {n_layers} total | '
          f'early 0–{early_end-1} {early_weights} | '
          f'mid {early_end}–{late_start-1} {mid_weights} | '
          f'late {late_start}–{n_layers-1} {late_weights}')

    merged = None
    model  = None
    for i in range(n_experts):
        print(f'  Expert {i+1}/{n_experts}: {expert_model_paths[i]}')
        model = AutoModelForCausalLM.from_pretrained(
            expert_model_paths[i], torch_dtype=torch.float32, device_map='cpu')
        sd = model.state_dict()
        if merged is None:
            merged = {
                k: _block_weights_fn(k, early_weights, mid_weights, late_weights,
                                     early_end, late_start)[i] * v.clone()
                for k, v in sd.items()
            }
        else:
            for k in merged:
                w = _block_weights_fn(k, early_weights, mid_weights, late_weights,
                                      early_end, late_start)
                merged[k].add_(w[i] * sd[k])
        del sd
        if i < n_experts - 1:
            del model
            gc.collect()

    model.load_state_dict(merged)
    del merged; gc.collect()
    model.half()
    model.save_pretrained(save_path)
    AutoTokenizer.from_pretrained(expert_model_paths[0]).save_pretrained(save_path)
    del model; gc.collect()
    print(f'  Saved blockwise-merged model → {save_path}')

# ---------------------------------------------------------------------------
# LoRA-based merge helpers  (use_lora=True path)
# ---------------------------------------------------------------------------

def load_lora_adapters(base_model, expert_model_paths):
    """Load each expert's LoRA adapter into CPU memory as a plain state dict.
    Called ONCE before the inference sweep; base_model is never modified."""
    from peft import set_peft_model_state_dict  # noqa: F401 — ensure peft available
    adapter_state_dicts = []
    for path in expert_model_paths:
        sf_path  = os.path.join(path, 'adapter_model.safetensors')
        bin_path = os.path.join(path, 'adapter_model.bin')
        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            sd = {k: v.clone().cpu() for k, v in load_file(sf_path).items()}
        elif os.path.exists(bin_path):
            sd = {k: v.clone().cpu()
                  for k, v in torch.load(bin_path, map_location='cpu').items()}
        else:
            raise FileNotFoundError(
                f'No adapter_model.safetensors or adapter_model.bin found in {path}')
        adapter_state_dicts.append(sd)
        print(f'  Cached LoRA adapter: {path}')
    return adapter_state_dicts


def apply_merged_lora(peft_model, adapter_state_dicts, weights):
    """Flat interpolation: merge adapter dicts with `weights` and hot-swap in-place."""
    from peft import set_peft_model_state_dict
    assert abs(sum(weights) - 1.0) < 1e-6, f'weights must sum to 1, got {sum(weights)}'
    merged = {
        key: sum(weights[k] * adapter_state_dicts[k][key].float()
                 for k in range(len(weights)))
        for key in adapter_state_dicts[0]
    }
    set_peft_model_state_dict(peft_model,
                              {k: v.to(peft_model.device) for k, v in merged.items()})
    del merged
    gc.collect()


def apply_merged_lora_blockwise(peft_model, adapter_state_dicts,
                                early_weights, mid_weights, late_weights, n_layers,
                                early_frac=EARLY_FRAC, late_frac=LATE_FRAC):
    """Blockwise interpolation: different weight vectors for early/mid/late LoRA keys."""
    from peft import set_peft_model_state_dict
    early_end  = int(n_layers * early_frac)
    late_start = n_layers - int(n_layers * late_frac)
    merged = {
        key: sum(
            _block_weights_fn(key, early_weights, mid_weights, late_weights,
                              early_end, late_start)[k]
            * adapter_state_dicts[k][key].float()
            for k in range(len(adapter_state_dicts))
        )
        for key in adapter_state_dicts[0]
    }
    set_peft_model_state_dict(peft_model,
                              {k: v.to(peft_model.device) for k, v in merged.items()})
    del merged
    gc.collect()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(model_name, target_device=None):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=target_device or 'auto',
    )


def load_expert_state_dicts(expert_paths):
    """Load full model state dicts on CPU in float32."""
    state_dicts = []
    for path in expert_paths:
        m = AutoModelForCausalLM.from_pretrained(path, device_map='cpu')
        state_dicts.append({k: v.clone() for k, v in m.state_dict().items()})
        del m
    return state_dicts

# ---------------------------------------------------------------------------
# Gating network save / load
# ---------------------------------------------------------------------------

def save_gating_network(gating_net, save_path):
    """Save GatingNetwork weights and architecture config together."""
    os.makedirs(save_path, exist_ok=True)
    torch.save(gating_net.state_dict(), os.path.join(save_path, 'gating_network.pt'))
    config = {
        'lm_hidden_size': gating_net.lm_hidden_size,
        'hidden_dim':     gating_net.hidden_dim,
        'num_experts':    gating_net.num_experts,
        'block_mode':     gating_net.block_mode,
    }
    with open(os.path.join(save_path, 'gating_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def load_gating_network(save_path, lm_hidden_size=4096, num_experts=2,
                        block_mode='uniform', hidden_dim=256, device='cuda'):
    """Resolve checkpoint and load GatingNetwork.

    Architecture params are read from gating_config.json when available;
    otherwise caller-supplied defaults are used.
    """
    resolved  = _resolve_checkpoint(save_path, 'gating_network.pt')
    ckpt_file = os.path.join(resolved, 'gating_network.pt')
    if not os.path.exists(ckpt_file):
        return None

    cfg_file = os.path.join(resolved, 'gating_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        lm_hidden_size = cfg.get('lm_hidden_size', lm_hidden_size)
        hidden_dim     = cfg.get('hidden_dim',     hidden_dim)
        num_experts    = cfg.get('num_experts',    num_experts)
        block_mode     = cfg.get('block_mode',     block_mode)

    net = GatingNetwork(lm_hidden_size=lm_hidden_size, num_experts=num_experts,
                        hidden_dim=hidden_dim, block_mode=block_mode)
    net.load_state_dict(torch.load(ckpt_file, map_location=device))
    return net.to(device)


def _resolve_checkpoint(checkpoint_path, filename):
    if not checkpoint_path:
        return checkpoint_path
    if os.path.exists(os.path.join(checkpoint_path, filename)):
        return checkpoint_path
    if not os.path.isdir(checkpoint_path):
        return checkpoint_path
    candidates = []
    for entry in os.listdir(checkpoint_path):
        subdir = os.path.join(checkpoint_path, entry)
        if not os.path.exists(os.path.join(subdir, filename)):
            continue
        key = (
            int(re.search(r'step_(\d+)',  entry).group(1)) if re.search(r'step_(\d+)',  entry) else -1,
            int(re.search(r'epoch_(\d+)', entry).group(1)) if re.search(r'epoch_(\d+)', entry) else -1,
            os.path.getmtime(subdir),
        )
        candidates.append((key, subdir))
    return sorted(candidates, reverse=True)[0][1] if candidates else checkpoint_path

# ---------------------------------------------------------------------------
# Reward-based selection helpers
# ---------------------------------------------------------------------------

def detect_weight_columns(df, block_mode: str):
    """Return the ordered list of weight column names from a rewards DataFrame."""
    if block_mode == 'uniform':
        return sorted([c for c in df.columns if c.startswith('w') and c[1:].isdigit()],
                      key=lambda c: int(c[1:]))
    e_cols = sorted([c for c in df.columns if c.endswith('_early')],
                    key=lambda c: int(c[1:c.index('_')]))
    m_cols = sorted([c for c in df.columns if c.endswith('_mid')],
                    key=lambda c: int(c[1:c.index('_')]))
    l_cols = sorted([c for c in df.columns if c.endswith('_late')],
                    key=lambda c: int(c[1:c.index('_')]))
    return e_cols + m_cols + l_cols


def build_reward_maps(rewards_df, dataset_df, reward_names, block_mode, loss_mode):
    """Precompute per-prompt reward maps needed by the gating network training loss.

    Returns:
        reward_basis_map : prompt_idx -> (n_w, n_rewards) float32 array  (lstsq fit W→R)
        opt_r_map        : (prompt_idx, pref_tuple) -> (n_rewards,) float32  [reward mode]
        r_star_map       : prompt_idx -> (n_rewards,) float32 ideal point     [chebyshev mode]
    """
    r_cols = [f'reward_{n}' for n in reward_names]
    w_cols = detect_weight_columns(rewards_df, block_mode)

    reward_basis_map = {}
    r_star_map       = {}
    for pidx in rewards_df['prompt_idx'].unique():
        sub = rewards_df[rewards_df['prompt_idx'] == pidx]
        W   = sub[w_cols].values.astype(np.float64)
        R   = sub[r_cols].values.astype(np.float64)
        B, _, _, _ = np.linalg.lstsq(W, R, rcond=None)
        reward_basis_map[int(pidx)] = B.astype(np.float32)
        r_star_map[int(pidx)]       = R.max(axis=0).astype(np.float32)

    opt_r_map = None
    if loss_mode == 'reward':
        opt_r_map = {}
        for pidx in rewards_df['prompt_idx'].unique():
            sub  = rewards_df[rewards_df['prompt_idx'] == pidx]
            R    = sub[r_cols].values.astype(np.float64)
            rows = dataset_df[dataset_df['prompt_idx'] == pidx]
            for _, drow in rows.iterrows():
                pref     = np.array([float(drow[f'pref_{n}']) for n in reward_names],
                                    dtype=np.float64)
                pref_key = tuple(round(float(p), 6) for p in pref)
                best_idx = utility_optimal_weights(R, pref, list(range(len(R))))
                opt_r_map[(int(pidx), pref_key)] = R[best_idx].astype(np.float32)

    return reward_basis_map, opt_r_map, r_star_map


def utility_optimal_weights(reward_matrix, preference, sample_weights):
    """Return the sample weight with the highest linear utility sum_i(pref_i * reward_i).

    Args:
        reward_matrix:  (n_samples, n_rewards) array of real measured rewards.
        preference:     (n_rewards,) preference vector summing to 1.
        sample_weights: list of weight vectors (uniform) or list of (early, mid, late)
                        tuples (blockwise). Must have n_samples entries.

    Returns:
        The entry from sample_weights with the highest utility (same type as input).
    """
    preference    = np.array(preference,    dtype=np.float64)
    reward_matrix = np.array(reward_matrix, dtype=np.float64)
    utilities     = reward_matrix @ preference   # (n_samples,)
    best_idx      = int(np.argmax(utilities))
    return sample_weights[best_idx]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def compute_hypervolume(reward_vectors):
    if len(reward_vectors) == 0:
        return 0.0
    return HV(ref_point=np.ones(len(reward_vectors[0])))(-np.array(reward_vectors))
