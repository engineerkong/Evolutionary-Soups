import gc
import os
import re
import shutil

import numpy as np
import torch
from pymoo.indicators.hv import HV
from transformers import AutoModelForCausalLM, AutoTokenizer

from new_architecture import GatingNetwork


def compute_hypervolume(reward_vectors):
    if len(reward_vectors) == 0:
        return 0.0
    return HV(ref_point=np.ones(len(reward_vectors[0])))(-np.array(reward_vectors))


def load_expert_state_dicts(expert_paths):
    """Load full model state dicts on CPU in float32."""
    state_dicts = []
    for path in expert_paths:
        m = AutoModelForCausalLM.from_pretrained(path, device_map='cpu')
        state_dicts.append({k: v.clone() for k, v in m.state_dict().items()})
        del m
    return state_dicts


def load_base_model(model_name, target_device=None):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=target_device or 'auto',
    )


def merge_and_save_weights(expert_model_paths, expert_weights, temp_save_path):
    """Merge full PPO model state dicts and save to disk (identical to eval_ppo_rs)."""
    models = [AutoModelForCausalLM.from_pretrained(p, device_map='cpu')
              for p in expert_model_paths]
    state_dicts = [m.state_dict() for m in models]

    for i, (sd, w) in enumerate(zip(state_dicts, expert_weights)):
        for key in list(sd.keys()):
            if i == 0:
                state_dicts[0][key] = w * sd[key]
            else:
                state_dicts[0][key] += w * sd[key]

    models[0].load_state_dict(state_dicts[0], strict=False)
    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path, ignore_errors=True)
    models[0].save_pretrained(temp_save_path)

    while models:
        del models[0]
    while state_dicts:
        del state_dicts[0]
    gc.collect()
    torch.cuda.empty_cache()


def _get_layer_idx(key: str) -> int | None:
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
    """Identify output-side tensors: lm_head and final norm."""
    return any(k in key for k in ('lm_head', 'model.norm'))

def _is_embed_tensor(key: str) -> bool:
    """Identify input-side tensors: token embeddings."""
    return 'embed_tokens' in key

def merge_and_save_weights_blockwise(
    expert_model_paths: list[str],
    early_weights: list[float],
    mid_weights:   list[float],
    late_weights:  list[float],
    save_path: str,
    early_frac: float = 1/3,
    late_frac:  float = 1/3,
):
    """
    Merge expert models with different weights per layer block.

    Tensor assignment:
      embed_tokens          → early_weights  (input side)
      layers 0..early_end-1 → early_weights
      layers early_end..late_start-1 → mid_weights
      layers late_start..n_layers-1  → late_weights
      model.norm + lm_head  → late_weights   (output side)

    Args:
        expert_model_paths : paths or HF ids for each expert model
        early_weights      : merge coefficients for early layers + embeddings
        mid_weights        : merge coefficients for middle layers
        late_weights       : merge coefficients for late layers + lm_head/norm
        save_path          : where to save the merged model
        early_frac         : fraction of layers treated as early (default 1/3)
        late_frac          : fraction of layers treated as late  (default 1/3)
    """
    n_experts = len(expert_model_paths)

    for name, w in [('early', early_weights), ('mid', mid_weights), ('late', late_weights)]:
        assert len(w) == n_experts, \
            f"{name}_weights has {len(w)} entries but there are {n_experts} experts"
        assert abs(sum(w) - 1.0) < 1e-6, \
            f"{name}_weights must sum to 1.0, got {sum(w):.6f}"

    print(f"Loading {n_experts} expert models...")
    models = [
        AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
        for p in expert_model_paths
    ]
    state_dicts = [m.state_dict() for m in models]

    n_layers  = models[0].config.num_hidden_layers
    early_end  = int(n_layers * early_frac)
    late_start = n_layers - int(n_layers * late_frac)

    print(f"  Total layers : {n_layers}")
    print(f"  Embeddings   : early_weights={early_weights}")
    print(f"  Early block  : layers 0–{early_end - 1}  weights={early_weights}")
    print(f"  Mid block    : layers {early_end}–{late_start - 1}  weights={mid_weights}")
    print(f"  Late block   : layers {late_start}–{n_layers - 1}  weights={late_weights}")
    print(f"  Norm + Head  : late_weights={late_weights}")

    def _get_block_weights(key: str) -> list[float]:
        if _is_embed_tensor(key):
            return early_weights
        if _is_head_tensor(key):
            return late_weights
        layer_idx = _get_layer_idx(key)
        if layer_idx is None:
            # Fallback for any unrecognised non-layer tensor
            return mid_weights
        if layer_idx < early_end:
            return early_weights
        if layer_idx >= late_start:
            return late_weights
        return mid_weights

    print("Merging tensors...")
    merged_state_dict = {}
    for key in state_dicts[0]:
        w = _get_block_weights(key)
        merged_state_dict[key] = sum(
            w[k] * state_dicts[k][key].float()
            for k in range(n_experts)
        )

    print(f"Saving merged model to {save_path}...")
    models[0].load_state_dict(merged_state_dict)
    models[0].half()
    models[0].save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(expert_model_paths[0])
    tokenizer.save_pretrained(save_path)
    print("Done.")

def save_gating_network(gating_net, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save(gating_net.state_dict(), os.path.join(save_path, 'gating_network.pt'))


def load_gating_network(save_path, lm_hidden_size=4096, num_experts=2, device='cuda'):
    """Resolve checkpoint and load GatingNetwork."""
    resolved = _resolve_checkpoint(save_path, 'gating_network.pt')
    ckpt_file = os.path.join(resolved, 'gating_network.pt')
    if not os.path.exists(ckpt_file):
        return None
    net = GatingNetwork(lm_hidden_size=lm_hidden_size, num_experts=num_experts)
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
