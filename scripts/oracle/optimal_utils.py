"""Utilities shared across the optimal/ pipeline scripts."""

import gc
from itertools import product
from pathlib import Path
from typing import List

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Simplex sampling  (mirrors nsgaii_utils.get_simplex_samples)
# ---------------------------------------------------------------------------

def get_simplex_samples(n_objectives: int, step: float = 0.1) -> List[List[float]]:
    steps = round(1.0 / step)
    vals  = [round(i * step, 8) for i in range(steps + 1)]
    return [list(c) for c in product(vals, repeat=n_objectives)
            if abs(sum(c) - 1.0) < 1e-6]


# ---------------------------------------------------------------------------
# LoRA adapter loading  (mirrors nsgaii_test._load_adapter_sd)
# ---------------------------------------------------------------------------

def load_adapter_sd(adapter_dir: str) -> dict:
    p        = Path(adapter_dir)
    st_path  = p / 'adapter_model.safetensors'
    bin_path = p / 'adapter_model.bin'
    if st_path.exists():
        sd = {}
        with safe_open(str(st_path), framework='pt', device='cpu') as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        return sd
    elif bin_path.exists():
        return torch.load(str(bin_path), map_location='cpu')
    raise FileNotFoundError(f'No adapter weights found in {adapter_dir}')


# ---------------------------------------------------------------------------
# Merge LoRA adapters and save to disk
#
# Mirrors nsgaii_test._build_rewarded_soup but saves the merged model to
# disk so collect_rewards.py can load it per-rank without re-merging.
# ---------------------------------------------------------------------------

def merge_lora_and_save(base_model_name: str,
                        expert_model_paths: List[str],
                        weights: List[float],
                        save_path: str) -> None:
    """Blend LoRA adapter weights and save the fully-merged model to disk."""
    assert abs(sum(weights) - 1.0) < 1e-6, f'weights must sum to 1, got {sum(weights)}'

    expert_sds = [load_adapter_sd(p) for p in expert_model_paths]
    peft_cfg   = LoraConfig.from_pretrained(expert_model_paths[0])

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map='cpu')

    blended_sd = {
        k: sum(weights[i] * expert_sds[i][k].to(torch.bfloat16)
               for i in range(len(weights)))
        for k in expert_sds[0]
    }

    model = get_peft_model(base, peft_cfg)
    set_peft_model_state_dict(model, blended_sd)
    model = model.merge_and_unload()

    model.save_pretrained(save_path)
    AutoTokenizer.from_pretrained(expert_model_paths[0]).save_pretrained(save_path)

    del model, base, blended_sd, expert_sds
    gc.collect()
    print(f'  Saved merged model → {save_path}')


# ---------------------------------------------------------------------------
# Oracle weight selection  (mirrors new_utils.utility_optimal_weights)
# ---------------------------------------------------------------------------

def utility_optimal_weights(reward_matrix, preference, sample_weights):
    """Return the sample_weight entry with the highest linear utility λ·r."""
    preference    = np.array(preference,    dtype=np.float64)
    reward_matrix = np.array(reward_matrix, dtype=np.float64)
    best_idx      = int(np.argmax(reward_matrix @ preference))
    return sample_weights[best_idx]
