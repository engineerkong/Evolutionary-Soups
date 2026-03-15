import gc
import os
import re
import shutil

import numpy as np
import torch
from pymoo.indicators.hv import HV
from transformers import AutoModelForCausalLM

from qmo_architecture import QTableGating


def compute_hypervolume(reward_vectors):
    if len(reward_vectors) == 0:
        return 0.0
    return HV(ref_point=np.ones(len(reward_vectors[0])))(-np.array(reward_vectors))


def resolve_gating_checkpoint_path(checkpoint_path):
    if not checkpoint_path or os.path.exists(os.path.join(checkpoint_path, "qtable.npy")):
        return checkpoint_path
    if not os.path.isdir(checkpoint_path):
        return checkpoint_path

    candidates = []
    for entry in os.listdir(checkpoint_path):
        subdir = os.path.join(checkpoint_path, entry)
        if not os.path.exists(os.path.join(subdir, "qtable.npy")):
            continue
        key = (
            int(re.search(r"step_(\d+)", entry).group(1)) if re.search(r"step_(\d+)", entry) else -1,
            int(re.search(r"epoch_(\d+)", entry).group(1)) if re.search(r"epoch_(\d+)", entry) else -1,
            os.path.getmtime(subdir),
        )
        candidates.append((key, subdir))
    return sorted(candidates, reverse=True)[0][1] if candidates else checkpoint_path


def merge_and_save_weights(expert_model_paths, expert_weights, temp_save_path):
    """Merge full PPO model state dicts and save to disk.
    Exactly replicates merge_weights_with_preference from utils.py:
      - Loads each model without torch_dtype (float32)
      - get_average_state_dict: i=0 sets, i>0 accumulates in-place on state_dicts[0]
      - Saves merged model with save_pretrained
    """
    models = []
    for path in expert_model_paths:
        model_tmp = AutoModelForCausalLM.from_pretrained(path, device_map='cpu')
        models.append(model_tmp)

    state_dicts = [m.state_dict() for m in models]

    # Replicate get_average_state_dict exactly
    for i, (state_dict, coefficient) in enumerate(zip(state_dicts, expert_weights)):
        current_weights = state_dict
        for key in list(current_weights.keys()):
            if i == 0:
                state_dicts[0][key] = coefficient * current_weights[key]
            else:
                state_dicts[0][key] += coefficient * current_weights[key]

    model_1 = models[0]
    model_1.load_state_dict(state_dicts[0], strict=False)

    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path, ignore_errors=True)
    model_1.save_pretrained(temp_save_path)

    while models:
        del models[0]
    while state_dicts:
        del state_dicts[0]
    gc.collect()
    torch.cuda.empty_cache()


def load_expert_state_dicts(expert_paths):
    """Load full model state dicts on CPU (float32), for in-memory merging in train."""
    state_dicts = []
    for expert_path in expert_paths:
        model_tmp = AutoModelForCausalLM.from_pretrained(expert_path, device_map="cpu")
        state_dicts.append({k: v.clone() for k, v in model_tmp.state_dict().items()})
        del model_tmp
    return state_dicts


def load_base_model(base_model_name, target_device=None):
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": target_device or "auto"}
    return AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)


def save_moe_qtable(q_table, save_path):
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "qtable.npy"), q_table.q_table)


def load_moe_qtable(q_table, save_path):
    resolved = resolve_gating_checkpoint_path(save_path)
    qtable_path = os.path.join(resolved, "qtable.npy")
    if not os.path.exists(qtable_path):
        return False
    q_table.q_table = np.load(qtable_path)
    return True