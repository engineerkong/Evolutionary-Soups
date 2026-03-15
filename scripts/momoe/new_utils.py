import gc
import os
import re
import shutil

import numpy as np
import torch
from pymoo.indicators.hv import HV
from transformers import AutoModelForCausalLM

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
