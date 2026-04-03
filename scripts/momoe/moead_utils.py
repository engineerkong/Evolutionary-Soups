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

from moead_architecture import GatingNetwork

# ---------------------------------------------------------------------------
# Reward model paths
# ---------------------------------------------------------------------------
REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
}


# ---------------------------------------------------------------------------
# Simplex sampling
# ---------------------------------------------------------------------------

def get_simplex_samples(
    n_objectives: int,
    step: float = 0.2,
) -> List[List[float]]:
    """Generate weight samples on the simplex."""
    steps = round(1.0 / step)
    vals  = [round(i * step, 8) for i in range(steps + 1)]
    return [
        list(combo)
        for combo in product(vals, repeat=n_objectives)
        if abs(sum(combo) - 1.0) < 1e-6
    ]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(model_name, target_device=None):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=target_device or 'auto',
    )

# ---------------------------------------------------------------------------
# Gating network save / load
# ---------------------------------------------------------------------------

def save_gating_network(gating_net, save_path):
    """Save GatingNetwork weights and architecture config together."""
    os.makedirs(save_path, exist_ok=True)
    torch.save(gating_net.state_dict(), os.path.join(save_path, 'gating_network.pt'))
    config = {
        'lm_hidden_size': gating_net.lm_hidden_size,
        'num_experts':    gating_net.num_experts,
        'hidden_size':    gating_net.hidden_size,
    }
    with open(os.path.join(save_path, 'gating_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def load_gating_network(save_path, lm_hidden_size=4096, num_experts=2, device='cuda'):
    """Resolve checkpoint and load GatingNetwork.

    Architecture params are read from gating_config.json when available;
    otherwise caller-supplied defaults are used.
    """
    resolved  = _resolve_checkpoint(save_path, 'gating_network.pt')
    ckpt_file = os.path.join(resolved, 'gating_network.pt')
    if not os.path.exists(ckpt_file):
        return None

    hidden_size = 64
    cfg_file = os.path.join(resolved, 'gating_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        lm_hidden_size = cfg.get('lm_hidden_size', lm_hidden_size)
        num_experts    = cfg.get('num_experts',    num_experts)
        hidden_size    = cfg.get('hidden_size',    hidden_size)

    net = GatingNetwork(lm_hidden_size=lm_hidden_size, num_experts=num_experts, hidden_size=hidden_size)
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

