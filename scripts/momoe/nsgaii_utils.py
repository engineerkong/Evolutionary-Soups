"""nsgaii_utils.py — Utilities for standard NSGA-II with plain GatingNetwork."""

import json
import os
import re
from itertools import product
from typing import List

import numpy as np
import torch

from nsgaii_architecture import GatingNetwork

# ---------------------------------------------------------------------------
# Reward model paths
# ---------------------------------------------------------------------------
REWARD_PATHS = {
    'harmless':          'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':           'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':           'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':           'Tristan/gpt2_reward_summarization',
    'faithful':          'CogComp/bart-faithful-summary-detector',
    'humor':             'mohameddhiab/humor-no-humor',
    'beaver_reward':     'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':       'PKU-Alignment/beaver-7b-v1.0-cost',
    'steer_helpfulness':        'urm://LxzGordon/URM-LLaMa-3.1-8B#0',
    'steer_correctness':        'urm://LxzGordon/URM-LLaMa-3.1-8B#1',
    'steer_coherence':          'urm://LxzGordon/URM-LLaMa-3.1-8B#2',
    'steer_complexity':         'urm://LxzGordon/URM-LLaMa-3.1-8B#3',
    'steer_verbosity':          'urm://LxzGordon/URM-LLaMa-3.1-8B#4',
    'uf_instruction_following': 'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1#6',
    'uf_truthfulness':          'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1#7',
    'uf_honesty':               'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1#8',
    'uf_helpfulness':           'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1#9',
}


# ---------------------------------------------------------------------------
# Simplex sampling
# ---------------------------------------------------------------------------

def get_simplex_samples(n_objectives: int, step: float = 0.2) -> List[List[float]]:
    steps = round(1.0 / step)
    vals  = [round(i * step, 8) for i in range(steps + 1)]
    return [list(c) for c in product(vals, repeat=n_objectives)
            if abs(sum(c) - 1.0) < 1e-6]


# ---------------------------------------------------------------------------
# Gating network save / load
# ---------------------------------------------------------------------------

def save_gating_network(gating_net, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    torch.save(gating_net.state_dict(), os.path.join(save_path, 'gating_network.pt'))
    config = {
        'type':           type(gating_net).__name__,
        'lm_hidden_size': gating_net.lm_hidden_size,
        'num_experts':    gating_net.num_experts,
        'hidden_size':    gating_net.hidden_size,
        'num_layers':     gating_net.num_layers,
        'fixed_alpha':    gating_net.fixed_alpha,
    }
    with open(os.path.join(save_path, 'gating_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def load_gating_network(save_path: str, lm_hidden_size: int = 4096,
                        num_experts: int = 2, num_layers: int = 32,
                        device: str = 'cuda'):
    resolved  = _resolve_checkpoint(save_path, 'gating_network.pt')
    ckpt_file = os.path.join(resolved, 'gating_network.pt')
    if not os.path.exists(ckpt_file):
        return None

    net_type    = 'GatingNetwork'
    hidden_size = 256
    fixed_alpha = None
    cfg_file = os.path.join(resolved, 'gating_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        net_type       = cfg.get('type',           net_type)
        lm_hidden_size = cfg.get('lm_hidden_size', lm_hidden_size)
        num_experts    = cfg.get('num_experts',     num_experts)
        hidden_size    = cfg.get('hidden_size',     hidden_size)
        num_layers     = cfg.get('num_layers',      num_layers)
        fixed_alpha    = cfg.get('fixed_alpha',     None)

    net = GatingNetwork(lm_hidden_size=lm_hidden_size, num_experts=num_experts,
                        hidden_size=hidden_size, num_layers=num_layers,
                        fixed_alpha=fixed_alpha)
    # strict=False allows loading old checkpoints that lack the alpha parameter
    net.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
    return net.to(device).bfloat16()


def _resolve_checkpoint(checkpoint_path: str, filename: str) -> str:
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
