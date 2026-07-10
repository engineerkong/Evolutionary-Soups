"""_dummy.py — Initialise a random ES population with a fixed seed and save it.

Imitates es_train.py up to and including ES.__init__, then saves the
un-evolved initial population directly as 'final/ind_XXX/'. No evaluation,
no evolution. Useful as a reproducible random-init baseline against which
evolved populations can be compared.
"""

import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from transformers import HfArgumentParser
from trl import set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from es_architecture import GatingNetwork, SimpleGatingNetwork, load_shared_experts
from es_utils import (
    REWARD_PATHS, load_main_tokenizer, net_to_params, params_to_net,
    save_gating_network,
)


# ---------------------------------------------------------------------------
# Script arguments  (mirrors es_train.py)
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:      str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths:   List[str] = field(default_factory=list)
    reward_names:         str       = 'harmless,helpful'
    population_size:      int       = 20
    fixed_alpha:          float     = 1.2
    gating_type:          str       = 'per_layer'  # 'per_layer' = GatingNetwork, 'simple' = SimpleGatingNetwork
    warm_start_path:      str       = ''
    gpu_id:               int       = -1
    save_directory:       str       = './models/ES/'
    run_name:             str       = 'dummy_es'
    seed:                 int       = 8888


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser      = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

output_dir = os.path.join(script_args.save_directory, script_args.run_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(script_args.seed)
np.random.seed(script_args.seed)

if 'RANK' in os.environ:
    torch.distributed.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=600))
accelerator = Accelerator()
gpu_id      = (script_args.gpu_id if script_args.gpu_id >= 0
               else accelerator.local_process_index)
device      = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_objectives = len(reward_names)

# Load tokenizer and expert models (needed to determine lm_hidden_size / num_layers)
sft_tokenizer = load_main_tokenizer(script_args.expert_model_paths[0])

expert_models = load_shared_experts(
    script_args.base_model_name, script_args.expert_model_paths, device,
    tokenizer=sft_tokenizer)

lm_hidden_size = expert_models[0].config.hidden_size
_num_layers    = len(expert_models[0].model.layers)
print(f'lm_hidden_size = {lm_hidden_size}, num_layers = {_num_layers}')

if script_args.gating_type == 'simple':
    template_net = SimpleGatingNetwork(
        lm_hidden_size=lm_hidden_size,
        num_experts=len(expert_models),
        fixed_alpha=script_args.fixed_alpha,
    )
    print(f'Gating type: SimpleGatingNetwork')
else:
    template_net = GatingNetwork(
        lm_hidden_size=lm_hidden_size,
        num_experts=len(expert_models),
        num_layers=_num_layers,
        alpha_init=script_args.fixed_alpha,
        fixed_alpha=script_args.fixed_alpha,
    )
    print(f'Gating type: GatingNetwork (per_layer)')

if script_args.warm_start_path:
    if script_args.gating_type == 'simple':
        from es_utils import load_simple_gating_network as _load
        loaded = _load(script_args.warm_start_path, lm_hidden_size=lm_hidden_size,
                       num_experts=len(expert_models), device='cpu')
    else:
        from es_utils import load_gating_network as _load
        loaded = _load(script_args.warm_start_path, lm_hidden_size=lm_hidden_size,
                       num_experts=len(expert_models), num_layers=_num_layers,
                       device='cpu')
    if loaded is not None:
        template_net = loaded.cpu().bfloat16()
        print(f'Warm-start from {script_args.warm_start_path}')

# Initialise population with the fixed seed — identical to ES.__init__
base_params    = net_to_params(template_net)
param_dim      = len(base_params)
population     = [
    base_params + np.random.randn(param_dim) * 0.05
    for _ in range(script_args.population_size)
]
dummy_fitness  = [np.zeros(n_objectives) for _ in range(script_args.population_size)]

print(f'\nPopulation of {script_args.population_size} individuals initialised '
      f'(seed={script_args.seed}, param_dim={param_dim}).  Skipping evolution.')

# Save directly as 'final'  (same layout as es_train.py)
is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
if is_main:
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    for i, params in enumerate(population):
        subdir = os.path.join(final_dir, f'ind_{i:03d}')
        net    = params_to_net(params, template_net, 'cpu')
        save_gating_network(net, subdir)
        with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
            json.dump({'fitness': dummy_fitness[i].tolist()}, f, indent=2)

    with open(os.path.join(output_dir, 'es_meta.json'), 'w') as f:
        json.dump({
            'reward_names':    reward_names,
            'z_star':          [0.0] * n_objectives,
            'final_fitness':   [f.tolist() for f in dummy_fitness],
            'num_generations': 0,
            'population_size': script_args.population_size,
            'lm_hidden_size':  lm_hidden_size,
        }, f, indent=2)

    print(f'Saved initial population → {final_dir}')
