"""HoE utility functions: model construction, checkpoint I/O, text cleaning."""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Locate the HoE source tree relative to this file:
#   scripts/hoe/  →  MOMoE/  →  workspace root  →  anonymous-repo-for-HoE/
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../MOMoE/scripts/hoe
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # .../MOMoE/scripts → .../MOMoE
_WORKSPACE_ROOT = _PROJECT_ROOT.parent                 # .../MOMoE → .../MOMOE (workspace)
HoE_ROOT = _WORKSPACE_ROOT / 'anonymous-repo-for-HoE' / 'Code' / 'HoE'


def _patch_sys_path():
    """Add HoE src/ to sys.path so hacked PEFT/LLaMA modules are importable."""
    for p in [str(HoE_ROOT / 'src'), str(HoE_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


_patch_sys_path()


def _load_adapter_weights(adapter_dir: str) -> dict:
    """Load adapter weights from safetensors or .bin, return {key: tensor}."""
    p = Path(adapter_dir)
    st_path = p / 'adapter_model.safetensors'
    bin_path = p / 'adapter_model.bin'
    if st_path.exists():
        from safetensors import safe_open
        weights = {}
        with safe_open(str(st_path), framework='pt', device='cpu') as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)
        return weights
    elif bin_path.exists():
        return torch.load(str(bin_path), map_location='cpu')
    raise FileNotFoundError(f'No adapter_model.safetensors or .bin in {adapter_dir}')


def _build_mola_state_dict(expert_model_paths: List[str]) -> dict:
    """Merge N standard LoRA adapters into a single MoLA state dict.

    MoLA key format (expected by set_peft_model_state_dict_moe):
        standard:  ...lora_A.weight
        MoLA:      ...lora_A_0.weight  (expert 0)
                   ...lora_A_1.weight  (expert 1)
    """
    mola_sd = {}
    for expert_idx, path in enumerate(expert_model_paths):
        for key, tensor in _load_adapter_weights(path).items():
            if 'lora_A.weight' in key:
                mola_sd[key.replace('lora_A.weight', f'lora_A_{expert_idx}.weight')] = tensor
            elif 'lora_B.weight' in key:
                mola_sd[key.replace('lora_B.weight', f'lora_B_{expert_idx}.weight')] = tensor
            elif expert_idx == 0:
                # non-LoRA keys (bias etc.) kept once from expert 0
                mola_sd[key] = tensor
    return mola_sd


def build_hoe_model(
    base_model_name: str,
    sft_model_name: str,
    expert_model_paths: List[str],
    number_experts: List[int],
    top_k: List[int],
    router_type: str = 'v1',
    num_rewards: int = 2,
    router_hidden_dim: int = 32,
    load_8bit: bool = False,
    device: str = 'cuda',
):
    """Build a HoE model from base LLaMA + SFT LoRA + N expert LoRA adapters.

    Mirrors the ppo.py pattern for loading base + SFT, then builds the MoLA
    multi-expert structure on the fly from the per-objective expert adapters
    (e.g. PPO-harmless, PPO-helpful) without requiring a pre-built MoLA checkpoint.

    lora_target_modules is read directly from the first expert's adapter_config.json
    to ensure consistency.

    Returns the model with model.dynamic_weights attached.
    """
    from peft import LoraConfig, PeftModel as _StdPeft
    from transformers import AutoConfig
    from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
    from src.mola_mapping_hacked import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
    from src.mola_peft_model_hacked import set_peft_model_state_dict_moe
    from hoe_architecture import ConditionedMOEModel

    # Derive target modules from expert adapters (must be consistent across experts)
    with open(Path(expert_model_paths[0]) / 'adapter_config.json') as f:
        lora_target_modules = json.load(f)['target_modules']
    print(f'[hoe_utils] lora_target_modules from expert adapter: {sorted(lora_target_modules)}')

    # ---- 1. Load base LLaMA_d + merge SFT LoRA ----
    model_config = AutoConfig.from_pretrained(base_model_name)
    model_config.lora_target_modules = lora_target_modules

    base = LlamaForCausalLM_d.from_pretrained(
        base_model_name,
        config=model_config,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16,
        device_map=device if device == 'cuda' else {'': device},
    )
    base = _StdPeft.from_pretrained(base, sft_model_name)
    base = base.merge_and_unload()

    # ---- 2. Create MoLA PeftModel (random-initialised expert weights) ----
    peft_config = LoraConfig.from_pretrained(expert_model_paths[0])
    # update_layer() only activates MoLA routing (creates self.router) when r is a list.
    # LoraConfig.r is a scalar, so we expand it to a per-layer list here.
    peft_config.r = [peft_config.r] * len(number_experts)
    mola_model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        base, peft_config, adapter_name='default',
        number_experts=number_experts, top_k=top_k,
    )

    # ---- 3. Load merged expert weights ----
    print(f'[hoe_utils] Loading {len(expert_model_paths)} expert adapter(s) into MoLA ...')
    mola_sd = _build_mola_state_dict(expert_model_paths)
    set_peft_model_state_dict_moe(mola_model, mola_sd)
    print(f'[hoe_utils] Expert weights loaded ({len(mola_sd)} keys).')

    # ---- 4. MoLA housekeeping + HoE preference-conditioned routers ----
    mola_model.get_new_parameters(number_experts, top_k, oblance=False)
    ConditionedMOEModel.init(
        mola_model,
        router_type=router_type,
        num_rewards=num_rewards,
        hidden_dim=router_hidden_dim,
    )
    return mola_model


def load_hoe_checkpoint(model, checkpoint_path: str):
    """Load trained router weights into a HoE model from a saved checkpoint directory."""
    from src.mola_peft_model_hacked import set_peft_model_state_dict_moe

    adapter_bin = os.path.join(checkpoint_path, 'adapter_model.bin')
    if not os.path.exists(adapter_bin):
        raise FileNotFoundError(f'Checkpoint not found: {adapter_bin}')

    state_dict = torch.load(adapter_bin, map_location='cpu')
    set_peft_model_state_dict_moe(model, state_dict)

    # Load value heads if present
    v_head_idx = 0
    while True:
        v_head_path = os.path.join(checkpoint_path, f'v_head{v_head_idx}.pt')
        if not os.path.exists(v_head_path):
            break
        if hasattr(model, 'v_heads') and v_head_idx < len(model.v_heads):
            model.v_heads[v_head_idx] = torch.load(v_head_path, map_location='cpu')
        v_head_idx += 1

    print(f'Loaded HoE checkpoint from {checkpoint_path}')


def parse_comma_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',')]


def parse_comma_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(',')]


def make_number_experts_str(n_experts: int, n_layers: int) -> str:
    """Build a comma-separated string of n_experts repeated n_layers times."""
    return ','.join([str(n_experts)] * n_layers)


def clean_response(response: str) -> str:
    """Strip padding tokens and truncate at conversation turn boundaries."""
    response = response.strip('[PAD] ').strip('<unk>').strip('<s>').strip('</s>')
    for sep in ['\n\nHuman:', '\nHuman:', '\n\nAssistant:', '\nAssistant:', '\n\n\n', '###']:
        response = response.split(sep)[0].strip()
    return response
