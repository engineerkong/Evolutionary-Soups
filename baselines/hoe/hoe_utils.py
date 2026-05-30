"""HoE utility functions: model construction, checkpoint I/O, text cleaning."""
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Locate the HoE source tree relative to this file:
#   baselines/hoe/  →  Evolutionary_Soups/  →  workspace root  →  anonymous-repo-for-HoE/
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../Evolutionary_Soups/baselines/hoe
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # .../Evolutionary_Soups/scripts → .../Evolutionary_Soups
_WORKSPACE_ROOT = _PROJECT_ROOT.parent                 # .../Evolutionary_Soups → .../Evolutionary_Soups (workspace)
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
    from peft import LoraConfig
    from transformers import AutoConfig
    from src.mola_modeling_llama_hacked import LlamaForCausalLM_d
    from src.mola_mapping_hacked import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
    from src.mola_peft_model_hacked import set_peft_model_state_dict_moe
    from baselines.hoe.hoe_architecture import ConditionedMOEModel

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


def save_router_weights(model, save_path: str):
    """Save only router parameters to router_weights.pt."""
    router_sd = {name: param.detach().cpu()
                 for name, param in model.named_parameters()
                 if 'router' in name}
    torch.save(router_sd, os.path.join(save_path, 'router_weights.pt'))
    print(f'[hoe_utils] Saved {len(router_sd)} router parameter tensors to {save_path}.')


def load_router_weights(model, pretrained_moe_path: str):
    """Inject pre-trained router weights (from Stage 1) into a freshly built MoLA model."""
    router_pt = os.path.join(pretrained_moe_path, 'router_weights.pt')
    if not os.path.exists(router_pt):
        raise FileNotFoundError(f'Router weights not found: {router_pt}')
    router_sd = torch.load(router_pt, map_location='cpu')
    current   = dict(model.named_parameters())
    loaded, missing = 0, []
    for name, tensor in router_sd.items():
        if name in current:
            current[name].data.copy_(tensor)
            loaded += 1
        else:
            missing.append(name)
    if missing:
        print(f'[hoe_utils] WARNING: {len(missing)} router keys not found in model: {missing[:5]}')
    print(f'[hoe_utils] Loaded {loaded}/{len(router_sd)} router weights from {pretrained_moe_path}.')


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
