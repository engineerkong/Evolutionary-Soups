import os
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from pymoo.indicators.hv import HV
from scripts.momoe.moe_architecture_v2 import (
    LoRAExpertFFNComplete,
    MoEFFNLayer,
    AttentionGatingNetwork
)

# ==================== hypervolume calculation ====================

def compute_hypervolume(reward_vectors):
    """
    Compute hypervolume of reward vectors.
    
    Args:
        reward_vectors: list of [num_rewards] arrays
    Returns:
        hv_value: hypervolume (higher is better)
    """
    if len(reward_vectors) == 0:
        return 0.0
    
    # Stack and negate (pymoo expects minimization)
    points = -np.array(reward_vectors)
    
    # Reference point (negated since we negated objectives).
    # Keep aligned with trainer-side HV definition in moe_architecture_v2.py.
    ref_point = np.ones(len(reward_vectors[0])) * 1.0
    
    hv_indicator = HV(ref_point=ref_point)
    hv_value = hv_indicator(points)
    
    return hv_value


def compute_hv_contributions(reward_vectors):
    """
    Leave-one-out hypervolume contributions.

    Args:
        reward_vectors: list of [num_rewards] arrays
    Returns:
        contributions: list[float], same length as reward_vectors
    """
    if len(reward_vectors) == 0:
        return []
    if len(reward_vectors) == 1:
        return [compute_hypervolume(reward_vectors)]

    full_hv = compute_hypervolume(reward_vectors)
    contributions = []
    for i in range(len(reward_vectors)):
        reduced = reward_vectors[:i] + reward_vectors[i + 1:]
        reduced_hv = compute_hypervolume(reduced)
        contributions.append(full_hv - reduced_hv)
    return contributions


def resolve_gating_checkpoint_path(checkpoint_path):
    """
    Resolve checkpoint directory that contains gating_weights.pt.
    If checkpoint_path already points to such a directory, return it.
    Otherwise search direct subdirectories and pick the latest by step/epoch/batch.
    """
    if not checkpoint_path:
        return checkpoint_path

    direct_weights = os.path.join(checkpoint_path, "gating_weights.pt")
    if os.path.exists(direct_weights):
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        return checkpoint_path

    candidates = []
    for entry in os.listdir(checkpoint_path):
        subdir = os.path.join(checkpoint_path, entry)
        if not os.path.isdir(subdir):
            continue
        if not os.path.exists(os.path.join(subdir, "gating_weights.pt")):
            continue

        step_match = re.search(r"step_(\d+)", entry)
        epoch_match = re.search(r"epoch_(\d+)", entry)
        batch_match = re.search(r"batch_(\d+)", entry)
        step = int(step_match.group(1)) if step_match else -1
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        batch = int(batch_match.group(1)) if batch_match else -1
        mtime = os.path.getmtime(subdir)
        candidates.append(((step, epoch, batch, mtime), subdir))

    if not candidates:
        return checkpoint_path

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# ==================== key function: convert architectures ====================

def convert_to_moe_model(base_model_name, lora_expert_paths, subspace_rank=8, d_model=512, num_rewards=2, target_device=None):
    """
    Convert LLaMA base model to MOMoE model (Preference-Conditioned)
    
    Args:
        base_model_name: base model path
        lora_expert_paths: LoRA expert paths list
        subspace_rank: subspace rank
        d_model: hidden dimension of attention mechanism
        num_rewards: number of reward objectives for preference conditioning
        target_device: target device for the model (for distributed training)
    
    Returns:
        moe_model: converted MOMoE model
    """
    print(f"Loading base model: {base_model_name}")
    if target_device is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=target_device
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype 
    print(f"Base model is on device: {model_device}, dtype: {model_dtype}")

    # Load LoRA experts - load state dicts only for efficiency
    print(f"Loading {len(lora_expert_paths)} LoRA experts...")
    lora_state_dicts = []
    for expert_path in lora_expert_paths:
        print(f"  Loading: {expert_path}")
        safetensor_path = os.path.join(expert_path, "adapter_model.safetensors")
        pytorch_path = os.path.join(expert_path, "adapter_model.bin")
        if os.path.exists(safetensor_path):
            from safetensors import safe_open
            state_dict = {}
            with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            lora_state_dicts.append(state_dict)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
            lora_state_dicts.append(state_dict)
        else:
            lora_model = PeftModel.from_pretrained(base_model, expert_path)
            lora_state_dicts.append(lora_model.state_dict())
            del lora_model
    
    num_lora_experts = len(lora_state_dicts)
    hidden_dim = base_model.config.hidden_size
    num_layers = base_model.config.num_hidden_layers
    
    print(f"Converting {num_layers} layers to MOMoE (Preference-Conditioned)...")
    
    # Helper to create LoRA layer from state dict
    def create_lora_layer(state_dict, layer_idx, proj_name, device, dtype):
        lora_A_key = None
        lora_B_key = None
        for key in state_dict.keys():
            if f"layers.{layer_idx}.mlp.{proj_name}.lora_A" in key:
                lora_A_key = key
            elif f"layers.{layer_idx}.mlp.{proj_name}.lora_B" in key:
                lora_B_key = key
        if lora_A_key is None or lora_B_key is None:
            raise KeyError(f"Could not find LoRA weights for layer {layer_idx} {proj_name}")
        
        lora_A_weight = state_dict[lora_A_key].to(device=device, dtype=dtype)
        lora_B_weight = state_dict[lora_B_key].to(device=device, dtype=dtype)
        
        class LoraWrapper(nn.Module):
            def __init__(self, lora_A_w, lora_B_w):
                super().__init__()
                self.lora_A = nn.ModuleDict({'default': nn.Module()})
                self.lora_B = nn.ModuleDict({'default': nn.Module()})
                self.lora_A['default'].weight = nn.Parameter(lora_A_w)
                self.lora_B['default'].weight = nn.Parameter(lora_B_w)
            def forward(self, x):
                return (x @ self.lora_A['default'].weight.t()) @ self.lora_B['default'].weight.t()
        
        return LoraWrapper(lora_A_weight, lora_B_weight)
    
    # Enumerate each layer and replace MLP
    for layer_idx in range(num_layers):
        if layer_idx % 5 == 0:
            print(f"Processing layer {layer_idx}/{num_layers}...")
        
        base_mlp = base_model.model.layers[layer_idx].mlp
        layer_device = next(base_mlp.parameters()).device
        
        lora_experts = []
        for state_dict in lora_state_dicts:
            gate_lora = create_lora_layer(state_dict, layer_idx, 'gate_proj', layer_device, model_dtype)
            up_lora = create_lora_layer(state_dict, layer_idx, 'up_proj', layer_device, model_dtype)
            down_lora = create_lora_layer(state_dict, layer_idx, 'down_proj', layer_device, model_dtype)
            
            expert = LoRAExpertFFNComplete(
                base_gate_proj=base_mlp.gate_proj,
                base_up_proj=base_mlp.up_proj,
                base_down_proj=base_mlp.down_proj,
                gate_proj_lora=gate_lora,
                up_proj_lora=up_lora,
                down_proj_lora=down_lora,
                act_fn=nn.SiLU()
            )
            lora_experts.append(expert)
        
        gate_network = AttentionGatingNetwork(
            hidden_dim=hidden_dim,
            num_lora_experts=num_lora_experts,
            subspace_rank=subspace_rank,
            d_model=d_model,
            num_rewards=num_rewards
        )
        gate_network = gate_network.to(layer_device, dtype=model_dtype)

        # [FIX 1] Pass model_ref=None initially; wired up after full construction below.
        moe_ffn = MoEFFNLayer(
            base_mlp=base_mlp,
            lora_experts=lora_experts,
            gate_network=gate_network,
            model_ref=None  # set after construction
        )
        
        base_model.model.layers[layer_idx].mlp = moe_ffn
    
    # [FIX 1] After all layers are replaced, wire the top-level model reference
    # into every MoEFFNLayer so they can read model._current_preference at
    # forward() time (including inside model.generate()).
    print("Wiring model references into MoEFFN layers...")
    for layer_idx in range(num_layers):
        mlp = base_model.model.layers[layer_idx].mlp
        if isinstance(mlp, MoEFFNLayer):
            mlp.set_model_ref(base_model)

    print("MoE conversion complete!")
    return base_model


# ==================== MoE Gating Weights Save/Load ====================

def save_moe_gating_weights(model, save_path):
    """Save only the trainable gating network weights"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    gating_state_dict = {}
    for name, param in model.named_parameters():
        if 'gate' in name and param.requires_grad:
            gating_state_dict[name] = param.cpu()
    
    torch.save(gating_state_dict, os.path.join(save_path, 'gating_weights.pt'))
    print(f"Saved gating weights to {save_path}/gating_weights.pt")

def load_moe_gating_weights(model, save_path):
    """Load gating network weights"""
    import os
    resolved_path = resolve_gating_checkpoint_path(save_path)
    weights_path = os.path.join(resolved_path, 'gating_weights.pt')
    
    if os.path.exists(weights_path):
        gating_state_dict = torch.load(weights_path, map_location=model.device)
        model.load_state_dict(gating_state_dict, strict=False)
        print(f"Loaded gating weights from {weights_path}")
        return True
    else:
        print(f"No gating weights found at {weights_path}")
        return False
