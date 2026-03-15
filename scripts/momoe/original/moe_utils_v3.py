import os
import re
import types

import numpy as np
import torch
from peft import PeftModel
from pymoo.indicators.hv import HV
from transformers import AutoModelForCausalLM

from moe_architecture_v3 import (
    build_preference_gating_network,
    MoEFFNLayer,
    ParameterCombinedFFN,
    ParameterCombinedLoRAProjection,
)


def compute_hypervolume(reward_vectors):
    if len(reward_vectors) == 0:
        return 0.0
    return HV(ref_point=np.ones(len(reward_vectors[0])))(-np.array(reward_vectors))


def resolve_gating_checkpoint_path(checkpoint_path):
    if not checkpoint_path or os.path.exists(os.path.join(checkpoint_path, "gating_weights.pt")):
        return checkpoint_path
    if not os.path.isdir(checkpoint_path):
        return checkpoint_path

    candidates = []
    for entry in os.listdir(checkpoint_path):
        subdir = os.path.join(checkpoint_path, entry)
        if not os.path.exists(os.path.join(subdir, "gating_weights.pt")):
            continue
        key = (
            int(re.search(r"step_(\d+)", entry).group(1)) if re.search(r"step_(\d+)", entry) else -1,
            int(re.search(r"epoch_(\d+)", entry).group(1)) if re.search(r"epoch_(\d+)", entry) else -1,
            os.path.getmtime(subdir),
        )
        candidates.append((key, subdir))
    return sorted(candidates, reverse=True)[0][1] if candidates else checkpoint_path


def _load_lora_state_dicts(base_model, expert_paths):
    state_dicts = []
    for expert_path in expert_paths:
        safetensor_path = os.path.join(expert_path, "adapter_model.safetensors")
        pytorch_path = os.path.join(expert_path, "adapter_model.bin")
        if os.path.exists(safetensor_path):
            from safetensors import safe_open

            state_dict = {}
            with safe_open(safetensor_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    state_dict[key] = handle.get_tensor(key)
            state_dicts.append(state_dict)
        elif os.path.exists(pytorch_path):
            state_dicts.append(torch.load(pytorch_path, map_location="cpu"))
        else:
            state_dicts.append(PeftModel.from_pretrained(base_model, expert_path).state_dict())
    return state_dicts


def _find_lora_pair(state_dict, layer_idx, proj_name, device, dtype):
    prefix = f"layers.{layer_idx}.mlp.{proj_name}"
    lora_a = next((v for k, v in state_dict.items() if prefix in k and ".lora_A" in k), None)
    lora_b = next((v for k, v in state_dict.items() if prefix in k and ".lora_B" in k), None)
    if lora_a is None or lora_b is None:
        raise KeyError(f"Missing LoRA weights for layer {layer_idx} {proj_name}")
    return lora_a.to(device=device, dtype=dtype), lora_b.to(device=device, dtype=dtype)


def convert_to_moe_model(base_model_name, lora_expert_paths, num_rewards=2, gating_type="linear", target_device=None):
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": target_device or "auto"}
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model_dtype = next(base_model.parameters()).dtype
    hidden_dim = base_model.config.hidden_size
    model_device = next(base_model.parameters()).device

    lora_state_dicts = _load_lora_state_dicts(base_model, lora_expert_paths)
    shared_gate = build_preference_gating_network(
        gating_type=gating_type,
        hidden_dim=hidden_dim,
        num_lora_experts=len(lora_state_dicts),
        num_rewards=num_rewards,
    ).to(model_device, dtype=torch.float32)
    base_model.add_module("shared_gate", shared_gate)
    object.__setattr__(base_model, "_current_preference", None)
    object.__setattr__(base_model, "_current_routing_weights", None)
    object.__setattr__(base_model, "_routing_session_active", False)

    def compute_shared_routing_weights(self, input_ids=None, attention_mask=None, inputs_embeds=None, preference=None):
        if preference is None:
            preference = getattr(self, "_current_preference", None)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("compute_shared_routing_weights requires input_ids or inputs_embeds.")
            inputs_embeds = self.get_input_embeddings()(input_ids.to(next(self.parameters()).device))
        hidden_states = inputs_embeds

        if attention_mask is not None:
            mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled_hidden = (hidden_states * mask).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
            hidden_states = pooled_hidden

        weights = self.shared_gate(hidden_states, preference=preference)
        object.__setattr__(self, "_current_routing_weights", weights)
        return weights

    def clear_shared_routing_weights(self):
        object.__setattr__(self, "_current_routing_weights", None)

    def set_preference(self, preference):
        if preference is None:
            object.__setattr__(self, "_current_preference", None)
            return
        if isinstance(preference, list):
            preference = torch.tensor(preference, dtype=torch.float32)
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)
        object.__setattr__(self, "_current_preference", preference.to(device=next(self.parameters()).device))

    def set_manual_lora_weights(self, manual_weights):
        if manual_weights is None:
            self.shared_gate.manual_weights = None
            return
        if isinstance(manual_weights, list):
            manual_weights = torch.tensor(manual_weights, dtype=torch.float32)
        self.shared_gate.manual_weights = manual_weights.to(device=next(self.parameters()).device)

    original_forward = base_model.forward

    def forward_with_shared_gate(self, *args, **kwargs):
        created_routing = False
        if getattr(self, "_current_routing_weights", None) is None:
            self.compute_shared_routing_weights(
                input_ids=kwargs.get("input_ids"),
                attention_mask=kwargs.get("attention_mask"),
                inputs_embeds=kwargs.get("inputs_embeds"),
                preference=kwargs.pop("preference", None),
            )
            created_routing = True
        try:
            return original_forward(*args, **kwargs)
        finally:
            if created_routing and not getattr(self, "_routing_session_active", False):
                self.clear_shared_routing_weights()

    original_generate = base_model.generate

    def generate_with_shared_gate(self, *args, **kwargs):
        self.clear_shared_routing_weights()
        object.__setattr__(self, "_routing_session_active", True)
        try:
            return original_generate(*args, **kwargs)
        finally:
            object.__setattr__(self, "_routing_session_active", False)
            self.clear_shared_routing_weights()

    object.__setattr__(base_model, "compute_shared_routing_weights", types.MethodType(compute_shared_routing_weights, base_model))
    object.__setattr__(base_model, "clear_shared_routing_weights", types.MethodType(clear_shared_routing_weights, base_model))
    object.__setattr__(base_model, "set_preference", types.MethodType(set_preference, base_model))
    object.__setattr__(base_model, "set_manual_lora_weights", types.MethodType(set_manual_lora_weights, base_model))
    object.__setattr__(base_model, "forward", types.MethodType(forward_with_shared_gate, base_model))
    object.__setattr__(base_model, "generate", types.MethodType(generate_with_shared_gate, base_model))

    for layer_idx, layer in enumerate(base_model.model.layers):
        base_mlp = layer.mlp
        device = next(base_mlp.parameters()).device
        projections = {}
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            lora_as, lora_bs = [], []
            for state_dict in lora_state_dicts:
                lora_a, lora_b = _find_lora_pair(state_dict, layer_idx, proj_name, device, model_dtype)
                lora_as.append(lora_a)
                lora_bs.append(lora_b)
            projections[proj_name] = ParameterCombinedLoRAProjection(
                torch.stack(lora_as, dim=0),
                torch.stack(lora_bs, dim=0),
            )

        layer.mlp = MoEFFNLayer(
            parameterized_ffn=ParameterCombinedFFN(
                base_mlp=base_mlp,
                gate_proj_lora=projections["gate_proj"],
                up_proj_lora=projections["up_proj"],
                down_proj_lora=projections["down_proj"],
            ),
        )

    for layer in base_model.model.layers:
        if isinstance(layer.mlp, MoEFFNLayer):
            layer.mlp.set_model_ref(base_model)
    return base_model


def save_moe_gating_weights(model, save_path):
    os.makedirs(save_path, exist_ok=True)
    core_model = model.module if hasattr(model, "module") else model
    torch.save(
        {name: param.cpu() for name, param in core_model.named_parameters() if "gate" in name and param.requires_grad},
        os.path.join(save_path, "gating_weights.pt"),
    )


def load_moe_gating_weights(model, save_path):
    weights_path = os.path.join(resolve_gating_checkpoint_path(save_path), "gating_weights.pt")
    if not os.path.exists(weights_path):
        return False

    core_model = model.module if hasattr(model, "module") else model
    target_state = core_model.state_dict()
    loaded_state = torch.load(weights_path, map_location=next(core_model.parameters()).device)
    filtered_state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in loaded_state.items()
        if isinstance(name, str) and "gate" in name and (name[7:] if name.startswith("module.") else name) in target_state
    }
    if not filtered_state:
        return False
    core_model.load_state_dict(filtered_state, strict=False)
    return True
