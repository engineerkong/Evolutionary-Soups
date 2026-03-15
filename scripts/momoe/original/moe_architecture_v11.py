import sys
import os
import weakref
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from pymoo.indicators.hv import HV

script_dir = Path(__file__).resolve().parent  # project/scripts/momoe
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_reward_model, get_rewards, get_clean_data


# ==================== LoRA Expert FFN Complete ====================
    
class LoRAExpertFFNComplete(nn.Module):
    """
    Complete FFN output (base + LoRA)
    The increment is obtained by subtracting base_output in MoEFFNLayer
    """
    def __init__(self, base_gate_proj, base_up_proj, base_down_proj,
                 gate_proj_lora, up_proj_lora, down_proj_lora, act_fn):
        super().__init__()
        self.base_gate_proj = base_gate_proj
        self.base_up_proj = base_up_proj
        self.base_down_proj = base_down_proj
        
        self.gate_proj_lora = gate_proj_lora
        self.up_proj_lora = up_proj_lora
        self.down_proj_lora = down_proj_lora
        
        self.act_fn = act_fn
    
    def forward(self, x):
        """
        Complete FFN: FFN(x) = down(SiLU(gate(x)) * up(x))
        where gate, up, down are all base + LoRA
        """
        # Gate: base + LoRA -> activate
        gate = self.base_gate_proj(x) + self.gate_proj_lora(x)
        gate = self.act_fn(gate)
        
        # Up: base + LoRA
        up = self.base_up_proj(x) + self.up_proj_lora(x)
        
        # Interaction
        intermediate = gate * up
        
        # Down: base + LoRA
        output = self.base_down_proj(intermediate) + self.down_proj_lora(intermediate)
        
        return output

# ==================== Preference-Conditioned Gating Network ====================

class AttentionGatingNetwork(nn.Module):
    """
    Simple preference-conditioned linear router.
    Router input is concatenated [pooled_hidden_state, preference].
    """
    def __init__(self, hidden_dim, num_lora_experts, subspace_rank=8, d_model=512, num_rewards=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lora_experts = num_lora_experts
        self.num_rewards = num_rewards

        # Keep signature compatibility with existing callers (subspace_rank, d_model),
        # but route with a simple linear head on concatenated features.
        self.router = nn.Linear(hidden_dim + num_rewards, num_lora_experts)
        # Phase-2 policy update flag: when True, build graph for routing branch
        # even if outer model forward runs under torch.no_grad().
        self._force_policy_grad = False
        # When enabled, cache per-step selected log-probs during generate().
        self._collect_policy_trajectory = False
        self._selected_log_probs_history = []
    
    def forward(self, x, preference=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            preference: [batch, num_rewards] or [num_rewards] -- REQUIRED in practice.
                        Falls back to uniform only as a safety net.
        Returns:
            lora_weights: [batch, seq_len, num_lora_experts]
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        if self.router.weight.device != device:
            self.to(device)
        
        # Resolve preference tensor
        if preference is not None:
            if isinstance(preference, list):
                preference = torch.tensor(preference, dtype=dtype, device=device)
            if preference.dim() == 1:
                preference = preference.unsqueeze(0)
            pref = preference.to(device=device, dtype=dtype)
        else:
            # Uniform fallback — should not happen during training
            pref = torch.ones(1, self.num_rewards, device=device, dtype=dtype) / self.num_rewards
        
        if pref.shape[0] == 1 and batch > 1:
            pref = pref.expand(batch, -1)  # [batch, num_rewards]

        if self._force_policy_grad:
            # Keep policy-gradient graph only for router branch, detached from
            # backbone activations to avoid retaining the full LLM graph.
            with torch.enable_grad():
                x_local = x.detach()
                pref_local = pref.detach()
                x_pooled = x_local.mean(dim=1)  # [batch, hidden_dim]
                router_input = torch.cat([x_pooled, pref_local], dim=-1)
                scores = self.router(router_input)                  # [batch, num_experts]
                lora_weights_seq = F.softmax(scores, dim=-1)        # [batch, num_experts]
        else:
            x_pooled = x.mean(dim=1)  # [batch, hidden_dim]
            router_input = torch.cat([x_pooled, pref], dim=-1)      # [batch, hidden_dim + num_rewards]
            scores = self.router(router_input)                       # [batch, num_experts]
            lora_weights_seq = F.softmax(scores, dim=-1)             # [batch, num_experts]

        if self._force_policy_grad or torch.is_grad_enabled():
            self._last_router_scores = scores
        else:
            self._last_router_scores = scores.detach()

        # Broadcast to all token positions: [batch, seq_len, num_experts]
        # if not self.training:
        #     lora_weights_seq = torch.zeros(batch, self.num_lora_experts, device=device, dtype=dtype)
        #     lora_weights_seq[:, 0] = 1.0  # force expert-0 => [1, 0]
        lora_weights = lora_weights_seq.unsqueeze(1).expand(-1, seq_len, -1)

        # Store for REINFORCE loss computation.
        # Use grad-mode rather than module.training so policy updates still work
        # even if a gate module is accidentally left in eval mode.
        if self._force_policy_grad or torch.is_grad_enabled():
            self._last_routing_weights = lora_weights_seq          # [batch, num_experts]
        else:
            self._last_routing_weights = lora_weights_seq.detach() # [batch, num_experts]

        if self._collect_policy_trajectory:
            # generate() usually runs under torch.no_grad(); force autograd tracking
            # for policy trajectory terms so REINFORCE loss has a valid graph.
            with torch.enable_grad():
                step_log_probs = torch.log(lora_weights_seq + 1e-8)  # [batch, num_experts]
                sampled_actions = torch.distributions.Categorical(probs=lora_weights_seq).sample()  # [batch]
                selected_log_probs = step_log_probs.gather(
                    dim=1,
                    index=sampled_actions.unsqueeze(1)
                ).squeeze(1)  # [batch]
            self._selected_log_probs_history.append(selected_log_probs)

        return lora_weights


# ==================== Preference-Conditioned MoE FFN Layer ====================

class MoEFFNLayer(nn.Module):
    """
    MoE FFN with preference-conditioned gating.

    [FIX 1] Preference is read from the top-level model object at every
    forward() call (including inside model.generate()), so generation and
    training always use the same up-to-date preference.

    [CIRCULAR REF FIX] The top-level model reference is stored as a weakref
    (a plain Python attribute, NOT an nn.Module attribute) to avoid the
    nn.Module.__setattr__ recursion that fires when an nn.Module is assigned
    as an attribute of another nn.Module that is already its descendant.
    Using weakref.ref() keeps the reference outside PyTorch's module registry,
    breaking the cycle:
        base_model -> layers[i].mlp -> _model_ref -> base_model -> ...
    """
    def __init__(self, base_mlp, lora_experts, gate_network, model_ref=None):
        super().__init__()
        self.base_mlp = base_mlp
        self.lora_experts = nn.ModuleList(lora_experts)
        self.gate = gate_network
        self.num_lora_experts = len(lora_experts)
        # Store as a plain Python attribute (not via nn.Module.__setattr__)
        # so PyTorch does not try to register it as a child module.
        # Use object.__setattr__ to bypass nn.Module's __setattr__ entirely.
        object.__setattr__(self, '_model_weakref', None)
        if model_ref is not None:
            object.__setattr__(self, '_model_weakref', weakref.ref(model_ref))

    def set_model_ref(self, model_ref):
        """
        Wire up the top-level model reference after construction.
        Must use object.__setattr__ to prevent nn.Module from registering
        the weakref wrapper as a submodule.
        """
        object.__setattr__(self, '_model_weakref', weakref.ref(model_ref))

    def forward(self, hidden_states, **kwargs):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # [FIX 1] Dereference the weakref to get the live model object,
        # then read _current_preference set by trainer.set_model_preference().
        preference = None
        model_weakref = object.__getattribute__(self, '_model_weakref')
        if model_weakref is not None:
            model = model_weakref()   # dereference; returns None if GC'd
            if model is not None:
                preference = getattr(model, '_current_preference', None)

        # 1. Base model output
        base_output = self.base_mlp(hidden_states)
        
        # 2. Get routing weights (per-sequence, broadcast to seq_len)
        lora_weights = self.gate(hidden_states, preference=preference)
        
        # 3. Compute weighted LoRA expert deltas
        lora_contribution = torch.zeros_like(hidden_states)
        for expert_idx, expert in enumerate(self.lora_experts):
            expert_full_output = expert(hidden_states)
            expert_delta = expert_full_output - base_output
            weight = lora_weights[:, :, expert_idx].unsqueeze(-1)
            lora_contribution += weight * expert_delta
        
        # 4. Combine
        final_output = base_output + lora_contribution
        return final_output


# ==================== Preference-Conditioned Trainer with Hypervolume ====================

class MoEGatingTrainer:
    """
    Trainer for MoE gating with preference conditioning and hypervolume loss.

    Changes vs original:
      [FIX 1] set_model_preference() writes to model._current_preference (top-level),
              removing all per-layer set_preference() calls.
      [FIX 2] forward_with_routing_log_probs() passes preference explicitly via
              model._current_preference; no side-channel needed.
      [FIX 3] REINFORCE loss uses the FULL routing distribution (expected log-prob
              under the soft distribution) instead of max_log_probs.  This makes
              the training objective consistent with the soft weighted-sum inference.
    """
    
    def __init__(self, moe_model, reward_models, instructions, learning_rate=1e-5,
                 num_rewards=2, num_pref_samples=10):
        self.model = moe_model
        core_model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(core_model, "gradient_checkpointing_enable"):
            core_model.gradient_checkpointing_enable()
        self.reward_models = reward_models
        self.instructions = instructions
        self.num_rewards = num_rewards
        self.num_pref_samples = num_pref_samples
        
        print(f"Initializing Preference-Conditioned MoE trainer with {num_rewards} rewards")
        print(f"Number of preference samples per input: {num_pref_samples}")
        
        # Collect gating parameters
        gating_params = []
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate'):
                gating_params.extend(layer.mlp.gate.parameters())
        
        self.optimizer = torch.optim.AdamW(gating_params, lr=learning_rate)
        
        # Freeze all parameters, then unfreeze gating
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate'):
                for param in layer.mlp.gate.parameters():
                    param.requires_grad = True
        
        print(f"Trainable gating parameters: {sum(p.numel() for p in gating_params):,}")
        
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9
        self.hv_baseline = 0.0
        # Preference-conditioned scalarized baseline.
        self.pref_reward_baseline = {}
        # Debug switch: MOMOE_DEBUG_GRAD=1
        self._debug_grad = os.getenv("MOMOE_DEBUG_GRAD", "0") == "1"

    def _core_model(self):
        """Return the underlying module when wrapped (e.g., DDP)."""
        return self.model.module if hasattr(self.model, 'module') else self.model
    
    def sample_preferences(self):
        """Sample preferences from Dirichlet distribution (uniform on simplex)."""
        preferences = []
        for _ in range(self.num_pref_samples):
            pref = np.random.dirichlet(np.ones(self.num_rewards))
            preferences.append(pref.tolist())
        return preferences

    def _pref_key(self, preference, ndigits=2):
        pref = np.asarray(preference, dtype=np.float32)
        return tuple(np.round(pref, decimals=ndigits).tolist())
    
    def set_model_preference(self, preference):
        """
        [FIX 1] Write preference to top-level model object.
        All MoEFFNLayer.forward() calls (including inside generate()) will read
        from model._current_preference via their _model_ref.
        Removes all per-layer gate.set_preference() calls.
        """
        if isinstance(preference, (list, np.ndarray)):
            pref_tensor = torch.tensor(preference, dtype=torch.float32)
        else:
            pref_tensor = preference
        # Store as [1, num_rewards] on CPU; MoEFFNLayer will move to correct device
        if pref_tensor.dim() == 1:
            pref_tensor = pref_tensor.unsqueeze(0)
        model_device = next(self.model.parameters()).device
        pref_tensor = pref_tensor.to(device=model_device)
        object.__setattr__(self._core_model(), '_current_preference', pref_tensor)
        # self.model._current_preference = pref_tensor

    def compute_hypervolume(self, reward_vectors):
        if len(reward_vectors) == 0:
            return 0.0
        points = -np.array(reward_vectors)
        ref_point = np.ones(len(reward_vectors[0])) * 1.0
        hv_indicator = HV(ref_point=ref_point)
        return hv_indicator(points)

    def compute_load_balance(self):
        """Compute load balance loss across all gating layers."""
        total_balance_loss = 0.0
        num_layers = 0
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, '_last_routing_weights'):
                # _last_routing_weights is now [batch, num_experts] (per-sequence)
                routing_weights = layer.mlp.gate._last_routing_weights
                expert_usage = routing_weights.mean(dim=0)  # [num_experts]
                target = 1.0 / layer.mlp.num_lora_experts
                balance_loss = ((expert_usage - target) ** 2).mean()
                total_balance_loss += balance_loss
                num_layers += 1
        
        if num_layers == 0:
            return torch.tensor(0.0, requires_grad=True,
                                device=next(self.model.parameters()).device)
        return total_balance_loss / num_layers

    def _enable_policy_trajectory_capture(self):
        """Enable per-step sampled log-prob capture inside generate()."""
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate'):
                gate = layer.mlp.gate
                gate._force_policy_grad = True
                gate._collect_policy_trajectory = True
                gate._selected_log_probs_history = []

    def _disable_policy_trajectory_capture(self):
        """Disable trajectory capture and reset runtime flags."""
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate'):
                gate = layer.mlp.gate
                gate._force_policy_grad = False
                gate._collect_policy_trajectory = False

    def _gather_selected_log_probs_trajectory(self):
        """
        Gather selected log-probs from all MoE layers and all generation steps.
        Returns a flat list of tensors, each with shape [batch].
        """
        selected_log_probs = []
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate'):
                gate = layer.mlp.gate
                if hasattr(gate, '_selected_log_probs_history'):
                    selected_log_probs.extend(gate._selected_log_probs_history)
        return selected_log_probs

    def _debug_print_router_weight_stats(self, tag=""):
        """Print routing weight statistics aggregated across all MoE layers."""
        if not self._debug_grad:
            return

        layer_means = []
        layer_stds = []
        layer_abs_diffs = []
        num_layers = 0

        for layer in self._core_model().model.layers:
            if not hasattr(layer.mlp, 'gate'):
                continue
            gate = layer.mlp.gate
            if not hasattr(gate, '_last_routing_weights'):
                continue
            weights = gate._last_routing_weights.detach().float()  # [batch, num_experts]
            if weights.numel() == 0:
                continue
            num_layers += 1
            layer_means.append(weights.mean().item())
            layer_stds.append(weights.std(unbiased=False).item())
            if weights.shape[-1] == 2:
                layer_abs_diffs.append((weights[:, 0] - weights[:, 1]).abs().mean().item())

        if num_layers == 0:
            print(f"[DEBUG][Weights]{tag} no_routing_weights")
            return

        weight_mean = float(np.mean(layer_means))
        weight_std = float(np.mean(layer_stds))
        if layer_abs_diffs:
            abs_diff = float(np.mean(layer_abs_diffs))
            print(
                f"[DEBUG][Weights]{tag} layers={num_layers} "
                f"weight_mean={weight_mean:.6f} weight_std={weight_std:.6f} "
                f"abs_weight_diff_01={abs_diff:.6f}"
            )
        else:
            print(
                f"[DEBUG][Weights]{tag} layers={num_layers} "
                f"weight_mean={weight_mean:.6f} weight_std={weight_std:.6f}"
            )

    def _debug_print_router_grad_status(self, tag=""):
        """Print a compact gradient-health summary for router parameters."""
        if not self._debug_grad:
            return
        grad_norm_sq = 0.0
        num_with_grad = 0
        num_total = 0
        sample_name = None
        sample_grad_mean = None

        for layer_idx, layer in enumerate(self._core_model().model.layers):
            if not hasattr(layer.mlp, 'gate'):
                continue
            for name, param in layer.mlp.gate.named_parameters():
                num_total += 1
                if param.grad is None:
                    continue
                num_with_grad += 1
                g = param.grad.detach()
                grad_norm_sq += (g * g).sum().item()
                if sample_name is None:
                    sample_name = f"layer{layer_idx}.{name}"
                    sample_grad_mean = g.abs().mean().item()

        grad_norm = grad_norm_sq ** 0.5
        print(
            f"[DEBUG][Grad]{tag} router_params_with_grad={num_with_grad}/{num_total} "
            f"global_grad_norm={grad_norm:.6f} "
            f"sample={sample_name} sample_abs_mean={sample_grad_mean}"
        )
        
    def forward_with_routing_log_probs(self, input_ids, attention_mask=None, preference=None):
        """
        Forward pass collecting routing log probs with preference conditioning.

        [FIX 1] Preference is set on the model object before forward(); the
        MoEFFNLayer reads it from model._current_preference automatically.
        No per-layer set_preference() needed.
        """
        # Clear previous routing weights
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate'):
                if self.model.training:
                    layer.mlp.gate.train()
                # Enable router-only grad capture for phase-2 policy update.
                layer.mlp.gate._force_policy_grad = True
                if hasattr(layer.mlp.gate, '_last_routing_weights'):
                    del layer.mlp.gate._last_routing_weights

        # [FIX 1] Set preference on top-level model object
        if preference is not None:
            self.set_model_preference(preference)
        
        # Forward pass — MoEFFNLayer will pick up preference from model._current_preference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
        
        # Collect routing log probs
        # _last_routing_weights is now [batch, num_experts] (per-sequence, no seq_len dim)
        routing_log_probs = []
        for layer in self._core_model().model.layers:
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, '_last_routing_weights'):
                routing_weights = layer.mlp.gate._last_routing_weights  # [batch, num_experts]
                log_probs = torch.log(routing_weights + 1e-8)            # [batch, num_experts]
                routing_log_probs.append(log_probs)
            if hasattr(layer.mlp, 'gate'):
                layer.mlp.gate._force_policy_grad = False
        
        return outputs, routing_log_probs
    
    def train_step_reinforce(self, batch, tokenizer,
                            alpha_hypervolume=0.1, alpha_balance=0.1, alpha_entropy=0.01,
                            update_per_preference=True):
        """
        Standard REINFORCE update for router parameters.

        Policy gradient objective:
            L = -E[(R - b) * log pi(a|s)]
        where b is a moving-average reward baseline.
        """
        # ---- Prepare inputs ----
        input_ids_list = []
        for ids in batch['input_ids']:
            if isinstance(ids, torch.Tensor):
                input_ids_list.append(ids.clone().detach())
            else:
                input_ids_list.append(torch.tensor(ids))
        
        max_length = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        attention_mask = []
        for ids in input_ids_list:
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([
                torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype),
                ids
            ])
            mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long)
            ])
            padded_input_ids.append(padded_ids)
            attention_mask.append(mask)
        
        input_ids = torch.stack(padded_input_ids).to(next(self.model.parameters()).device)
        attention_mask = torch.stack(attention_mask).to(next(self.model.parameters()).device)
        batch_size = input_ids.shape[0]
        
        sampled_preferences = self.sample_preferences()
        sampled_preferences = [[1.0, 0.0]]
        all_scalarized_rewards = []

        # Store generated trajectories and scalarized rewards for policy update
        pref_generation_data = []

        # ========== Phase 1: Sample trajectories and rewards (soft routing, no grad) ==========
        self.model.eval()
        for pref in sampled_preferences:
            self.set_model_preference(pref)
            with torch.no_grad():
                # HF generate() returns full sequences (prompt + sampled continuation).
                generated_sequences = self._core_model().generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            full_attention_mask = (generated_sequences != tokenizer.pad_token_id).long()

            full_responses = tokenizer.batch_decode(generated_sequences)
            full_prompts = tokenizer.batch_decode(input_ids)
            full_prompts, full_responses = get_clean_data(full_responses, full_prompts)
            
            queries_responses = [
                (self.instructions.get_input(text), self.instructions.get_response(text))
                for text in full_responses
            ]
            
            if hasattr(self.instructions, 'get_post'):
                rewards_list = self.reward_models.get_reward_model_scores(
                    queries_responses, self.instructions.get_post, normalize_rewards=False
                )
            else:
                rewards_list = self.reward_models.get_reward_model_scores(
                    queries_responses, normalize_rewards=False
                )
            
            pref_rewards = []
            for j in range(batch_size):
                reward_vector = [rewards_list[k][j] for k in range(self.num_rewards)]
                scalarized = float(sum(pref[k] * reward_vector[k] for k in range(self.num_rewards)))
                all_scalarized_rewards.append(scalarized)
                pref_rewards.append(scalarized)

            pref_generation_data.append({
                'pref': pref,
                'generated_sequences': generated_sequences,
                'full_attention_mask': full_attention_mask,
                'pref_rewards': pref_rewards,
            })
            
            del generated_sequences
            torch.cuda.empty_cache()
        
        # ========== Phase 2: Soft-routing policy update ==========
        self.model.train()
        num_prefs = max(1, len(pref_generation_data))
        accumulated_policy_loss = 0.0
        accumulated_total_loss = 0.0

        if not update_per_preference:
            self.optimizer.zero_grad()

        for pref_data in pref_generation_data:
            pref = pref_data['pref']
            full_input_ids = pref_data['generated_sequences']
            full_attention_mask = pref_data['full_attention_mask']
            pref_rewards = pref_data['pref_rewards']

            if update_per_preference:
                self.optimizer.zero_grad()
            
            pref_rewards_tensor = torch.tensor(
                pref_rewards,
                dtype=torch.float32,
                device=full_input_ids.device
            )
            advantages = pref_rewards_tensor - self.reward_baseline

            with torch.enable_grad():
                _, routing_log_probs = self.forward_with_routing_log_probs(
                    full_input_ids,
                    full_attention_mask,
                    preference=pref
                )

            if len(routing_log_probs) == 0:
                continue
            self._debug_print_router_weight_stats(tag=f"[pref={pref}]")

            pref_policy_loss = 0.0
            for log_probs in routing_log_probs:
                probs = torch.exp(log_probs)
                expected_log_probs = (probs * log_probs).sum(dim=-1)
                pref_policy_loss += -(advantages.detach() * expected_log_probs).mean()
            pref_total_loss = pref_policy_loss

            if update_per_preference:
                policy_val = pref_policy_loss.item() if isinstance(pref_policy_loss, torch.Tensor) else float(pref_policy_loss)
                total_val = pref_total_loss.item() if isinstance(pref_total_loss, torch.Tensor) else float(pref_total_loss)
                print(f"Pref {pref}: "
                      f"Policy Loss={policy_val:.4f}, "
                      f"Total Loss={total_val:.4f}")
                if isinstance(pref_total_loss, torch.Tensor) and pref_total_loss.requires_grad:
                    pref_total_loss.backward()
                    self._debug_print_router_grad_status(tag=f"[pref={pref}]")
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    self.optimizer.step()
                else:
                    print(
                        "Warning: pref_total_loss has no grad_fn; skipping optimizer step for this preference. "
                        f"routing_layers={len(routing_log_probs)}"
                    )

            accumulated_policy_loss += pref_policy_loss / num_prefs
            accumulated_total_loss += pref_total_loss / num_prefs

            del routing_log_probs
            torch.cuda.empty_cache()
        
        # Update baseline
        rewards_tensor = torch.tensor(all_scalarized_rewards, dtype=torch.float32)
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline +
            (1 - self.baseline_momentum) * rewards_tensor.mean().item()
        )

        if not update_per_preference:
            accumulated_total_loss.backward()
            self._debug_print_router_grad_status(tag="[accumulated]")
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()
        
        return {
            'policy_loss': accumulated_policy_loss.item() if isinstance(accumulated_policy_loss, torch.Tensor) else accumulated_policy_loss,
            'hv_policy_loss': 0.0,
            'balance_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': accumulated_total_loss.item() if isinstance(accumulated_total_loss, torch.Tensor) else float(accumulated_total_loss),
            'mean_reward': rewards_tensor.mean().item(),
            'std_reward': rewards_tensor.std(unbiased=False).item(),
            'baseline': self.reward_baseline
        }
