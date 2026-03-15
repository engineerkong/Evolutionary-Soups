import sys
import weakref
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import get_clean_data


class BasePreferenceGatingNetwork(nn.Module):
    def __init__(self, hidden_dim, num_lora_experts, num_rewards=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lora_experts = num_lora_experts
        self.num_rewards = num_rewards
        self._force_policy_grad = False
        self.manual_weights = None

    def _prepare_preference(self, hidden_states, preference):
        batch = hidden_states.shape[0]
        device = hidden_states.device
        dtype = next(self.parameters()).dtype

        if preference is None:
            preference = torch.ones(1, self.num_rewards, device=device, dtype=dtype) / self.num_rewards
        elif isinstance(preference, list):
            preference = torch.tensor(preference, device=device, dtype=dtype)

        if preference.dim() == 1:
            preference = preference.unsqueeze(0)
        if preference.shape[0] == 1 and batch > 1:
            preference = preference.expand(batch, -1)
        return preference.to(device=device, dtype=dtype)

    def _pool_hidden_states(self, hidden_states):
        pooled = hidden_states.mean(dim=1) if self._force_policy_grad else hidden_states.detach().mean(dim=1)
        pooled = pooled.to(dtype=next(self.parameters()).dtype)
        return pooled

    def _store_weights(self, weights):
        self._last_routing_weights = weights if (self._force_policy_grad or torch.is_grad_enabled()) else weights.detach()
        return weights

    def _manual_weights_or_none(self, hidden_states):
        if self.manual_weights is None:
            return None

        batch = hidden_states.shape[0]
        weights = torch.as_tensor(
            self.manual_weights,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if weights.dim() != 2:
            raise ValueError("manual_weights must be 1D or 2D.")
        if weights.shape[-1] != self.num_lora_experts:
            raise ValueError(
                f"manual_weights size mismatch: got {weights.shape[-1]}, expected {self.num_lora_experts}"
            )
        if weights.shape[0] == 1 and batch > 1:
            weights = weights.expand(batch, -1)
        elif weights.shape[0] != batch:
            raise ValueError(f"manual_weights batch mismatch: got {weights.shape[0]}, expected 1 or {batch}")

        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return self._store_weights(weights)

class SimplifiedPreferenceGatingNetwork(BasePreferenceGatingNetwork):
    def __init__(self, hidden_dim, num_lora_experts, num_rewards=2):
        super().__init__(hidden_dim, num_lora_experts, num_rewards)
        self.router = nn.Linear(num_rewards, num_lora_experts)
        with torch.no_grad():
            # Break symmetry so the router can explore different experts early.
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
            self.router.bias.zero_()


    def forward(self, hidden_states, preference=None):
        manual_weights = self._manual_weights_or_none(hidden_states)
        if manual_weights is not None:
            return manual_weights
        preference = self._prepare_preference(hidden_states, preference)
        router_input = preference - (1.0 / self.num_rewards)
        logits = 2.0 * self.router(router_input)
        weights = F.softmax(logits, dim=-1)
        return self._store_weights(weights)
    
class LinearPreferenceGatingNetwork(BasePreferenceGatingNetwork):
    def __init__(self, hidden_dim, num_lora_experts, num_rewards=2):
        super().__init__(hidden_dim, num_lora_experts, num_rewards)
        self.router = nn.Linear(hidden_dim + num_rewards, num_lora_experts)

    def forward(self, hidden_states, preference=None):
        manual_weights = self._manual_weights_or_none(hidden_states)
        if manual_weights is not None:
            return manual_weights
        preference = self._prepare_preference(hidden_states, preference)
        pooled = self._pool_hidden_states(hidden_states)
        router_input = torch.cat([pooled, preference], dim=-1)
        weights = F.softmax(self.router(router_input), dim=-1)
        return self._store_weights(weights)


class QKAttentionPreferenceGatingNetwork(BasePreferenceGatingNetwork):
    def __init__(self, hidden_dim, num_lora_experts, num_rewards=2):
        super().__init__(hidden_dim, num_lora_experts, num_rewards)
        self.query_proj = nn.Linear(hidden_dim + num_rewards, hidden_dim)
        self.expert_keys = nn.Parameter(torch.randn(num_lora_experts, hidden_dim) * (hidden_dim ** -0.5))

    def forward(self, hidden_states, preference=None):
        manual_weights = self._manual_weights_or_none(hidden_states)
        if manual_weights is not None:
            return manual_weights
        preference = self._prepare_preference(hidden_states, preference)
        pooled = self._pool_hidden_states(hidden_states)
        query = self.query_proj(torch.cat([pooled, preference], dim=-1))
        scores = torch.matmul(query, self.expert_keys.t()) / (self.hidden_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return self._store_weights(weights)


class FiLMPreferenceGatingNetwork(BasePreferenceGatingNetwork):
    def __init__(self, hidden_dim, num_lora_experts, num_rewards=2):
        super().__init__(hidden_dim, num_lora_experts, num_rewards)
        self.film = nn.Linear(num_rewards, hidden_dim * 2)
        self.router = nn.Linear(hidden_dim, num_lora_experts)

    def forward(self, hidden_states, preference=None):
        manual_weights = self._manual_weights_or_none(hidden_states)
        if manual_weights is not None:
            return manual_weights
        preference = self._prepare_preference(hidden_states, preference)
        pooled = self._pool_hidden_states(hidden_states)
        gamma, beta = self.film(preference).chunk(2, dim=-1)
        modulated = pooled * (1.0 + gamma) + beta
        weights = F.softmax(self.router(modulated), dim=-1)
        return self._store_weights(weights)


def build_preference_gating_network(gating_type, hidden_dim, num_lora_experts, num_rewards=2):
    gating_type = gating_type.lower()
    gating_builders = {
        "simplified": SimplifiedPreferenceGatingNetwork,
        "linear": LinearPreferenceGatingNetwork,
        "qk_attention": QKAttentionPreferenceGatingNetwork,
        "film": FiLMPreferenceGatingNetwork,
    }
    if gating_type not in gating_builders:
        raise ValueError(f"Unsupported gating_type: {gating_type}")
    return gating_builders[gating_type](hidden_dim, num_lora_experts, num_rewards=num_rewards)


class ParameterCombinedLoRAProjection(nn.Module):
    def __init__(self, lora_a_weights, lora_b_weights):
        super().__init__()
        self.lora_a = nn.Parameter(lora_a_weights, requires_grad=False)
        self.lora_b = nn.Parameter(lora_b_weights, requires_grad=False)

    def forward(self, hidden_states, expert_weights):
        batch = hidden_states.shape[0]
        expert_weights = expert_weights.to(dtype=hidden_states.dtype)
        if expert_weights.dim() == 1:
            expert_weights = expert_weights.unsqueeze(0)
        if expert_weights.shape[0] == 1 and batch > 1:
            expert_weights = expert_weights.expand(batch, -1)
        mixed_a = torch.einsum("be,eri->bri", expert_weights, self.lora_a)
        mixed_b = torch.einsum("be,eor->bor", expert_weights, self.lora_b)
        low_rank = torch.einsum("bsi,bri->bsr", hidden_states, mixed_a)
        return torch.einsum("bsr,bor->bso", low_rank, mixed_b)


class ParameterCombinedFFN(nn.Module):
    def __init__(self, base_mlp, gate_proj_lora, up_proj_lora, down_proj_lora):
        super().__init__()
        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.down_proj = base_mlp.down_proj
        self.gate_proj_lora = gate_proj_lora
        self.up_proj_lora = up_proj_lora
        self.down_proj_lora = down_proj_lora
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, expert_weights):
        gate = self.act_fn(self.gate_proj(hidden_states) + self.gate_proj_lora(hidden_states, expert_weights))
        up = self.up_proj(hidden_states) + self.up_proj_lora(hidden_states, expert_weights)
        hidden = gate * up
        return self.down_proj(hidden) + self.down_proj_lora(hidden, expert_weights)


class MoEFFNLayer(nn.Module):
    def __init__(self, parameterized_ffn, model_ref=None):
        super().__init__()
        self.parameterized_ffn = parameterized_ffn
        object.__setattr__(self, "_model_weakref", weakref.ref(model_ref) if model_ref is not None else None)

    def set_model_ref(self, model_ref):
        object.__setattr__(self, "_model_weakref", weakref.ref(model_ref))

    def forward(self, hidden_states, **kwargs):
        model_ref = object.__getattribute__(self, "_model_weakref")
        if model_ref is None:
            raise RuntimeError("MoEFFNLayer is missing a model reference for shared gating.")

        model = model_ref()
        if model is None:
            raise RuntimeError("MoEFFNLayer model reference is no longer valid.")

        weights = getattr(model, "_current_routing_weights", None)
        if weights is None:
            raise RuntimeError("Shared routing weights must be prepared before FFN execution.")
        return self.parameterized_ffn(hidden_states, weights)


class MoEGatingTrainer:
    def __init__(
        self,
        moe_model,
        reward_models,
        instructions,
        learning_rate=1e-5,
        num_rewards=2,
        num_pref_samples=10,
        random_exploration_steps=100,
        ppo_clip_range=0.2,
    ):
        self.model = moe_model
        self.reward_models = reward_models
        self.instructions = instructions
        self.num_rewards = num_rewards
        self.num_pref_samples = num_pref_samples
        self.random_exploration_steps = random_exploration_steps
        self.ppo_clip_range = ppo_clip_range
        self.train_step_count = 0
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.5
        self.preference_baselines = {}

        for param in self.model.parameters():
            param.requires_grad = False

        gate_params = list(self.core_model.shared_gate.parameters())
        for param in gate_params:
            param.requires_grad = True

        self.optimizer = torch.optim.AdamW(gate_params, lr=learning_rate)
        print(f"Trainable gating parameters: {sum(p.numel() for p in gate_params):,}")

    @property
    def core_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def sample_preferences(self):
        return [np.random.dirichlet(np.ones(self.num_rewards)).tolist() for _ in range(self.num_pref_samples)]

    def _sample_random_lora_weights(self, batch_size, device):
        concentration = torch.ones(batch_size, self.core_model.shared_gate.num_lora_experts, device=device)
        return torch.distributions.Dirichlet(concentration).sample()

    def _preference_key(self, preference):
        if isinstance(preference, torch.Tensor):
            preference = preference.detach().cpu().tolist()
        return tuple(round(float(value), 2) for value in preference)

    def set_model_preference(self, preference):
        if isinstance(preference, np.ndarray):
            preference = preference.tolist()
        self.core_model.set_preference(preference)

    def _pad_inputs(self, batch, tokenizer):
        items = [ids.clone().detach() if isinstance(ids, torch.Tensor) else torch.tensor(ids) for ids in batch["input_ids"]]
        max_length = max(len(ids) for ids in items)
        padded_ids, masks = [], []
        for ids in items:
            pad = max_length - len(ids)
            padded_ids.append(torch.cat([torch.full((pad,), tokenizer.pad_token_id, dtype=ids.dtype), ids]))
            masks.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))
        device = next(self.model.parameters()).device
        return torch.stack(padded_ids).to(device), torch.stack(masks).to(device)

    def _score_generations(self, generated_sequences, prompts, preference, tokenizer):
        responses = tokenizer.batch_decode(generated_sequences)
        prompt_text = tokenizer.batch_decode(prompts)
        prompt_text, responses = get_clean_data(responses, prompt_text)
        query_response_pairs = [
            (self.instructions.get_input(text), self.instructions.get_response(text))
            for text in responses
        ]
        if hasattr(self.instructions, "get_post"):
            rewards_list = self.reward_models.get_reward_model_scores(
                query_response_pairs,
                self.instructions.get_post,
                normalize_rewards=False,
                round_digits=None,
            )
        else:
            rewards_list = self.reward_models.get_reward_model_scores(
                query_response_pairs,
                normalize_rewards=False,
                round_digits=None,
            )
        return [
            float(sum(preference[k] * rewards_list[k][idx] for k in range(self.num_rewards)))
            for idx in range(len(query_response_pairs))
        ]

    def _routing_log_prob(self, action_weights, policy_weights):
        action_weights = action_weights.to(device=policy_weights.device, dtype=policy_weights.dtype)
        return (action_weights.detach() * torch.log(policy_weights + 1e-8)).sum(dim=-1)

    def _routing_policy_terms(self, input_ids, attention_mask, preference):
        self.core_model.shared_gate._force_policy_grad = True

        self.set_model_preference(preference)
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            weights = self.core_model.shared_gate._last_routing_weights
            return weights
        finally:
            self.core_model.shared_gate._force_policy_grad = False

    def train_step_reinforce(self, batch, tokenizer, grad_scale=1.0, step_optimizer=True, **kwargs):
        input_ids, attention_mask = self._pad_inputs(batch, tokenizer)
        all_rewards = []
        all_lora_weights = []
        total_loss = 0.0
        total_policy_loss = 0.0
        preferences = self.sample_preferences()
        use_random_exploration = self.train_step_count < self.random_exploration_steps

        self.model.eval()
        pref_data = []
        for preference in preferences:
            self.set_model_preference(preference)
            exploration_weights = None
            try:
                if use_random_exploration:
                    exploration_weights = self._sample_random_lora_weights(input_ids.shape[0], input_ids.device)
                    self.core_model.set_manual_lora_weights(exploration_weights)
                with torch.no_grad():
                    generated = self.core_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    all_lora_weights.append(self.core_model.shared_gate._last_routing_weights.detach().float().cpu())
            finally:
                self.core_model.set_manual_lora_weights(None)
            rewards = self._score_generations(generated, input_ids, preference, tokenizer)
            all_rewards.extend(rewards)
            old_weights = all_lora_weights[-1].to(dtype=torch.float32)
            stored_exploration_weights = (
                exploration_weights.detach().float().cpu() if exploration_weights is not None else None
            )
            pref_data.append((preference, input_ids, attention_mask, rewards, old_weights, stored_exploration_weights))

        self.model.train()
        for preference, prompt_ids, prompt_mask, rewards, old_weights, stored_exploration_weights in pref_data:
            pref_key = self._preference_key(preference)
            pref_baseline = self.preference_baselines.get(pref_key, self.reward_baseline)
            advantages = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device) - pref_baseline
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            current_weights = self._routing_policy_terms(prompt_ids, prompt_mask, preference)
            action_weights = old_weights.to(device=prompt_ids.device, dtype=current_weights.dtype)
            old_log_prob = self._routing_log_prob(action_weights, action_weights).detach()
            current_log_prob = self._routing_log_prob(action_weights, current_weights)
            ratio = torch.exp(current_log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip_range, 1.0 + self.ppo_clip_range)
            surrogate_1 = ratio * advantages.detach()
            surrogate_2 = clipped_ratio * advantages.detach()
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

            gate_params_summary = {
                name: param.detach().float().cpu().tolist()
                for name, param in self.core_model.shared_gate.named_parameters()
            }
            current_lora_weights = current_weights.detach().float().cpu().tolist()
            rollout_lora_weights = action_weights.detach().float().cpu().tolist()
            print(
                f"pref={preference} "
                f"gate_params={gate_params_summary} "
                f"exploration_weights={stored_exploration_weights[0] if stored_exploration_weights is not None else None} "
                f"rollout_weights={rollout_lora_weights[0]} "
                f"lora_weights={current_lora_weights[0]} "
                f"reward_mean={float(np.mean(rewards)):.6f} "
                f"ratio_mean={ratio.detach().float().mean().item():.6f} "
                f"policy_loss={policy_loss.detach().float().item():.6f}"
            )

            if not isinstance(policy_loss, torch.Tensor) or not policy_loss.requires_grad:
                raise RuntimeError(
                    "policy_loss has no gradient path for the shared gating network."
                )
            (policy_loss / max(1.0, float(grad_scale))).backward()


            total_policy_loss += policy_loss / max(1, len(pref_data))
            total_loss += policy_loss / max(1, len(pref_data))
            reward_mean = float(np.mean(rewards))
            self.preference_baselines[pref_key] = (
                self.baseline_momentum * pref_baseline +
                (1 - self.baseline_momentum) * reward_mean
            )

        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        mean_lora_weights = (
            torch.cat(all_lora_weights, dim=0).mean(dim=0).tolist()
            if all_lora_weights else []
        )
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline +
            (1 - self.baseline_momentum) * rewards_tensor.mean().item()
        )

        if step_optimizer:
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.train_step_count += 1

        return {
            "policy_loss": total_policy_loss.item() if isinstance(total_policy_loss, torch.Tensor) else float(total_policy_loss),
            "hv_policy_loss": 0.0,
            "balance_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss),
            "mean_reward": rewards_tensor.mean().item(),
            "mean_lora_weights": mean_lora_weights,
            "std_reward": rewards_tensor.std(unbiased=False).item(),
            "baseline": self.reward_baseline,
            "random_exploration": float(use_random_exploration),
        }
