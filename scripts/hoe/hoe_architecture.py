"""HoE (Hierarchy of Experts) core architecture.

DynamicWeights: shared state that broadcasts the current preference vector to all router layers.
CustomLinear variants: router implementations of increasing sophistication.
ConditionedMOEModel: wraps a MoLA PeftModel, replacing every router Linear with a
    preference-conditioned layer.
ConditionedMOEModelWithValueHead: adds per-reward value heads for PPO training.
WeightSampler: generates per-batch preference weights from a bimodal distribution.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeights:
    """Shared preference state broadcast to all router layers."""

    def __init__(self, num_rewards: int = 2):
        self.num_rewards = num_rewards
        self.weights = None
        self.labels = None

    def set_dynamic_weights(self, weights: torch.Tensor, batch_size: int = 1):
        assert weights.shape == torch.Size([batch_size, self.num_rewards])
        self.weights = weights

    def set_dynamic_labels(self, labels: torch.Tensor, batch_size: int = 1):
        self.labels = labels


class CustomLinearv0(nn.Linear):
    """Bias-only router: learns a constant mixture, ignores hidden state and preferences."""

    def __init__(self, in_features: int, out_features: int, dynamic_weights: DynamicWeights):
        super().__init__(in_features, out_features, bias=True)
        nn.init.constant_(self.bias, 1.0)
        nn.init.constant_(self.weight, 0.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = super().forward(input.to(self.weight.dtype))
        return F.softmax(logits.float(), dim=-1).to(input.dtype)


class CustomLinearv1(nn.Linear):
    """Preference-conditioned router: concatenates preference vector to hidden state.

    Initialized so the identity mapping (weight[i, -num_rewards+i] = 1) routes
    sample i to expert i when preferences are one-hot.
    """

    def __init__(self, in_features: int, out_features: int, dynamic_weights: DynamicWeights):
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features
        self.dynamic_weights = dynamic_weights
        super().__init__(in_features + num_rewards, out_features, bias=False)
        self._init_weights()

    def _init_weights(self):
        num_rewards = self.out_features
        init_w = torch.zeros((self.out_features, self.in_features))
        for idx in range(num_rewards):
            init_w[idx, -num_rewards + idx] = 1.0
        self.weight.data = init_w

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weights = self.dynamic_weights.weights
        assert weights is not None
        assert input.shape[0] % weights.shape[0] == 0
        seq_ratio = input.shape[0] // weights.shape[0]
        extra = weights.unsqueeze(1).repeat(1, seq_ratio, 1).view(-1, weights.shape[1]).to(input.device, input.dtype)
        new_input = torch.cat([input, extra], dim=-1)
        logits = super().forward(new_input.to(self.weight.dtype))
        return F.softmax(logits.float(), dim=-1).to(input.dtype)


class CustomLinearv2(nn.Linear):
    """Preference-conditioned router with additive bias and element-wise modulation."""

    def __init__(self, in_features: int, out_features: int, dynamic_weights: DynamicWeights):
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features
        self.dynamic_weights = dynamic_weights
        super().__init__(in_features + num_rewards, out_features, bias=True)
        self._init_weights()

    def _init_weights(self):
        self.weight.data = torch.zeros((self.out_features, self.in_features))
        self.bias.data = torch.ones_like(self.bias.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weights = self.dynamic_weights.weights
        assert weights is not None
        assert input.shape[0] % weights.shape[0] == 0
        seq_ratio = input.shape[0] // weights.shape[0]
        extra = weights.unsqueeze(1).repeat(1, seq_ratio, 1).view(-1, weights.shape[1]).to(input.device, input.dtype)
        new_input = torch.cat([input, extra], dim=-1)
        logits = super().forward(new_input.to(self.weight.dtype))
        output = F.softmax(logits.float(), dim=-1).to(input.dtype)
        return torch.einsum('ln,ln->ln', output, extra)


class CustomLinearv3(nn.Module):
    """Dual-path preference-conditioned router with learned weighting.

    Path 1 (scale): outputs a per-expert scale modulated by the preference vector.
    Path 2 (offset): outputs a preference-free bias correction.
    Final: scale * preference + offset.
    This is the default router used in HoE training.
    """

    def __init__(self, in_features: int, out_features: int, dynamic_weights: DynamicWeights,
                 hidden_dim: int = 32):
        num_rewards = dynamic_weights.num_rewards
        assert num_rewards == out_features
        self.dynamic_weights = dynamic_weights
        super().__init__()
        self.linear1 = nn.Linear(in_features + num_rewards, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, out_features, bias=False)
        self.linear3 = nn.Linear(in_features + num_rewards, hidden_dim, bias=True)
        self.linear4 = nn.Linear(hidden_dim, out_features, bias=False)
        self._init_weights()

    def _init_weights(self, scale: float = 0.001):
        for layer in [self.linear1, self.linear3]:
            layer.weight.data.normal_(mean=0.0, std=scale)
            layer.bias.data.zero_()
        for layer in [self.linear2, self.linear4]:
            layer.weight.data.normal_(mean=0.0, std=scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weights = self.dynamic_weights.weights
        assert weights is not None
        assert input.shape[0] % weights.shape[0] == 0
        seq_ratio = input.shape[0] // weights.shape[0]
        extra = weights.unsqueeze(1).repeat(1, seq_ratio, 1).view(-1, weights.shape[1]).to(input.device, input.dtype)
        new_input = torch.cat([input, extra], dim=-1).to(self.linear1.weight.dtype)

        x0 = torch.relu(self.linear1(new_input))
        x0 = self.linear2(x0)
        x0 = x0 + torch.ones_like(x0.detach())

        x1 = torch.relu(self.linear3(new_input))
        x1 = self.linear4(x1)

        return torch.einsum('ln,ln->ln', x0, extra) + x1


ROUTER_TYPES = {
    'v0': CustomLinearv0,
    'v1': CustomLinearv1,
    'v2': CustomLinearv2,
    'v3': CustomLinearv3,
}


class ValueHead(nn.Module):
    """Per-reward scalar value head for PPO."""

    def __init__(self, config, summary_dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(config, 'word_embed_proj_dim', None)
        if getattr(config, 'is_encoder_decoder', False) and hasattr(config, 'decoder'):
            hidden_size = config.decoder.hidden_size
        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class ConditionedMOEModel:
    """Installs preference-conditioned routers into a MoLA model.

    Call ConditionedMOEModel.init(model) to replace every router Linear layer
    with a preference-conditioned variant and attach a DynamicWeights handle
    as model.dynamic_weights.
    """

    @classmethod
    def init(cls, model, router_type: str = 'v1', num_rewards: int = 2, hidden_dim: int = 32):
        """Replace router layers in-place and return the shared DynamicWeights."""
        dynamic_weights = DynamicWeights(num_rewards=num_rewards)
        RouterClass = ROUTER_TYPES[router_type]
        cls._replace_routers(model, dynamic_weights, RouterClass, hidden_dim)
        model.dynamic_weights = dynamic_weights
        return dynamic_weights

    @classmethod
    def _replace_routers(cls, model, dynamic_weights, RouterClass, hidden_dim):
        for name, module in model.named_children():
            if 'router' in name and isinstance(module, nn.Linear):
                in_f, out_f = module.in_features, module.out_features
                if RouterClass is CustomLinearv3:
                    new_layer = RouterClass(in_f, out_f, dynamic_weights, hidden_dim=hidden_dim)
                else:
                    new_layer = RouterClass(in_f, out_f, dynamic_weights)
                setattr(model, name, new_layer.to(module.weight.device).to(module.weight.dtype))
            else:
                cls._replace_routers(module, dynamic_weights, RouterClass, hidden_dim)


class ConditionedMOEModelWithValueHead(nn.Module):
    """HoE model wrapper: base MoLA model + per-reward value heads.

    Only router parameters and value heads are trained; expert LoRA weights are frozen.
    Value function is a preference-weighted sum across per-reward value heads:
        V(s) = sum_k w_k * V_k(s)
    """

    def __init__(self, model, num_rewards: int, v_heads_init=None, trainable_module=None):
        super().__init__()
        self.model = model
        self.num_rewards = num_rewards
        # Match original: default trains only router layers; pass "all" to train everything.
        self.trainable_module = ['router'] if trainable_module is None else trainable_module
        self.dynamic_weights = model.dynamic_weights

        if v_heads_init is None:
            self.v_heads = nn.ModuleList(
                [ValueHead(model.config) for _ in range(num_rewards)]
            )
            self._init_value_heads()
        else:
            self.v_heads = nn.ModuleList(
                [torch.load(p, map_location=model.device) for p in v_heads_init]
            )

    def _init_value_heads(self, initializer_range: float = 0.2):
        for v_head in self.v_heads:
            v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            v_head.summary.bias.data.zero_()

    def forward(self, weights: torch.Tensor, input_ids=None, past_key_values=None,
                attention_mask=None, **kwargs):
        kwargs['output_hidden_states'] = True
        kwargs['past_key_values'] = past_key_values
        self.dynamic_weights.set_dynamic_weights(weights=weights, batch_size=input_ids.shape[0])

        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = model_output.hidden_states[-1]
        lm_logits = model_output.logits

        if last_hidden.device != self.v_heads[0].summary.weight.device:
            last_hidden = last_hidden.to(self.v_heads[0].summary.weight.device)

        value_per_head = [self.v_heads[i](last_hidden).squeeze(-1) for i in range(self.num_rewards)]
        # preference-weighted value: V(s) = sum_k w_k * V_k(s)
        value = torch.einsum('bn,nbq->bq', weights.to(torch.float32), torch.stack(value_per_head))

        return {'lm_logits': lm_logits, 'value': value}

    def generate(self, weights: torch.Tensor, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        self.dynamic_weights.set_dynamic_weights(weights=weights, batch_size=batch_size)
        return self.model.generate(*args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        for v_head in self.v_heads:
            v_head.train(mode)
        if mode:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                # Iterate over trainable_module list or string (original iterates characters of "all")
                for module_key in self.trainable_module:
                    if module_key in name:
                        param.requires_grad = True
        else:
            for _, param in self.model.named_parameters():
                param.requires_grad = False

    def eval(self):
        self.model.eval()

    def test_grad(self):
        for target in ['router', 'v_head']:
            for name, param in self.named_parameters():
                if target in name:
                    print(f'{target} | requires_grad={param.requires_grad} | grad={param.grad}')
                    break

    def save_pretrained(self, save_directory: str):
        import os
        self.model.save_pretrained(save_directory)
        for idx, v_head in enumerate(self.v_heads):
            torch.save(v_head, os.path.join(save_directory, f'v_head{idx}.pt'))

    def can_generate(self):
        return True


class WeightSampler:
    """Generates per-batch preference weight tensors.

    Sampling strategy: bimodal distribution with peaks at 0 and 1, mixed with uniform.
    The uniform_ratio controls the fraction of uniform samples — starts high (1.0) for
    broad exploration, then decreases during training to sharpen at extremes.
    """

    def __init__(self, reward_dim: int = 2, uniform_ratio: float = 0.0):
        self.reward_dim = reward_dim
        self.uniform_ratio = uniform_ratio
        self.step = 0

    def _sample_corner(self, n: int) -> np.ndarray:
        """Sample from vertices of the probability simplex (one-hot vectors)."""
        indices = np.random.randint(0, self.reward_dim, size=n)
        corners = np.zeros((n, self.reward_dim))
        corners[np.arange(n), indices] = 1.0
        return corners

    def _sample_uniform_simplex(self, n: int) -> np.ndarray:
        """Sample uniformly from the probability simplex via Dirichlet(1,...,1)."""
        return np.random.dirichlet(np.ones(self.reward_dim), size=n)

    def generate_weights(self, n_samples: int):
        is_uniform = np.random.rand(n_samples) < self.uniform_ratio
        uniform_samples = self._sample_uniform_simplex(n_samples)
        corner_samples = self._sample_corner(n_samples)
        weights = np.where(is_uniform[:, None], uniform_samples, corner_samples)
        weights = torch.tensor(weights, dtype=torch.float32)
        return [w for w in weights]

    def update(self):
        self.step += 1
        if self.step == 50:
            self.uniform_ratio = 0.3
        if self.step == 100:
            self.uniform_ratio = 0.5
        if self.step == 150:
            self.uniform_ratio = 0.5
