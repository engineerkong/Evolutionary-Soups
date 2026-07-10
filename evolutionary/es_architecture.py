"""es_architecture.py — GatingNetwork + MoEForCausalLM for ES.

Standard ES design: each population individual is a plain GatingNetwork
(no preference conditioning).  At inference, select the individual whose
reward vector best matches the desired preference:
    best_i = argmax_i  dot(λ, fitness_i)
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache


# ---------------------------------------------------------------------------
# Expert loading — one shared trunk, per-expert MLPs
# ---------------------------------------------------------------------------
#
# The SFT/PPO checkpoints are LoRA adapters with
# ``target_modules = [gate_proj, up_proj, down_proj]``, so after
# ``merge_and_unload()`` two experts differ ONLY in their per-layer MLP weights;
# attention, embeddings, lm_head and every LayerNorm are bit-identical copies of
# the base model (verified: 2.41 B of 6.74 B params, 35.8%, are shared).
#
# Both MoE classes here already respect that: everything outside the FFN reads
# ``self.experts[0]`` and only ``experts[i].model.layers[l].mlp`` is per-expert.
# Loading E full models therefore wastes (E-1) × 4.82 GB of bf16 weights
# (9.6 GB at E=3) plus (E-1) full disk reads.
#
# ``load_shared_experts`` returns expert 0 as a real model (the trunk) and every
# other expert as a *view*: the same Python objects for every submodule except
# ``layers[l].mlp``.  Numerics are unchanged — the merge still runs on the same
# device in the same dtype, so logits are bit-for-bit identical to the old path.

def _shell(module: nn.Module) -> nn.Module:
    """Shallow copy of ``module`` with its OWN child dicts.

    ``copy.copy`` shares ``_modules``/``_parameters`` with the original, so
    swapping a child would mutate the source too.  Re-bind those dicts (their
    *values* stay shared, which is the whole point).
    """
    new = copy.copy(module)
    new._parameters = OrderedDict(module._parameters)
    new._buffers    = OrderedDict(module._buffers)
    new._modules    = OrderedDict(module._modules)
    return new


def _expert_view(trunk, mlps):
    """A CausalLM-shaped view of ``trunk`` whose per-layer MLPs are ``mlps``."""
    layers = nn.ModuleList()
    for layer, mlp in zip(trunk.model.layers, mlps):
        lv = _shell(layer)
        lv.mlp = mlp                     # nn.Module.__setattr__ → lv._modules
        layers.append(lv)
    inner = _shell(trunk.model)
    inner.layers = layers
    view = _shell(trunk)
    view.model = inner
    return view


def load_shared_experts(base_model_name, adapter_paths, device,
                        tokenizer=None, dtype=torch.bfloat16,
                        merge_device=None, verbose=True):
    """Load LoRA experts that share every weight except the per-layer MLPs.

    Returns a list of ``len(adapter_paths)`` CausalLM-shaped modules, all in
    ``eval()`` with ``requires_grad=False``.  ``experts[0]`` is a real model;
    the rest are views onto it.

    ``merge_device`` defaults to ``device``.  Keep it there: peft upcasts the
    LoRA product to fp32 when merging on CPU but not on CUDA, so a CPU merge is
    *not* bit-identical to the old GPU path.  Transient peak is
    ``trunk + one full model + already-kept MLPs``, which is still below the old
    steady state of ``E × full model`` for every E ≥ 2.
    """
    merge_device = merge_device or device

    def _merged(path):
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=dtype, device_map=merge_device)
        if tokenizer is not None:
            # Callers disagreed on resize-before vs resize-after-merge.  It is moot:
            # the adapters carry no embed/lm_head keys, so merge never touches them,
            # and for these checkpoints len(tokenizer) == config.vocab_size makes the
            # resize a no-op.  RNG consumption is identical to the old path either way
            # (both do one from_pretrained + one PeftModel.from_pretrained per expert).
            base.resize_token_embeddings(len(tokenizer))
        from peft import PeftModel                      # local: keep import cheap
        return PeftModel.from_pretrained(base, path).merge_and_unload()

    if verbose:
        print(f'Loading {len(adapter_paths)} expert models (shared trunk) …', flush=True)
        print(f'  Expert 1: {adapter_paths[0]}', flush=True)
    trunk = _merged(adapter_paths[0]).to(device)
    experts = [trunk]

    for i, path in enumerate(adapter_paths[1:], start=2):
        if verbose:
            print(f'  Expert {i}: {path}  (MLPs only)', flush=True)
        m = _merged(path)
        mlps = [m.model.layers[l].mlp.to(device) for l in range(len(m.model.layers))]
        experts.append(_expert_view(trunk, mlps))
        del m                            # frees attn / embed / lm_head / norms
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for e in experts:
        e.eval()
        for p in e.parameters():
            p.requires_grad = False
    if verbose:
        shared = sum(p.numel() for p in trunk.parameters()) - sum(
            p.numel() for l in trunk.model.layers for p in l.mlp.parameters())
        print(f'  All {len(experts)} experts on {device}; '
              f'{shared / 1e9:.2f} B params shared across them.', flush=True)
    return experts


# ---------------------------------------------------------------------------
# α-entmax helpers (Peters et al., 2019)
# ---------------------------------------------------------------------------

def _sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Exact sparsemax (α=2 entmax) — closed-form projection onto the simplex."""
    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    z_cumsum = torch.cumsum(z_sorted, dim=-1)
    mask = (1.0 + k * z_sorted) > z_cumsum
    k_z = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(-1, k_z - 1) - 1.0) / k_z.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def _entmax_bisect(z: torch.Tensor, alpha: float, n_iter: int = 10) -> torch.Tensor:
    """α-entmax via bisection for arbitrary α > 1 (Peters et al., 2019).

    p_i* = max(0, (α-1)·(z_i - τ))^(1/(α-1)),  τ s.t. Σp=1.

    """
    am1 = alpha - 1.0
    z = z - z.max(dim=-1, keepdim=True).values   # shift for numerical stability
    tau_hi = z.max(dim=-1, keepdim=True).values   # = 0 after shift
    tau_lo = torch.full_like(tau_hi, -1.0 / am1)  # guaranteed lower bound
    for _ in range(n_iter):
        tau_mid = (tau_hi + tau_lo) * 0.5
        p = torch.clamp(am1 * (z - tau_mid), min=0.0).pow(1.0 / am1)
        err = p.sum(dim=-1, keepdim=True) - 1.0
        tau_lo = torch.where(err >= 0, tau_mid, tau_lo)  # sum>=1 → tau too low  → raise lo
        tau_hi = torch.where(err < 0,  tau_mid, tau_hi)  # sum<1  → tau too high → lower hi
    tau = (tau_hi + tau_lo) * 0.5
    p = torch.clamp(am1 * (z - tau), min=0.0).pow(1.0 / am1)
    s = p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return p / s


def _apply_entmax(z: torch.Tensor, alpha) -> torch.Tensor:
    """Dispatch α-entmax. alpha may be a pre-extracted Python float or a scalar Tensor."""
    a = alpha if isinstance(alpha, float) else float(alpha.clamp(1.0, 2.0))
    a = max(1.0, min(2.0, a))
    if a < 1.0 + 1e-4:
        return F.softmax(z, dim=-1)
    if abs(a - 2.0) < 1e-4:
        return _sparsemax(z)
    return _entmax_bisect(z, a)


# ---------------------------------------------------------------------------
# Gating Network
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Per-layer gating: last-token hidden state → expert merging weights.

    Input:  (B, seq, lm_hidden_size)  or  (B, lm_hidden_size)
            A (B, seq, H) input is reduced to its last position (B, H); the
            tokenizer pads left, so the last position is always a real token.
    Output: (B, num_experts)  — sparse weights via α-entmax (α=1→softmax, α=2→sparsemax)

    alpha is a per-layer nn.Parameter (shape [num_layers]) included in the
    evolutionary genome via net_to_params / params_to_net.
    """

    def __init__(self, lm_hidden_size: int = 4096, num_experts: int = 2,
                 hidden_size: int = 256, num_layers: int = 32,
                 alpha_init: float = 1.0, fixed_alpha: float = None):
        super().__init__()
        self.lm_hidden_size = lm_hidden_size
        self.num_experts    = num_experts
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.fixed_alpha    = fixed_alpha  # None = learnable, float = fixed (not in genome)
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        ).to(torch.bfloat16)
        if fixed_alpha is None:
            # Per-layer α registered as Parameter so it's part of the evolutionary genome
            self.alpha = nn.Parameter(
                torch.ones(num_layers, dtype=torch.float32) * alpha_init)

    def alpha_floats(self) -> list:
        """Return all per-layer alpha values as Python floats (single GPU→CPU transfer)."""
        if self.fixed_alpha is not None:
            return [max(1.0, min(2.0, self.fixed_alpha))] * self.num_layers
        return self.alpha.detach().clamp(1.0, 2.0).tolist()

    def forward(self, hidden_states: torch.Tensor,
                layer_idx: int = 0,
                alpha_val: float = None) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]   # last token (tokenizer pads left)
        logits = self.net(hidden_states.to(self.net[0].weight.dtype))
        if alpha_val is None:
            alpha_val = (self.fixed_alpha if self.fixed_alpha is not None
                         else float(self.alpha[layer_idx].clamp(1.0, 2.0)))
        return _apply_entmax(logits.float(), alpha_val).to(logits.dtype)


# ---------------------------------------------------------------------------
# MoE Language Model  (identical to moead_architecture.MoEForCausalLM)
# ---------------------------------------------------------------------------

def _sample_token(logits: torch.Tensor, temperature: float,
                  top_p: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        top_vals, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < top_vals[:, -1:]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float('-inf')
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class MoEForCausalLM(nn.Module):
    """Mixture-of-Experts LM: N frozen expert LLMs combined at each transformer layer.

    Gating is per-sequence (prefill coefficients reused for every decode step)
    on post-attention hidden states (before FFN).
    """

    def __init__(self, expert_models: list, gating_net):
        super().__init__()
        self.experts     = nn.ModuleList(expert_models)
        self.gating_net  = gating_net
        self.config      = expert_models[0].config
        self.num_experts = len(expert_models)
        self.num_layers  = len(expert_models[0].model.layers)

        for exp in self.experts:
            for p in exp.parameters():
                p.requires_grad = False

        self._expert_streams = [torch.cuda.Stream() for _ in range(self.num_experts)]
        self._ln_ready = torch.cuda.Event()

    def _embed(self, input_ids):
        return self.experts[0].model.embed_tokens(input_ids)

    @staticmethod
    def _build_causal_mask(attention_mask, hidden_states, past_len):
        B, seq_len, _ = hidden_states.shape
        total_len = past_len + seq_len
        dtype, device = hidden_states.dtype, hidden_states.device
        min_val = torch.finfo(dtype).min
        mask = torch.full((seq_len, total_len), min_val, dtype=dtype, device=device)
        col  = torch.arange(total_len, device=device).unsqueeze(0)
        row  = torch.arange(seq_len,   device=device).unsqueeze(1)
        mask[col <= (past_len + row)] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()
        if attention_mask is not None:
            am       = attention_mask[:, :total_len].to(dtype=dtype)
            pad_mask = (1.0 - am) * min_val
            mask     = mask + pad_mask.unsqueeze(1).unsqueeze(2)
        return mask

    @staticmethod
    def _pkv_seq_len(kv_cache):
        return 0 if kv_cache is None else kv_cache.get_seq_length()

    def _moe_layers(self, hidden_states, causal_mask, position_ids,
                    kv_cache, use_cache=True, seq_coefficients=None):
        if kv_cache is None:
            kv_cache = DynamicCache()
        B = hidden_states.shape[0]
        is_prefill = seq_coefficients is None
        if is_prefill:
            seq_coefficients = []
            # Single GPU→CPU transfer for all 32 alpha values instead of one per layer
            cached_alphas = self.gating_net.alpha_floats()

        for l in range(self.num_layers):
            layer0 = self.experts[0].model.layers[l]

            # ── Attention sub-layer (shared across experts) ──────────────────
            residual       = hidden_states
            hidden_ln      = layer0.input_layernorm(hidden_states)
            attn_out, _, _ = layer0.self_attn(
                hidden_states=hidden_ln, attention_mask=causal_mask,
                position_ids=position_ids, past_key_value=kv_cache,
                output_attentions=False, use_cache=use_cache,
            )
            hidden_states = residual + attn_out

            # ── Gate on post-attn hidden states (before FFN) ─────────────────
            if is_prefill:
                coeff = self.gating_net(hidden_states, layer_idx=l,
                                        alpha_val=cached_alphas[l])
                seq_coefficients.append(coeff)
            else:
                coeff = seq_coefficients[l]

            # ── FFN sub-layer (all experts in parallel) ───────────────────────
            ln_hidden = self.experts[0].model.layers[l].post_attention_layernorm(hidden_states)
            self._ln_ready.record()

            residual    = hidden_states
            ffn_outputs = [None] * self.num_experts
            for i, (expert, stream) in enumerate(zip(self.experts, self._expert_streams)):
                stream.wait_event(self._ln_ready)
                with torch.cuda.stream(stream):
                    ffn_outputs[i] = expert.model.layers[l].mlp(ln_hidden)
            cur = torch.cuda.current_stream()
            for stream in self._expert_streams:
                cur.wait_stream(stream)

            # ── Merge expert FFN outputs ──────────────────────────────────────
            coefficients  = coeff.to(hidden_states.dtype)
            hidden_states = residual + sum(
                coefficients[:, i].view(B, 1, 1) * ffn_outputs[i]
                for i in range(self.num_experts)
            )

        return hidden_states, kv_cache, seq_coefficients

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=128,
                 do_sample=False, temperature=1.0, top_p=0.9, top_k=0, **kwargs):
        device = input_ids.device
        B      = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated        = input_ids.clone()
        kv_cache         = DynamicCache()
        cur_attn         = attention_mask.clone()
        seq_coefficients = None

        for step in range(max_new_tokens):
            past_len = self._pkv_seq_len(kv_cache)
            cur_ids  = generated if step == 0 else generated[:, -1:]

            if step == 0:
                position_ids = cur_attn.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(cur_attn == 0, 1)
            else:
                position_ids = cur_attn.sum(dim=-1, keepdim=True).long() - 1

            hidden = self._embed(cur_ids)
            causal_mask = self._build_causal_mask(cur_attn, hidden, past_len)

            hidden, kv_cache, seq_coefficients = self._moe_layers(
                hidden, causal_mask, position_ids, kv_cache,
                use_cache=True, seq_coefficients=seq_coefficients)

            next_logits = self.experts[0].lm_head(
                self.experts[0].model.norm(hidden))[:, -1, :]

            if do_sample:
                next_token = _sample_token(next_logits, temperature, top_p, top_k)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            cur_attn  = torch.cat(
                [cur_attn, torch.ones(B, 1, dtype=cur_attn.dtype, device=device)], dim=1)

            eos = getattr(self.config, 'eos_token_id', None)
            if eos is not None and (next_token.squeeze(-1) == eos).all():
                break

        return generated


# ---------------------------------------------------------------------------
# Simple prompt-level gating  (no per-layer mixing)
# ---------------------------------------------------------------------------

class SimpleGatingNetwork(nn.Module):
    """Prompt-level gating: last-token hidden state of prompt → expert merging weights.

    Computed ONCE per prompt during prefill; weights are reused for all decode steps.
    Unlike GatingNetwork, there is no per-layer application — experts are merged at
    the final hidden state level (before lm_head).

    Input:  (B, lm_hidden_size)  — last non-padding token hidden state
    Output: (B, num_experts)     — softmax merging weights
    """

    def __init__(self, lm_hidden_size: int = 4096, num_experts: int = 2,
                 hidden_size: int = 256, fixed_alpha: float = 1.0):
        super().__init__()
        self.lm_hidden_size = lm_hidden_size
        self.num_experts    = num_experts
        self.hidden_size    = hidden_size
        self.fixed_alpha    = fixed_alpha
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        ).to(torch.bfloat16)

    def alpha_floats(self) -> list:
        return [max(1.0, min(2.0, self.fixed_alpha))] * 999

    def forward(self, last_token_hidden: torch.Tensor,
                layer_idx: int = 0, alpha_val: float = None) -> torch.Tensor:
        if last_token_hidden.dim() == 3:
            last_token_hidden = last_token_hidden[:, -1, :]   # take last token if seq given
        logits = self.net(last_token_hidden.to(self.net[0].weight.dtype))
        alpha  = self.fixed_alpha if self.fixed_alpha is not None else 1.0
        return _apply_entmax(logits.float(), alpha).to(logits.dtype)


class SimpleMoEForCausalLM(nn.Module):
    """MoE LM that merges expert outputs at the final hidden state level.

    Gating is computed ONCE from the prompt's last-token hidden state (using
    expert[0]'s transformer, since all experts share the same untrained embedding).
    At each decode step, all experts run independently and their final hidden
    states are linearly merged before the shared lm_head.
    """

    def __init__(self, expert_models: list, gating_net: SimpleGatingNetwork):
        super().__init__()
        self.experts     = nn.ModuleList(expert_models)
        self.gating_net  = gating_net
        self.config      = expert_models[0].config
        self.num_experts = len(expert_models)

        for exp in self.experts:
            for p in exp.parameters():
                p.requires_grad = False

        self._expert_streams = [torch.cuda.Stream() for _ in range(self.num_experts)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, input_ids):
        # Embedding is shared (untrained); use expert[0].
        return self.experts[0].model.embed_tokens(input_ids)

    @staticmethod
    def _last_token_hidden(hidden: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """Return (B, H) — hidden state of the last real token per sample.
        Assumes left-padding (tokenizer.padding_side='left'), so the last
        position is always a real token.
        """
        return hidden[:, -1, :]                            # (B, H)

    @staticmethod
    def _build_causal_mask(attention_mask, hidden_states, past_len):
        B, seq_len, _ = hidden_states.shape
        total_len     = past_len + seq_len
        dtype, device = hidden_states.dtype, hidden_states.device
        min_val       = torch.finfo(dtype).min
        mask = torch.full((seq_len, total_len), min_val, dtype=dtype, device=device)
        col  = torch.arange(total_len, device=device).unsqueeze(0)
        row  = torch.arange(seq_len,   device=device).unsqueeze(1)
        mask[col <= (past_len + row)] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()
        if attention_mask is not None:
            am       = attention_mask[:, :total_len].to(dtype=dtype)
            pad_mask = (1.0 - am) * min_val
            mask     = mask + pad_mask.unsqueeze(1).unsqueeze(2)
        return mask

    @staticmethod
    def _pkv_seq_len(kv_cache):
        return 0 if kv_cache is None else kv_cache.get_seq_length()

    def _run_expert(self, expert, hidden, causal_mask, position_ids, kv_cache):
        """Run one expert's transformer layers; return (last_hidden, kv_cache)."""
        for layer in expert.model.layers:
            residual       = hidden
            hidden_ln      = layer.input_layernorm(hidden)
            attn_out, _, _ = layer.self_attn(
                hidden_states=hidden_ln, attention_mask=causal_mask,
                position_ids=position_ids, past_key_value=kv_cache,
                output_attentions=False, use_cache=(kv_cache is not None),
            )
            hidden = residual + attn_out
            residual = hidden
            hidden   = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return expert.model.norm(hidden)   # (B, seq, H)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=128,
                 do_sample=False, temperature=1.0, top_p=0.9, top_k=0, **kwargs):
        device = input_ids.device
        B      = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ── Prefill: compute prompt gating weights ONCE ──────────────────
        prompt_hidden = self._embed(input_ids)
        past_len      = 0
        causal_mask   = self._build_causal_mask(attention_mask, prompt_hidden, past_len)
        position_ids  = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # Compute gating using model.forward() — same path as build_last_token_cache.
        gating_out = self.experts[0].model(input_ids=input_ids,
                                           attention_mask=attention_mask)
        last_h = self._last_token_hidden(gating_out.last_hidden_state, attention_mask)
        coeff  = self.gating_net(last_h)                           # (B, num_experts)

        # ── Build KV caches for all experts using the prompt ─────────────
        kv_caches = [DynamicCache() for _ in self.experts]
        # expert[0] cache already populated; redo for all experts in parallel.
        kv_caches = []
        expert_finals = []
        for expert in self.experts:
            kv = DynamicCache()
            h  = self._run_expert(expert, self._embed(input_ids),
                                  causal_mask, position_ids, kv)
            kv_caches.append(kv)
            expert_finals.append(h[:, -1:, :])   # last token only

        # ── Merge prefill last-token hidden states and get first logits ──
        coeff_t = coeff.to(expert_finals[0].dtype)   # (B, num_experts)
        merged  = sum(coeff_t[:, i].view(B, 1, 1) * expert_finals[i]
                      for i in range(self.num_experts))
        next_logits  = self.experts[0].lm_head(merged)[:, -1, :]

        # ── Decode ───────────────────────────────────────────────────────
        generated   = input_ids.clone()
        cur_attn    = attention_mask.clone()

        for step in range(max_new_tokens):
            if step == 0:
                logits = next_logits
            else:
                cur_ids     = generated[:, -1:]
                past_len    = self._pkv_seq_len(kv_caches[0])
                pos_ids     = cur_attn.sum(dim=-1, keepdim=True).long() - 1
                dec_mask    = self._build_causal_mask(cur_attn,
                                                      self._embed(cur_ids), past_len)
                finals = []
                for expert, kv in zip(self.experts, kv_caches):
                    h = self._run_expert(expert, self._embed(cur_ids),
                                         dec_mask, pos_ids, kv)
                    finals.append(h)
                merged = sum(coeff_t[:, i].view(B, 1, 1) * finals[i]
                             for i in range(self.num_experts))
                logits = self.experts[0].lm_head(merged)[:, -1, :]

            if do_sample:
                next_token = _sample_token(logits, temperature, top_p, top_k)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            cur_attn  = torch.cat(
                [cur_attn, torch.ones(B, 1, dtype=cur_attn.dtype, device=device)], dim=1)

            eos = getattr(self.config, 'eos_token_id', None)
            if eos is not None and (next_token.squeeze(-1) == eos).all():
                break

        return generated
