import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import get_clean_data


# ---------------------------------------------------------------------------
# Gating Network
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Per-layer gating network: hidden_states → combination coefficients for one layer.

    Called once per transformer layer with the layer's input hidden states.
    Mean-pools over the sequence dimension, then applies a two-layer MLP.

    Input:  hidden_states (batch, seq, lm_hidden_size)  — current layer input
    Output: (batch, num_experts)  — combination coefficients summing to 1
    """

    def __init__(self, lm_hidden_size=4096, num_experts=2, hidden_size=64):
        super().__init__()
        self.num_experts    = num_experts
        self.lm_hidden_size = lm_hidden_size
        self.hidden_size    = hidden_size

        self.net = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, seq, H) → mean-pool → (B, H)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)
        return F.softmax(self.net(hidden_states), dim=-1)  # (B, num_experts)


# ---------------------------------------------------------------------------
# MoE Language Model
# ---------------------------------------------------------------------------

def _sample_token(logits: torch.Tensor, temperature: float,
                  top_p: float, top_k: int) -> torch.Tensor:
    """Sample one token per batch item from logits. Returns (B, 1)."""
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

    Unlike weight merging (where parameter tensors are interpolated before
    inference), here ALL expert parameters stay separate and intact.  At every
    transformer depth l the same hidden states are fed to every expert's layer l
    in parallel; the outputs are then combined with per-layer GatingNetwork weights:

        w_l  = GatingNetwork( mean_pool(h_{l−1}) )   # (B, num_experts)
        h_l  = Σ_i  w_l[b, i] · ExpertI.layers[l]( h_{l−1} )

    The GatingNetwork is called once per layer during the forward pass using the
    layer's input hidden states.  Only the GatingNetwork is updated during MOEA/D;
    all expert LLMs are frozen.
    """

    def __init__(self, expert_models: list, gating_net: GatingNetwork):
        super().__init__()
        self.experts     = nn.ModuleList(expert_models)
        self.gating_net  = gating_net
        self.config      = expert_models[0].config
        self.num_experts = len(expert_models)
        self.num_layers  = len(expert_models[0].model.layers)

        for exp in self.experts:
            for p in exp.parameters():
                p.requires_grad = False

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings from the first expert (all experts share vocab/embedding)."""
        return self.experts[0].model.embed_tokens(input_ids)

    @staticmethod
    def _build_causal_mask(attention_mask: torch.Tensor,
                           hidden_states: torch.Tensor,
                           past_len: int) -> torch.Tensor:
        """Build a (B, 1, seq_len, past_len+seq_len) additive causal mask.

        Uses finfo(dtype).min as the mask value so it works with both
        float32 and bfloat16 activations.  Padding positions from
        attention_mask (0 = pad) are also masked out.
        """
        B, seq_len, _ = hidden_states.shape
        total_len = past_len + seq_len
        dtype     = hidden_states.dtype
        device    = hidden_states.device
        min_val   = torch.finfo(dtype).min

        # Lower-triangular causal mask: query i attends to keys 0 … past_len+i
        mask = torch.full((seq_len, total_len), min_val, dtype=dtype, device=device)
        causal_idx = torch.arange(seq_len, device=device)
        for q in range(seq_len):
            mask[q, : past_len + q + 1] = 0.0
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).clone()

        # Mask padded key positions
        if attention_mask is not None:
            # attention_mask: (B, total_len_or_less), 1=real 0=pad
            am = attention_mask[:, :total_len].to(dtype=dtype)  # (B, total_len)
            pad_mask = (1.0 - am) * min_val                     # (B, total_len)
            mask = mask + pad_mask.unsqueeze(1).unsqueeze(2)    # broadcast over (B,1,seq_len,total_len)

        return mask

    @staticmethod
    def _pkv_seq_len(kv_cache) -> int:
        """Return the sequence length already in the shared KV cache."""
        if kv_cache is None:
            return 0
        return kv_cache.get_seq_length()

    # ── Core MoE forward step ─────────────────────────────────────────────────

    def _moe_layers(self, hidden_states: torch.Tensor,
                    causal_mask,
                    position_ids: torch.Tensor,
                    kv_cache,
                    use_cache: bool = True,
                    seq_coefficients: list = None):
        """Pass hidden_states through all MoE layers.

        LoRA experts only differ in their FFN weights (gate_proj/up_proj/down_proj).
        Attention weights are identical across all experts (same as SFT base).

        Per layer:
          1. Shared self-attention using expert[0] weights + single KV cache.
          2. Gated FFN: each expert's FFN is run separately, outputs combined
             with GatingNetwork coefficients.

        seq_coefficients : list of num_layers tensors, each (B, num_experts),
            computed once per layer from prefill hidden states and reused during
            decode (per-layer, per-sequence gating).
            If None, computed fresh at each layer from the prefill hidden states.

        Returns (hidden_states, kv_cache, seq_coefficients).
        """
        if kv_cache is None:
            kv_cache = DynamicCache()

        B = hidden_states.shape[0]
        is_prefill = seq_coefficients is None
        if is_prefill:
            seq_coefficients = []

        for l in range(self.num_layers):
            layer0 = self.experts[0].model.layers[l]

            # 1. Shared attention (identical weights across all experts)
            residual       = hidden_states
            hidden_ln      = layer0.input_layernorm(hidden_states)
            attn_out, _, _ = layer0.self_attn(
                hidden_states=hidden_ln,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=kv_cache,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = residual + attn_out

            # 2. Gated FFN — per-layer, per-sequence gating.
            # During prefill: compute from this layer's hidden states and cache.
            # During decode: reuse the coefficient computed for this layer at prefill.
            if is_prefill:
                coeff = self.gating_net(hidden_states.float())  # (B, num_experts)
                seq_coefficients.append(coeff)
            else:
                coeff = seq_coefficients[l]
            coefficients = coeff.to(hidden_states.dtype)

            residual    = hidden_states
            ffn_outputs = [
                expert.model.layers[l].mlp(
                    expert.model.layers[l].post_attention_layernorm(hidden_states)
                )
                for expert in self.experts
            ]
            hidden_states = residual + sum(
                coefficients[:, i].view(B, 1, 1) * ffn_outputs[i]
                for i in range(self.num_experts)
            )

        return hidden_states, kv_cache, seq_coefficients

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor = None,
                 max_new_tokens: int = 128,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 top_k: int = 0,
                 **kwargs) -> torch.Tensor:
        """Autoregressive generation with per-layer MoE combination.

        Parameters
        ----------
        input_ids      : (B, prompt_len) — tokenised prompts.
        attention_mask : (B, prompt_len) — 1 for real tokens, 0 for padding.

        At each transformer layer the GatingNetwork produces combination coefficients
        from the current hidden states, applied directly to combine expert outputs.
        """
        device = input_ids.device
        B      = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated        = input_ids.clone()
        kv_cache         = DynamicCache()
        cur_attn         = attention_mask.clone()
        seq_coefficients = None   # computed once on prefill, reused every decode step

        for step in range(max_new_tokens):
            past_len = self._pkv_seq_len(kv_cache)
            cur_ids  = generated if step == 0 else generated[:, -1:]

            # Position ids derived from attention mask to handle left-padding correctly.
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

            # lm_head and model.norm are not LoRA targets — identical across experts.
            next_logits = self.experts[0].lm_head(
                self.experts[0].model.norm(hidden))[:, -1, :]   # (B, vocab)

            if do_sample:
                next_token = _sample_token(next_logits, temperature, top_p, top_k)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)
            cur_attn  = torch.cat(
                [cur_attn, torch.ones(B, 1, dtype=cur_attn.dtype, device=device)], dim=1)

            eos = getattr(self.config, 'eos_token_id', None)
            if eos is not None and (next_token.squeeze(-1) == eos).all():
                break

        return generated