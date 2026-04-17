"""nsgaii_architecture.py — GatingNetwork + MoEForCausalLM for NSGA-II.

Standard NSGA-II design: each population individual is a plain GatingNetwork
(no preference conditioning).  At inference, select the individual whose
reward vector best matches the desired preference:
    best_i = argmax_i  dot(λ, fitness_i)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import get_clean_data


# ---------------------------------------------------------------------------
# Gating Network  (identical to moead_architecture.GatingNetwork)
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Per-layer gating: mean-pool hidden states → expert merging weights.

    Input:  (B, seq, lm_hidden_size)  or  (B, lm_hidden_size)
    Output: (B, num_experts)  — sum to 1 via softmax
    """

    def __init__(self, lm_hidden_size: int = 4096, num_experts: int = 2,
                 hidden_size: int = 256):
        super().__init__()
        self.lm_hidden_size = lm_hidden_size
        self.num_experts    = num_experts
        self.hidden_size    = hidden_size
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=1)
        return F.softmax(self.net(hidden_states.float()), dim=-1)


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
                coeff = self.gating_net(hidden_states.float())
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
