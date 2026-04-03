"""
Multi-Turn Gating Network for LoRA Weight Merging
==================================================
Pipeline:
  1. Load two pre-trained LoRA adapters (helpful, harmless)
  2. Load HH-RLHF dataset and extract turn-level preference pairs
  3. Transformer-based attention gating network outputs λ = [λ_helpful, λ_harmless]
  4. Merge LoRA experts with λ, compute DPO loss, train gating network

Usage:
  pip install transformers peft datasets torch
  python multiturn_gating.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from torch.func import functional_call

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # ── model paths  (match your codebase conventions) ───────────────────────
    # sft_model_name   : base SFT model – used by load_main_tokenizer()
    # expert_model_paths : ordered list [helpful_path, harmless_path]
    #                      each path is passed to load_base_model()
    # gpu_id           : which GPU to place every model on
    sft_model_name: str         = "./models/sft/assistant_sft/model/"               # tokenizer source
    expert_model_paths: List[str] = field(
        default_factory=lambda: ["./models/ppo/assistant_ppo_harmless_2701/batch_832/", 
                                 "./models/ppo/assistant_ppo_helpful_2701/batch_832/"]
    )
    expert_names: List[str]     = field(
        default_factory=lambda: ["harmless", "helpful"]
    )
    gpu_id: int                 = 2

    # ── gating network ───────────────────────────────────────────────────────
    n_experts: int              = 2    # must equal len(expert_model_paths)
    # gate_hidden_dim is set automatically from lm_hidden_size after loading;
    # you can override it here if you want a projection layer instead.
    gate_hidden_dim: int        = 128
    gate_nhead: int             = 4
    gate_nlayers: int           = 2
    gate_dropout: float         = 0.1
    max_history_turns: int      = 8

    # ── training ─────────────────────────────────────────────────────────────
    dataset_name: str           = "Anthropic/hh-rlhf"
    max_seq_len: int            = 256
    batch_size: int             = 64
    grad_accum_steps: int       = 16
    lr: float                   = 3e-4
    n_epochs: int               = 3
    warmup_ratio: float         = 0.1
    beta: float                 = 0.1   # DPO temperature
    save_path: str              = "./gating_net.pt"


cfg = Config()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD BASE MODEL + TWO LoRA ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── stubs: replace with your actual implementations ──────────────────────────
# These three functions mirror the signatures used in your codebase.
# Swap out the bodies below with your real load_main_tokenizer /
# load_base_model implementations.


def load_base_model(model_path: str, target_device: str):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=target_device or 'auto',
    )
# ─────────────────────────────────────────────────────────────────────────────


def load_models(cfg: Config):
    """
    Adapted to match your codebase pattern:
      - load_main_tokenizer(cfg.sft_model_name)
      - load_base_model(path, target_device=f'cuda:{gpu_id}')  per expert
      - probe lm_hidden_size from the first expert's hidden states

    Returns
    -------
    tokenizer      : shared tokenizer
    feature_models : List[nn.Module]  – one frozen model per expert,
                     in the same order as cfg.expert_model_paths
    lora_deltas    : Dict[str -> Dict[str -> Tensor]]
                     ΔW = W_expert_merged - W_base  per expert name
    lm_hidden_size : int  – last hidden-state dimension (for gating net)
    """
    print("\n[1] Loading tokenizer and expert models ...")

    target_device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_name)

    # ── load every expert model (frozen) ──────────────────────────────────────
    expert_models: List[torch.nn.Module] = []
    for path in cfg.expert_model_paths:
        m = load_base_model(path, target_device=target_device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        expert_models.append(m)
        print(f"   Loaded expert from {path}")

    assert len(expert_models) == cfg.n_experts, (
        f"cfg.n_experts={cfg.n_experts} but "
        f"{len(expert_models)} paths given in cfg.expert_model_paths"
    )

    # ── probe lm_hidden_size ──────────────────────────────────────────────────
    with torch.no_grad():
        dummy     = tokenizer("hello", return_tensors="pt").to(target_device)
        dummy_out = expert_models[0](**dummy, output_hidden_states=True)
        lm_hidden_size = dummy_out.hidden_states[-1].shape[-1]
    print(f"   lm_hidden_size = {lm_hidden_size}")

    # ── extract ΔW task vectors ───────────────────────────────────────────────
    # We need a single "base" state-dict to subtract from each expert.
    # Strategy: use the first expert's base model weights.
    # If all experts share the same base (typical), this is exact.
    # For PeftModel, merge_and_unload() gives the full-rank merged weights.
    print("   Extracting ΔW task vectors ...")

    def _get_merged_sd(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Return a plain state-dict with LoRA weights merged in (if applicable)."""
        if isinstance(model, PeftModel):
            # merge LoRA into base weights → full-rank state-dict
            merged = model.merge_and_unload()
            return {k: v.clone() for k, v in merged.state_dict().items()}
        return {k: v.clone() for k, v in model.state_dict().items()}

    # Base state-dict: merge the first expert to get W_base + ΔW_0,
    # then we compute all deltas relative to a *shared* reference.
    # If you have an explicit separate base model, load it here instead.
    base_sd = _get_merged_sd(expert_models[0])

    # Re-load base without LoRA to get the true W_base.
    # We do this by loading the sft_model_name directly.
    print("   Loading base SFT model for ΔW reference ...")
    base_ref = AutoModelForCausalLM.from_pretrained(cfg.sft_model_name)
    base_ref_sd = {k: v.clone() for k, v in base_ref.state_dict().items()}
    del base_ref   # free memory immediately

    lora_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, model in zip(cfg.expert_names, expert_models):
        expert_sd = _get_merged_sd(model)
        delta: Dict[str, torch.Tensor] = {}
        for k, v in expert_sd.items():
            if k in base_ref_sd:
                diff = (v.to(target_device) - base_ref_sd[k].to(target_device))
                if diff.abs().max() > 1e-9:     # skip identical weights
                    delta[k] = diff
        lora_deltas[name] = delta
        print(f"   {name}: {len(delta)} non-zero ΔW tensors")

    return tokenizer, expert_models, lora_deltas, lm_hidden_size


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET  –  HH-RLHF  →  turn-level preference pairs
# ═══════════════════════════════════════════════════════════════════════════════

def parse_dialogue(text: str) -> List[Dict[str, str]]:
    """
    Parse HH-RLHF format:
      '\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...'
    into a list of {'role': 'human'/'assistant', 'content': '...'}
    """
    turns = []
    for chunk in text.strip().split("\n\nHuman: ")[1:]:
        parts = chunk.split("\n\nAssistant: ", 1)
        human_text = parts[0].strip()
        turns.append({"role": "human", "content": human_text})
        if len(parts) == 2:
            turns.append({"role": "assistant", "content": parts[1].strip()})
    return turns


def find_fork_turn(chosen_turns: List, rejected_turns: List) -> int:
    """Return index of first assistant turn where chosen != rejected.
    Returns -1 if no diverging assistant turn is found."""
    min_len = min(len(chosen_turns), len(rejected_turns))
    for i in range(min_len):
        if (chosen_turns[i]["role"] == "assistant" and
                chosen_turns[i]["content"] != rejected_turns[i]["content"]):
            return i
    # Fallback: return the last assistant turn present in both sequences
    for i in range(min_len - 1, -1, -1):
        if chosen_turns[i]["role"] == "assistant":
            return i
    return -1  # no assistant turn found


class HHRLHFTurnDataset(Dataset):
    """
    Each item:
      history_texts : List[str]  – previous assistant utterances (context)
      prompt        : str        – the human turn at the fork
      chosen        : str        – preferred assistant response
      rejected      : str        – dis-preferred assistant response
    """

    def __init__(self, cfg: Config, split: str = "train"):
        print(f"\n[2] Loading HH-RLHF ({split}) ...")
        raw = load_dataset(cfg.dataset_name, split=split,
                           streaming=False)
        self.items: List[Dict] = []

        for sample in raw:
            try:
                chosen_turns  = parse_dialogue(sample["chosen"])
                rejected_turns = parse_dialogue(sample["rejected"])
                fork = find_fork_turn(chosen_turns, rejected_turns)
                # fork must point to a valid assistant turn
                if fork <= 0:
                    continue
                if chosen_turns[fork]["role"] != "assistant":
                    continue

                # shared history = assistant turns before the fork
                history_texts = [
                    t["content"] for t in chosen_turns[:fork]
                    if t["role"] == "assistant"
                ][-cfg.max_history_turns:]

                # the human prompt at the fork
                human_prompt = ""
                for t in reversed(chosen_turns[:fork]):
                    if t["role"] == "human":
                        human_prompt = t["content"]
                        break

                self.items.append({
                    "history_texts": history_texts,
                    "prompt":        human_prompt,
                    "chosen":        chosen_turns[fork]["content"],
                    "rejected":      rejected_turns[fork]["content"],
                })
            except Exception:
                continue

        print(f"   {len(self.items)} turn-level pairs extracted.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, cfg):
    """
    Returns a dict of tensors ready for training.
    history_ids    : (B, T_hist, L)   – tokenised history turns
    history_mask   : (B, T_hist)      – 1 where turn exists
    prompt_ids     : (B, L)
    chosen_ids     : (B, L)
    rejected_ids   : (B, L)
    """
    B = len(batch)
    max_turns = max(len(x["history_texts"]) for x in batch)
    max_turns = max(max_turns, 1)

    def tok(texts, max_len):
        return tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt"
        )

    # ── history: (B, T_hist, L) ───────────────────────────────────────────
    L = cfg.max_seq_len
    hist_ids  = torch.zeros(B, max_turns, L, dtype=torch.long)
    hist_mask = torch.zeros(B, max_turns, dtype=torch.float)

    for b, item in enumerate(batch):
        for t, text in enumerate(item["history_texts"]):
            enc = tok([text], L)
            hist_ids[b, t]  = enc["input_ids"][0]
            hist_mask[b, t] = 1.0

    # ── prompt / chosen / rejected ────────────────────────────────────────
    prompts   = [x["prompt"]   for x in batch]
    chosens   = [x["chosen"]   for x in batch]
    rejecteds = [x["rejected"] for x in batch]

    prompt_enc   = tok(prompts,   L)
    chosen_enc   = tok(chosens,   L)
    rejected_enc = tok(rejecteds, L)

    # ── Build full (prompt + response) sequences for DPO log-prob scoring ─
    # The LM must condition on the prompt when scoring the response.
    # We tokenize each half to at most L//2 tokens, then concatenate.
    # chosen_resp_mask / rejected_resp_mask: 1 only at response token positions.
    half = L // 2
    chosen_full_ids    = torch.zeros(B, L, dtype=torch.long)
    rejected_full_ids  = torch.zeros(B, L, dtype=torch.long)
    chosen_resp_mask   = torch.zeros(B, L, dtype=torch.float)
    rejected_resp_mask = torch.zeros(B, L, dtype=torch.float)

    for b, item in enumerate(batch):
        p_ids = tokenizer(item["prompt"],   add_special_tokens=True,
                          truncation=True, max_length=half)["input_ids"]
        c_ids = tokenizer(item["chosen"],   add_special_tokens=False,
                          truncation=True, max_length=half)["input_ids"]
        r_ids = tokenizer(item["rejected"], add_special_tokens=False,
                          truncation=True, max_length=half)["input_ids"]

        def _fill(full_row, mask_row, resp_ids):
            combined = p_ids + resp_ids
            n = min(len(combined), L)
            full_row[:n] = torch.tensor(combined[:n], dtype=torch.long)
            resp_start = min(len(p_ids), L)
            resp_end   = min(len(p_ids) + len(resp_ids), L)
            if resp_end > resp_start:
                mask_row[resp_start:resp_end] = 1.0

        _fill(chosen_full_ids[b],   chosen_resp_mask[b],   c_ids)
        _fill(rejected_full_ids[b], rejected_resp_mask[b], r_ids)

    return {
        "history_ids":         hist_ids,
        "history_mask":        hist_mask,
        "prompt_ids":          prompt_enc["input_ids"],
        "chosen_ids":          chosen_enc["input_ids"],
        "rejected_ids":        rejected_enc["input_ids"],
        "chosen_mask":         chosen_enc["attention_mask"],
        "rejected_mask":       rejected_enc["attention_mask"],
        # Full prompt+response sequences for DPO scoring
        "chosen_full_ids":     chosen_full_ids,
        "rejected_full_ids":   rejected_full_ids,
        "chosen_resp_mask":    chosen_resp_mask,
        "rejected_resp_mask":  rejected_resp_mask,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TRANSFORMER-BASED GATING NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class TurnEncoder(nn.Module):
    """
    Encodes a single turn into a fixed-size vector.

    Two operating modes (selected at construction time):
    ─────────────────────────────────────────────────────
    Mode A – token-id input  (lightweight, no LLM needed)
      Inputs : ids (B, L)  – token ids
      Uses   : small learnable embedding + mean-pool
      When   : lm_hidden_size is None

    Mode B – LLM hidden-state input  (richer representations)
      Inputs : hidden (B, L, lm_hidden_size)  from feature_models
      Uses   : linear projection → gate_hidden_dim, then mean-pool
      When   : lm_hidden_size is provided

    In your pipeline the feature_models are already loaded and frozen,
    so Mode B just requires passing their last hidden states here.
    """

    def __init__(
        self,
        gate_hidden_dim: int,
        max_len: int,
        vocab_size: Optional[int]     = None,   # required for Mode A
        lm_hidden_size: Optional[int] = None,   # required for Mode B
    ):
        super().__init__()
        D = gate_hidden_dim
        self.lm_hidden_size = lm_hidden_size

        assert vocab_size is not None, "vocab_size required for TurnEncoder"
        # Id-mode path is always present (used during inference without a feature model)
        self.embed   = nn.Embedding(vocab_size, D, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, D)
        self.norm    = nn.LayerNorm(D)

        # Hidden-state projection (only when lm_hidden_size is provided)
        if lm_hidden_size is not None:
            self.proj      = nn.Linear(lm_hidden_size, D, bias=False)
            self.norm_proj = nn.LayerNorm(D)

    def forward(
        self,
        ids:    Optional[torch.Tensor] = None,   # (B, L)       – id mode
        hidden: Optional[torch.Tensor] = None,   # (B, L, H_lm) – hidden-state mode
    ) -> torch.Tensor:
        """Returns (B, D) turn representation.
        Uses hidden-state mode when hidden is provided, id mode otherwise."""
        if hidden is not None and self.lm_hidden_size is not None:
            x    = self.proj(hidden)                              # (B, L, D)
            x    = self.norm_proj(x)
            mask = (hidden.abs().sum(-1) > 1e-9).float().unsqueeze(-1)  # (B,L,1)
        else:
            assert ids is not None, "pass ids= when hidden states are not available"
            B, L = ids.shape
            pos  = torch.arange(L, device=ids.device).unsqueeze(0)
            x    = self.embed(ids) + self.pos_emb(pos)           # (B, L, D)
            x    = self.norm(x)
            mask = (ids != 0).float().unsqueeze(-1)              # (B, L, 1)

        # mean-pool over non-padding positions
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)     # (B, D)


class MultiTurnGatingNetwork(nn.Module):
    """
    Transformer-based gating network.

    Architecture
    ────────────
    1. TurnEncoder  :  each turn → vector of size gate_hidden_dim
       - if lm_hidden_size is given: projects from LLM hidden states
         (the last hidden layer of feature_models[0])
       - otherwise: uses a small learnable token embedding
    2. Transformer encoder (self-attention over [prompt | hist_1 | … | hist_T])
       – prompt token attends to all history turns freely
    3. MLP head on the prompt position's output → λ ∈ Δ^{N-1}

    Parameters
    ----------
    cfg            : Config
    vocab_size     : tokenizer vocab size (needed when NOT using LLM hidden states)
    lm_hidden_size : last hidden dim reported by feature_models (may be None)
    """

    def __init__(
        self,
        cfg: Config,
        vocab_size: int,
        lm_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        D = cfg.gate_hidden_dim
        self.lm_hidden_size = lm_hidden_size

        self.turn_enc = TurnEncoder(
            gate_hidden_dim = D,
            max_len         = cfg.max_seq_len,
            vocab_size      = vocab_size,
            lm_hidden_size  = lm_hidden_size,
        )

        # Unused projection stubs kept for API compatibility
        self.query_proj = nn.Linear(D, D)
        self.key_proj   = nn.Linear(D, D)
        self.val_proj   = nn.Linear(D, D)

        # Transformer encoder layers (self-attention over history + prompt)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.gate_nhead,
            dim_feedforward=D * 4,
            dropout=cfg.gate_dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 num_layers=cfg.gate_nlayers)

        # MLP head: context → λ
        self.head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(cfg.gate_dropout),
            nn.Linear(D // 2, cfg.n_experts),
        )

    def forward(
        self,
        history_ids:     torch.Tensor,           # (B, T, L)       always required
        history_mask:    torch.Tensor,           # (B, T)          1=real, 0=pad
        prompt_ids:      torch.Tensor,           # (B, L)          always required
        # ── optional: pass LLM hidden states instead of re-encoding from ids ──
        # Obtain these by running feature_models[0] on the token ids with
        # output_hidden_states=True, then taking .hidden_states[-1].
        # Shape: (B, T, L, lm_hidden_size) for history,
        #        (B, L, lm_hidden_size)    for prompt.
        history_hidden:  Optional[torch.Tensor] = None,  # (B, T, L, H_lm)
        prompt_hidden:   Optional[torch.Tensor] = None,  # (B, L, H_lm)
    ) -> torch.Tensor:
        """
        Returns
        -------
        lambda_weights : (B, N_experts)  – softmax-normalised merging weights
        """
        B, T, L = history_ids.shape
        use_hidden = (self.lm_hidden_size is not None
                      and history_hidden is not None
                      and prompt_hidden  is not None)

        if use_hidden:
            # ── Mode B: encode from LLM hidden states ─────────────────────
            # history_hidden : (B, T, L, H_lm)  →  flatten to (B*T, L, H_lm)
            H_lm = history_hidden.shape[-1]
            hist_flat_h  = history_hidden.view(B * T, L, H_lm)
            hist_enc     = self.turn_enc(hidden=hist_flat_h).view(B, T, -1)  # (B,T,D)
            prompt_enc   = self.turn_enc(hidden=prompt_hidden).unsqueeze(1)  # (B,1,D)
        else:
            # ── Mode A: encode from token ids (lightweight fallback) ───────
            hist_flat  = history_ids.view(B * T, L)
            hist_enc   = self.turn_enc(ids=hist_flat).view(B, T, -1)        # (B,T,D)
            prompt_enc = self.turn_enc(ids=prompt_ids).unsqueeze(1)         # (B,1,D)

        # ── concatenate: [prompt | history turns] ─────────────────────────
        sequence = torch.cat([prompt_enc, hist_enc], dim=1)    # (B, T+1, D)

        # src_key_padding_mask: True = position is padding (ignored)
        prompt_valid = torch.ones(B, 1, device=history_mask.device)
        valid_mask   = torch.cat([prompt_valid, history_mask], dim=1)  # (B, T+1)
        padding_mask = (valid_mask == 0)                               # (B, T+1)

        # ── transformer self-attention ─────────────────────────────────────
        out = self.transformer(sequence,
                               src_key_padding_mask=padding_mask)  # (B, T+1, D)

        # Prompt position aggregates all history context via attention
        context = out[:, 0, :]                    # (B, D)

        # ── MLP head → λ ──────────────────────────────────────────────────
        logits = self.head(context)               # (B, N_experts)
        lam    = F.softmax(logits, dim=-1)        # (B, N_experts)
        return lam


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  LoRA MERGING  +  LOG-PROB COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_lambda_to_model(
    base_model:   nn.Module,
    lora_deltas:  Dict[str, Dict[str, torch.Tensor]],
    lam:          torch.Tensor,           # (N_experts,)  scalar per expert
) -> Dict[str, torch.Tensor]:
    """
    Compute the merged parameter dict:
        W_merged = W_base + Σ_i λ_i * ΔW_i

    Returns an overriding state-dict (only the keys that have deltas).
    Does NOT modify base_model in place – we use functional_call instead.
    """
    expert_names = list(lora_deltas.keys())   # ["harmless", "helpful"]
    override: Dict[str, torch.Tensor] = {}

    # Collect all param keys that appear in at least one adapter
    all_keys = set()
    for deltas in lora_deltas.values():
        all_keys.update(deltas.keys())

    base_sd = dict(base_model.named_parameters())

    for key in all_keys:
        base_val = base_sd.get(key)
        if base_val is None:
            continue
        # Accumulate in float32 to avoid dtype promotion issues
        # (model weights are bfloat16, lam is float32).
        # Cast back to the original parameter dtype before storing.
        orig_dtype = base_val.dtype
        merged = base_val.float().clone()
        for i, name in enumerate(expert_names):
            if key in lora_deltas[name]:
                delta = lora_deltas[name][key].to(merged.device).float()
                merged = merged + lam[i].float() * delta
        override[key] = merged.to(orig_dtype)

    return override


def compute_log_prob(
    base_model:   nn.Module,
    lora_deltas:  Dict[str, Dict[str, torch.Tensor]],
    lam:          torch.Tensor,   # (B, N)
    input_ids:    torch.Tensor,   # (B, L)  context+response tokens
    response_ids: torch.Tensor,   # (B, L)  response tokens (same shape)
    response_mask:torch.Tensor,   # (B, L)  1 where response token
) -> torch.Tensor:
    """
    Compute per-sample log p(response | context, λ-merged model).

    We iterate over the batch because each sample has its own λ.
    For efficiency in practice, batch-merge with vmap or use a shared λ.
    """
    B = input_ids.shape[0]
    log_probs = []

    for b in range(B):
        lam_b      = lam[b]                        # (N,)
        override   = apply_lambda_to_model(base_model, lora_deltas, lam_b)

        # Functional forward pass with overridden weights
        # torch.func.functional_call avoids copying the entire model
        params_and_buffers = dict(base_model.named_parameters())
        params_and_buffers.update(override)

        # No torch.no_grad() here — gradients must flow through
        # lam → override → logits so the DPO loss can update the gate.
        logits = functional_call(
            base_model,
            params_and_buffers,
            (input_ids[b:b+1],),
        ).logits                              # (1, L, V)

        # shift: predict token t+1 from position t
        shift_logits  = logits[:, :-1, :].contiguous()   # (1, L-1, V)
        shift_labels  = response_ids[b:b+1, 1:].contiguous()  # (1, L-1)
        shift_mask    = response_mask[b:b+1, 1:].float()

        # cross-entropy per token
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(1, -1)                            # (1, L-1)

        # mean log-prob over response tokens only
        lp = -(ce * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
        log_probs.append(lp)

    return torch.cat(log_probs, dim=0)           # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DPO LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def dpo_loss(
    log_prob_chosen:       torch.Tensor,  # (B,)
    log_prob_rejected:     torch.Tensor,  # (B,)
    log_prob_chosen_ref:   torch.Tensor,  # (B,)  reference (uniform λ)
    log_prob_rejected_ref: torch.Tensor,  # (B,)
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Standard DPO objective:
      L = -E[ log σ( β*(log π/π_ref)_chosen - β*(log π/π_ref)_rejected ) ]
    """
    chosen_reward   = beta * (log_prob_chosen   - log_prob_chosen_ref)
    rejected_reward = beta * (log_prob_rejected - log_prob_rejected_ref)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: Config):

    # ── load everything ───────────────────────────────────────────────────────
    tokenizer, feature_models, lora_deltas, lm_hidden_size = load_models(cfg)

    # Use the first feature_model as the "base" for log-prob computation.
    # All feature_models share the same base weights + different LoRA deltas,
    # so any of them works as the structural reference for functional_call.
    base_model = feature_models[0]
    target_device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = HHRLHFTurnDataset(cfg, split="train")

    def _collate(batch):
        return collate_fn(batch, tokenizer, cfg)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=True,
    )

    # ── gating network ────────────────────────────────────────────────────────
    print("\n[3] Building gating network ...")
    gate = MultiTurnGatingNetwork(
        cfg,
        vocab_size      = len(tokenizer),
        lm_hidden_size  = lm_hidden_size,   # enables hidden-state mode
    ).to(target_device)
    n_params = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    print(f"   Gating network parameters: {n_params:,}")
    print(f"   TurnEncoder mode: {'LLM hidden states' if lm_hidden_size else 'token ids'}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(gate.parameters(), lr=cfg.lr,
                                  weight_decay=0.01)
    total_steps   = math.ceil(len(loader) / cfg.grad_accum_steps) * cfg.n_epochs
    warmup_steps  = int(total_steps * cfg.warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # reference λ = uniform [1/N, 1/N, …]
    n_exp      = cfg.n_experts
    ref_lambda = torch.full((n_exp,), 1.0 / n_exp, device=target_device)

    print(f"\n[4] Training for {cfg.n_epochs} epochs "
          f"({total_steps} optimisation steps) ...\n")

    global_step = 0
    for epoch in range(cfg.n_epochs):
        gate.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):

            # ── move to device ────────────────────────────────────────────
            history_ids   = batch["history_ids"].to(target_device)    # (B, T, L)
            history_mask  = batch["history_mask"].to(target_device)   # (B, T)
            prompt_ids    = batch["prompt_ids"].to(target_device)     # (B, L)
            chosen_ids         = batch["chosen_ids"].to(target_device)     # (B, L)
            rejected_ids       = batch["rejected_ids"].to(target_device)
            chosen_mask        = batch["chosen_mask"].to(target_device)
            rejected_mask      = batch["rejected_mask"].to(target_device)
            chosen_full_ids    = batch["chosen_full_ids"].to(target_device)
            rejected_full_ids  = batch["rejected_full_ids"].to(target_device)
            chosen_resp_mask   = batch["chosen_resp_mask"].to(target_device)
            rejected_resp_mask = batch["rejected_resp_mask"].to(target_device)

            # ── optionally extract LLM hidden states for richer encoding ──
            # This runs the frozen feature_model once per batch to get
            # the last-layer hidden states for all history turns and the prompt.
            # Skip (set to None) if you want the lightweight id-based mode.
            history_hidden = None
            prompt_hidden  = None
            if lm_hidden_size is not None:
                with torch.no_grad():
                    B, T, L = history_ids.shape
                    # flatten history turns → encode → reshape
                    hist_flat = history_ids.view(B * T, L)
                    h_out = base_model(hist_flat, output_hidden_states=True)
                    history_hidden = (
                        h_out.hidden_states[-1].float()   # (B*T, L, H_lm) → fp32
                        .view(B, T, L, lm_hidden_size)
                    )
                    p_out = base_model(prompt_ids, output_hidden_states=True)
                    prompt_hidden = p_out.hidden_states[-1].float()  # (B, L, H_lm) → fp32

            # ── gating network forward: λ  (B, N) ────────────────────────
            lam = gate(
                history_ids,  history_mask, prompt_ids,
                history_hidden=history_hidden,
                prompt_hidden=prompt_hidden,
            )  # (B, N)

            # ── reference λ expanded to batch ────────────────────────────
            ref_lam = ref_lambda.unsqueeze(0).expand(lam.shape[0], -1)

            # ── log-probs under gated λ (prompt+response as input) ───────
            # input_ids = full sequence so the LM conditions on the prompt;
            # response_mask = 1 only at response token positions.
            lp_chosen   = compute_log_prob(base_model, lora_deltas, lam,
                                           chosen_full_ids,   chosen_full_ids,   chosen_resp_mask)
            lp_rejected = compute_log_prob(base_model, lora_deltas, lam,
                                           rejected_full_ids, rejected_full_ids, rejected_resp_mask)

            # ── log-probs under reference λ (no grad) ─────────────────────
            with torch.no_grad():
                lp_chosen_ref   = compute_log_prob(base_model, lora_deltas, ref_lam,
                                                   chosen_full_ids,   chosen_full_ids,   chosen_resp_mask)
                lp_rejected_ref = compute_log_prob(base_model, lora_deltas, ref_lam,
                                                   rejected_full_ids, rejected_full_ids, rejected_resp_mask)

            # ── DPO loss ──────────────────────────────────────────────────
            loss = dpo_loss(lp_chosen, lp_rejected,
                            lp_chosen_ref, lp_rejected_ref,
                            beta=cfg.beta)
            loss = loss / cfg.grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * cfg.grad_accum_steps

            # ── gradient accumulation step ────────────────────────────────
            if (step + 1) % cfg.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    avg = epoch_loss / (step + 1)
                    lam_mean = lam.detach().mean(0).cpu().tolist()
                    lam_str  = ", ".join(f"{v:.3f}" for v in lam_mean)
                    print(f"   epoch {epoch+1}  step {global_step:4d}  "
                          f"loss={avg:.4f}  λ_mean=[{lam_str}]")

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\n── Epoch {epoch+1} done  avg_loss={avg_epoch_loss:.4f} ──\n")

    # ── save ─────────────────────────────────────────────────────────────────
    torch.save(gate.state_dict(), cfg.save_path)
    print(f"[✓] Gating network saved to {cfg.save_path}")
    return gate


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  INFERENCE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_lambda(
    gate:         MultiTurnGatingNetwork,
    tokenizer,
    cfg:          Config,
    history:      List[str],   # previous assistant turns
    prompt:       str,
) -> List[float]:
    """
    Given a conversation history and new prompt, return λ weights.

    Example
    -------
    >>> lam = infer_lambda(gate, tokenizer, cfg,
    ...     history=["Sure, here is a Python snippet ...",
    ...              "You're welcome! Let me know ..."],
    ...     prompt="Actually, can you make it safer?")
    >>> print(lam)   # e.g. [0.23, 0.77]  → harmless expert dominates
    """
    gate.eval()
    target_device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
    L = cfg.max_seq_len

    def tok(texts):
        return tokenizer(texts, padding="max_length", truncation=True,
                         max_length=L, return_tensors="pt")["input_ids"]

    T = min(len(history), cfg.max_history_turns)
    history = history[-T:] if T > 0 else []

    # handle empty history gracefully
    if T == 0:
        T = 1
        hist_ids  = torch.zeros(1, 1, L, dtype=torch.long)
        hist_mask = torch.zeros(1, 1)
    else:
        hist_ids  = torch.zeros(1, T, L, dtype=torch.long)
        hist_mask = torch.zeros(1, T)
        for i, h in enumerate(history):
            hist_ids[0, i]  = tok([h])[0]
            hist_mask[0, i] = 1.0

    prompt_ids = tok([prompt])

    hist_ids   = hist_ids.to(target_device)
    hist_mask  = hist_mask.to(target_device)
    prompt_ids = prompt_ids.to(target_device)

    # id-mode only at inference (no feature_model needed)
    lam = gate(hist_ids, hist_mask, prompt_ids,
               history_hidden=None, prompt_hidden=None)   # (1, N)
    return lam[0].cpu().tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    gate = train(cfg)

    # ── quick inference demo ─────────────────────────────────────────────────
    print("\n── Inference demo ──")
    # tokenizer was already loaded inside train(); reload for standalone use
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_name)

    examples = [
        {
            "history": [],
            "prompt":  "How do I make a bomb?",
            "expect":  "harmless weight should be HIGH",
        },
        {
            "history": ["Sure! Here are three tips for healthy eating ..."],
            "prompt":  "Can you give me more detailed advice?",
            "expect":  "helpful weight should be HIGH",
        },
        {
            "history": [
                "I can help you with that coding problem.",
                "Here is the fixed version of your script.",
            ],
            "prompt":  "Actually, I think the previous answer was dangerous. Can you reconsider?",
            "expect":  "harmless weight rises due to 'dangerous' signal",
        },
    ]

    for ex in examples:
        lam = infer_lambda(gate, tokenizer, cfg, ex["history"], ex["prompt"])
        print(f"\n  Prompt : {ex['prompt'][:70]}")
        print(f"  History: {len(ex['history'])} turn(s)")
        lam_str = "  ".join(
            f"{n}={v:.3f}" for n, v in zip(cfg.expert_names, lam)
        )
        print(f"  λ       : {lam_str}")
        print(f"  Expect  : {ex['expect']}")