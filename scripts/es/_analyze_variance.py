"""_analyze_variance.py — Measure hidden-state and gating-coefficient anisotropy.

Captures real LLM hidden states for prompts in a gating dataset CSV, then
passes them through both GatingNetwork and SimpleGatingNetwork checkpoints
to measure how much the output coefficients vary across prompts.

Questions answered:
  1. Are per-layer last-token hidden states (GatingNetwork input) anisotropic?
  2. Are last-token hidden states (SimpleGatingNetwork input) anisotropic?
  3. Does each gating checkpoint produce consistent or prompt-varying coefficients?

TO RUN:
    accelerate launch --num_processes=1 scripts/es/_analyze_variance.py \
        --base_model_name meta-llama/Llama-2-7b-hf \
        --expert_model_paths ./models/ppo/ppo_beaver_reward_2204/best_model \
                             ./models/ppo/ppo_beaver_cost_2204/best_model \
        --gating_dataset_csv ./results/optimal/optimal_beaver_1205/gating_dataset.csv \
        --gating_paths_per_layer ./models/ES/dummy_beaver/final \
        --gating_paths_simple    ./models/gating_pretrain_1205/gating_pretrain_beaver \
        --max_prompts 200 --batch_size 16 --max_prompt_len 256
"""

import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import PeftModel
from transformers import AutoModelForCausalLM, HfArgumentParser
from trl import set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from es_architecture import GatingNetwork, SimpleGatingNetwork, _apply_entmax
from es_utils import load_gating_network, load_main_tokenizer, load_simple_gating_network


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:        str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths:     List[str] = field(default_factory=list)
    gating_dataset_csv:     str       = './results/optimal/optimal_beaver_1205/gating_dataset.csv'
    # Directories or glob patterns for checkpoints to test
    gating_paths_per_layer: List[str] = field(default_factory=list)  # GatingNetwork
    gating_paths_simple:    List[str] = field(default_factory=list)  # SimpleGatingNetwork
    max_prompts:            int       = 200    # cap prompts for speed
    batch_size:             int       = 16
    max_prompt_len:         int       = 256
    gpu_id:                 int       = -1
    seed:                   int       = 8888
    output_json:            str       = ''     # optional: dump full results


# ---------------------------------------------------------------------------
# Hidden state extraction  (mirrors pretraining_opt.py exactly)
# ---------------------------------------------------------------------------

def build_per_layer_cache(prompts, expert_model, tokenizer,
                          max_length, batch_size, device):
    """(N, num_layers, H) float32 CPU — post-attn last real token per layer.

    Matches GatingNetwork.forward, which reduces a (B, seq, H) input to its
    last position; we gather the last non-padding token so this is correct
    regardless of the tokenizer's padding side.
    """
    enc = tokenizer(prompts, max_length=max_length, truncation=True,
                    padding=True, return_tensors='pt')
    ids  = enc['input_ids']
    mask = enc['attention_mask']
    N    = len(prompts)
    num_layers = len(expert_model.model.layers)
    all_hidden = []

    for start in range(0, N, batch_size):
        b_ids  = ids[start:start + batch_size].to(device)
        b_mask = mask[start:start + batch_size].to(device)

        post_attn = {}

        def make_hook(l, layer_mod):
            def hook(module, inp, output):
                residual = layer_mod._hook_residual
                post_attn[l] = (residual + output[0]).detach().float().cpu()
            return hook

        def make_pre(layer_mod):
            def pre(module, inp):
                layer_mod._hook_residual = inp[0].detach()
            return pre

        handles = []
        for l, layer in enumerate(expert_model.model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre(layer)))
            handles.append(layer.self_attn.register_forward_hook(make_hook(l, layer)))

        with torch.no_grad():
            expert_model(input_ids=b_ids, attention_mask=b_mask)

        for h in handles:
            h.remove()

        stacked  = torch.stack([post_attn[l] for l in range(num_layers)], dim=1)  # (B, num_layers, seq, H)
        last_idx = (b_mask.sum(dim=1) - 1).clamp(min=0).cpu()                     # (B,)
        idx      = last_idx.view(-1, 1, 1, 1).expand(-1, num_layers, 1, stacked.shape[-1])
        all_hidden.append(stacked.gather(2, idx).squeeze(2))  # (B, num_layers, H) — last real token

    return torch.cat(all_hidden, dim=0)  # (N, num_layers, H)


def build_last_token_cache(prompts, expert_model, tokenizer,
                           max_length, batch_size, device):
    """(N, H) float32 CPU — last non-padding token from final layer (post-norm)."""
    enc = tokenizer(prompts, max_length=max_length, truncation=True,
                    padding=True, return_tensors='pt')
    ids  = enc['input_ids']
    mask = enc['attention_mask']
    N    = len(prompts)
    all_last = []

    for start in range(0, N, batch_size):
        b_ids  = ids[start:start + batch_size].to(device)
        b_mask = mask[start:start + batch_size].to(device)
        with torch.no_grad():
            out = expert_model.model(input_ids=b_ids, attention_mask=b_mask)
        final_h  = out.last_hidden_state.float().cpu()     # (B, seq, H)
        last_idx = (b_mask.sum(dim=1) - 1).clamp(min=0).cpu()
        idx      = last_idx.view(-1, 1, 1).expand(-1, 1, final_h.shape[-1])
        all_last.append(final_h.gather(1, idx).squeeze(1))  # (B, H)

    return torch.cat(all_last, dim=0)  # (N, H)


# ---------------------------------------------------------------------------
# Coefficient analysis
# ---------------------------------------------------------------------------

def coeff_stats(coeffs: torch.Tensor):
    """coeffs: (N, num_experts). Return per-expert mean/std/min/max for expert 0."""
    w0 = coeffs[:, 0]
    return {
        'mean':  float(w0.mean()),
        'std':   float(w0.std()),
        'min':   float(w0.min()),
        'max':   float(w0.max()),
        'range': float(w0.max() - w0.min()),
    }


@torch.no_grad()
def run_per_layer_gating(gating: GatingNetwork,
                         hidden_cache: torch.Tensor) -> torch.Tensor:
    """
    hidden_cache: (N, num_layers, H)
    Returns (N, num_experts) — average coefficient over all layers per prompt.
    """
    N, L, H = hidden_cache.shape
    alphas   = gating.alpha_floats()
    per_layer_coeffs = []

    for l in range(L):
        h_l = hidden_cache[:, l, :].bfloat16().to(next(gating.parameters()).device)
        logits = gating.net(h_l)                          # (N, num_experts)
        alpha  = alphas[l]
        coeffs = _apply_entmax(logits.float(), alpha)     # (N, num_experts)
        per_layer_coeffs.append(coeffs.cpu())

    return torch.stack(per_layer_coeffs, dim=0).mean(dim=0)  # (N, num_experts)


@torch.no_grad()
def run_simple_gating(gating: SimpleGatingNetwork,
                      last_token_cache: torch.Tensor) -> torch.Tensor:
    """
    last_token_cache: (N, H)
    Returns (N, num_experts).
    """
    h = last_token_cache.bfloat16().to(next(gating.parameters()).device)
    logits = gating.net(h)                                   # (N, num_experts)
    alpha  = gating.fixed_alpha if gating.fixed_alpha is not None else 1.0
    return _apply_entmax(logits.float(), alpha).cpu()        # (N, num_experts)


# ---------------------------------------------------------------------------
# Resolve checkpoint paths  (single dir or dir containing sub-dirs)
# ---------------------------------------------------------------------------

def resolve_ckpts(paths: List[str]) -> List[str]:
    resolved = []
    for p in paths:
        if os.path.isfile(os.path.join(p, 'gating_network.pt')):
            resolved.append(p)
        elif os.path.isdir(p):
            # look one level deep for sub-dirs that contain a checkpoint
            for sub in sorted(os.listdir(p)):
                full = os.path.join(p, sub)
                if os.path.isfile(os.path.join(full, 'gating_network.pt')):
                    resolved.append(full)
    return resolved


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_section(title):
    print(f'\n{"=" * 70}')
    print(f'  {title}')
    print('=' * 70)


def print_stats_table(rows):
    """rows: list of (label, stats_dict)"""
    hdr = f"{'checkpoint':<35} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'range':>7}  note"
    print(hdr)
    print('-' * len(hdr))
    for label, s in rows:
        note = 'CONSISTENT' if s['std'] < 0.05 else ('mixed' if s['std'] < 0.20 else 'RANDOM')
        print(f"{label:<35} {s['mean']:>7.3f} {s['std']:>7.3f} "
              f"{s['min']:>7.3f} {s['max']:>7.3f} {s['range']:>7.3f}  {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser      = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    np.random.seed(script_args.seed)

    if 'RANK' in os.environ:
        torch.distributed.init_process_group(
            backend='nccl', timeout=datetime.timedelta(minutes=600))
    accelerator = Accelerator()
    gpu_id = (script_args.gpu_id if script_args.gpu_id >= 0
               else accelerator.local_process_index)
    device = f'cuda:{gpu_id}'

    # ── Load prompts ─────────────────────────────────────────────────────────
    df = pd.read_csv(script_args.gating_dataset_csv)
    prompts = (df.drop_duplicates('prompt_idx')
                 .sort_values('prompt_idx')['prompt_text']
                 .tolist()[:script_args.max_prompts])
    print(f'Loaded {len(prompts)} prompts from {script_args.gating_dataset_csv}')

    # ── Load tokenizer + expert[0] ───────────────────────────────────────────
    sft_tokenizer = load_main_tokenizer(script_args.expert_model_paths[0])
    print(f'\nLoading expert[0] for hidden state extraction: '
          f'{script_args.expert_model_paths[0]}')
    base = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    expert0 = PeftModel.from_pretrained(base, script_args.expert_model_paths[0]).merge_and_unload()
    expert0.resize_token_embeddings(len(sft_tokenizer))
    expert0.eval()
    for p in expert0.parameters():
        p.requires_grad = False

    lm_hidden_size = expert0.config.hidden_size
    num_layers     = len(expert0.model.layers)
    print(f'  lm_hidden_size={lm_hidden_size}, num_layers={num_layers}')

    # ── Build hidden state caches ─────────────────────────────────────────────
    need_per_layer = len(script_args.gating_paths_per_layer) > 0
    need_simple    = len(script_args.gating_paths_simple) > 0

    per_layer_cache = last_token_cache = None

    if need_per_layer:
        print(f'\nBuilding last token per-layer hidden cache ({len(prompts)} prompts) …')
        per_layer_cache = build_per_layer_cache(
            prompts, expert0, sft_tokenizer,
            script_args.max_prompt_len, script_args.batch_size, device)
        print(f'  shape: {tuple(per_layer_cache.shape)}')

        # ── Anisotropy stats for per-layer cache ─────────────────────────────
        # Flatten to (N*L, H), compute mean norm and std of projections
        flat = per_layer_cache.view(-1, lm_hidden_size).float()  # (N*L, H)
        norms = flat.norm(dim=-1)
        mu    = flat.mean(dim=0)
        mu_norm = mu.norm().item()
        centered = flat - mu
        centered_norm = centered.norm(dim=-1).mean().item()
        print(f'  Anisotropy check (per-layer):')
        print(f'    mean vector norm   : {mu_norm:.2f}')
        print(f'    avg deviation norm : {centered_norm:.2f}')
        print(f'    ratio mu/deviation : {mu_norm / max(centered_norm, 1e-6):.2f}x  '
              f'(>1 = anisotropic)')

    if need_simple:
        print(f'\nBuilding last-token single hidden cache ({len(prompts)} prompts) …')
        last_token_cache = build_last_token_cache(
            prompts, expert0, sft_tokenizer,
            script_args.max_prompt_len, script_args.batch_size, device)
        print(f'  shape: {tuple(last_token_cache.shape)}')

        flat2   = last_token_cache.float()
        mu2     = flat2.mean(dim=0)
        mu2_norm = mu2.norm().item()
        c2_norm  = (flat2 - mu2).norm(dim=-1).mean().item()
        print(f'  Anisotropy check (single):')
        print(f'    mean vector norm   : {mu2_norm:.2f}')
        print(f'    avg deviation norm : {c2_norm:.2f}')
        print(f'    ratio mu/deviation : {mu2_norm / max(c2_norm, 1e-6):.2f}x')

    # Free expert0 from GPU
    del expert0
    torch.cuda.empty_cache()

    all_results = {}

    # ── GatingNetwork checkpoints ────────────────────────────────
    ckpts_per_layer = resolve_ckpts(script_args.gating_paths_per_layer)
    if ckpts_per_layer:
        print_section(f'GatingNetwork (last-token)  —  {len(ckpts_per_layer)} checkpoints')
        rows = []
        for ckpt in ckpts_per_layer:
            name = os.path.basename(ckpt)
            g = load_gating_network(ckpt, lm_hidden_size=lm_hidden_size,
                                    num_experts=2, num_layers=num_layers, device='cpu')
            if g is None:
                print(f'  [SKIP] {name}')
                continue
            g.eval()
            coeffs = run_per_layer_gating(g, per_layer_cache)  # (N, 2)
            s = coeff_stats(coeffs)
            rows.append((name, s))
            all_results[f'per_layer/{name}'] = s
        print_stats_table(rows)

    # ── SimpleGatingNetwork checkpoints ──────────────────────────────────────
    ckpts_simple = resolve_ckpts(script_args.gating_paths_simple)
    if ckpts_simple:
        print_section(f'SimpleGatingNetwork (last-token)  —  {len(ckpts_simple)} checkpoints')
        rows = []
        for ckpt in ckpts_simple:
            name = os.path.basename(ckpt)
            g = load_simple_gating_network(ckpt, lm_hidden_size=lm_hidden_size,
                                           num_experts=2, device='cpu')
            if g is None:
                print(f'  [SKIP] {name}')
                continue
            g.eval()
            coeffs = run_simple_gating(g, last_token_cache)   # (N, 2)
            s = coeff_stats(coeffs)
            rows.append((name, s))
            all_results[f'simple/{name}'] = s
        print_stats_table(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section('Summary')
    consistent = sum(1 for s in all_results.values() if s['std'] < 0.05)
    mixed      = sum(1 for s in all_results.values() if 0.05 <= s['std'] < 0.20)
    random_    = sum(1 for s in all_results.values() if s['std'] >= 0.20)
    print(f'  CONSISTENT (std < 0.05) : {consistent}')
    print(f'  mixed      (0.05–0.20)  : {mixed}')
    print(f'  RANDOM     (std ≥ 0.20) : {random_}')

    if script_args.output_json:
        with open(script_args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n  Full results saved → {script_args.output_json}')
