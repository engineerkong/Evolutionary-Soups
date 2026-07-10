"""_pretraining.py — Supervised pretraining of GatingNetwork from an optimal-weight CSV.

For each preference λ on the simplex, train one gating network with MSE loss:
    loss = MSE( gating(hidden(prompt)), optimal_w(prompt, λ) )
where `optimal_w(prompt, λ)` is read from a per-prompt CSV produced upstream
and `hidden(prompt)` is taken from the loaded experts.

Efficiency: expert forward passes run ONCE per prompt and the resulting
hidden-state cache is shared across all per-λ trainings.

Each saved gating network is loadable as `warm_start_path` in es_train.py.
"""

import copy
import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import HfArgumentParser
from trl import set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from es_architecture import GatingNetwork, SimpleGatingNetwork, load_shared_experts
from es_utils import load_main_tokenizer, save_gating_network


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:    str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    gating_dataset_csv: str       = './results/optimal/optimal_beaver_1205/gating_dataset.csv'
    gating_hidden_size: int       = 256
    fixed_alpha:        float     = 1.0    # 1.0=softmax (recommended for supervised training)
    gating_type:        str       = 'simple'  # 'simple' = SimpleGatingNetwork, 'per_layer' = GatingNetwork

    # Training (applied independently to each of the 11 gating networks)
    num_epochs:         int       = 20
    batch_size:         int       = 64
    learning_rate:      float     = 1e-3
    weight_decay:       float     = 1e-4
    max_prompt_len:     int       = 256
    seed:               int       = 8888

    # I/O
    output_dir:         str       = './models/gating_opt/'
    run_name:           str       = 'opt_pretrain'
    gpu_id:             int       = -1
    log_steps:          int       = 20
    wandb_name:         str       = ''


# ---------------------------------------------------------------------------
# Tokenize + extract hidden states for all prompts  (run once)
# ---------------------------------------------------------------------------

def build_hidden_cache(
    prompts: List[str],
    expert_models: list,
    tokenizer,
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Return (N, num_layers, H) float32 CPU tensor of LAST-TOKEN post-attn states.

    Captures POST-ATTENTION (pre-FFN) hidden states via forward hooks on each
    transformer layer's self_attn output, then keeps only the last real token
    per layer. This matches exactly what MoEForCausalLM feeds to the
    GatingNetwork at inference time (see _moe_layers: gating is called on
    `residual + attn_out`, and GatingNetwork.forward takes hidden[:, -1, :]).
    """
    enc = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt',
    )
    input_ids      = enc['input_ids']       # (N, L)
    attention_mask = enc['attention_mask']  # (N, L)
    N = len(prompts)

    all_hidden = []  # list of (B, num_layers, H)

    for start in range(0, N, batch_size):
        ids  = input_ids[start:start + batch_size].to(device)
        mask = attention_mask[start:start + batch_size].to(device)

        # Use only expert_models[0] — at inference MoEForCausalLM._moe_layers calls
        # experts[0].model.layers[l].self_attn for attention at every layer.
        m = expert_models[0]
        post_attn_states = {}  # layer_idx → (B, L, H)

        def make_hook(layer_idx, layer_module):
            def hook(module, inp, output):
                residual = layer_module._hook_residual
                attn_out = output[0]
                post_attn_states[layer_idx] = (residual + attn_out).detach().float().cpu()
            return hook

        def make_pre_hook(layer_module):
            def pre_hook(module, inp):
                layer_module._hook_residual = inp[0].detach()
            return pre_hook

        handles = []
        for l, layer in enumerate(m.model.layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(layer)))
            handles.append(layer.self_attn.register_forward_hook(make_hook(l, layer)))

        with torch.no_grad():
            m(input_ids=ids, attention_mask=mask)

        for h in handles:
            h.remove()

        # Stack → (B, num_layers, L, H), then take the last real token per layer
        # to match GatingNetwork.forward which does hidden_states[:, -1, :].
        num_layers = len(m.model.layers)
        layer_hidden = torch.stack(
            [post_attn_states[l] for l in range(num_layers)], dim=1
        )  # (B, num_layers, L, H)
        last_idx = (mask.sum(dim=1) - 1).clamp(min=0).cpu()                 # (B,)
        idx      = last_idx.view(-1, 1, 1, 1).expand(-1, num_layers, 1, layer_hidden.shape[-1])
        last     = layer_hidden.gather(2, idx).squeeze(2)  # (B, num_layers, H)
        all_hidden.append(last)

        if (start // batch_size + 1) % 5 == 0 or start + batch_size >= N:
            print(f'  hidden cache: {min(start + batch_size, N)}/{N}', flush=True)

    return torch.cat(all_hidden, dim=0)  # (N, num_layers, H)  cpu float32


def build_last_token_cache(
    prompts: List[str],
    expert_model,          # expert_models[0] — embedding is shared/untrained
    tokenizer,
    max_length: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Return (N, H) float32 CPU tensor of last-token hidden states from the final layer.

    Uses expert_models[0].model forward pass to get post-norm last hidden state.
    The last non-padding token is taken as the prompt representation,
    matching what SimpleMoEForCausalLM._last_token_hidden does at inference.
    """
    enc = tokenizer(prompts, max_length=max_length, truncation=True,
                    padding=True, return_tensors='pt')
    input_ids      = enc['input_ids']       # (N, L)
    attention_mask = enc['attention_mask']  # (N, L)
    N = len(prompts)
    all_last = []

    for start in range(0, N, batch_size):
        ids  = input_ids[start:start + batch_size].to(device)
        mask = attention_mask[start:start + batch_size].to(device)

        with torch.no_grad():
            out = expert_model.model(input_ids=ids, attention_mask=mask)
        # last_hidden_state is post-norm (matches SimpleMoEForCausalLM which applies expert.model.norm)
        final_h = out.last_hidden_state.float().cpu()   # (B, L, H)

        # With left-padding, last real token is always at the final position.
        last_token = final_h[:, -1, :]                  # (B, H)
        all_last.append(last_token)

        if (start // batch_size + 1) % 5 == 0 or start + batch_size >= N:
            print(f'  last-token cache: {min(start + batch_size, N)}/{N}', flush=True)

    return torch.cat(all_last, dim=0)   # (N, H)  cpu float32


# ---------------------------------------------------------------------------
# Per-preference training
# ---------------------------------------------------------------------------

def train_one_gating(
    hidden_cache:   torch.Tensor,   # (N, num_layers, H) or (N, H) cpu
    targets:        torch.Tensor,   # (N, num_experts) cpu — for this specific preference
    lm_hidden_size: int,
    num_experts:    int,
    num_layers:     int,
    args:           ScriptArguments,
    device:         str,
    pref_label:     str,
) -> GatingNetwork:
    """Train a fresh gating network for one preference and return it (on CPU).

    Supports two modes via args.gating_type:
    - 'simple'    : SimpleGatingNetwork, hidden_cache is (N, H) last-token states.
    - 'per_layer' : GatingNetwork, hidden_cache is (N, num_layers, H), flattened.
    """
    if args.gating_type == 'simple':
        gating_net = SimpleGatingNetwork(
            lm_hidden_size=lm_hidden_size,
            num_experts=num_experts,
            hidden_size=args.gating_hidden_size,
            fixed_alpha=args.fixed_alpha,
        ).to(device)
        flat_h = hidden_cache                  # (N, H) — already 2D
        flat_t = targets                       # (N, num_experts)
    else:
        gating_net = GatingNetwork(
            lm_hidden_size=lm_hidden_size,
            num_experts=num_experts,
            hidden_size=args.gating_hidden_size,
            num_layers=num_layers,
            fixed_alpha=args.fixed_alpha,
        ).to(device)
        N, L, H = hidden_cache.shape
        flat_h  = hidden_cache.view(N * L, H)
        flat_t  = targets.repeat_interleave(L, dim=0)

    ds     = TensorDataset(flat_h, flat_t)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(
        gating_net.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.num_epochs * len(loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.learning_rate * 0.01)

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        gating_net.train()
        epoch_loss = 0.0
        for h_batch, t_batch in loader:
            h_batch = h_batch.to(device)
            t_batch = t_batch.to(device)

            predicted = gating_net(h_batch)             # (B, num_experts)
            # KL(target || predicted): target is the "true" distribution
            # clamp targets away from 0 to avoid log(0); add small eps
            t_safe = t_batch.clamp(min=1e-6)
            t_safe = t_safe / t_safe.sum(dim=-1, keepdim=True)  # re-normalise after clamp
            loss = F.kl_div(
                predicted.float().log(),   # log Q (predicted)
                t_safe,                    # P (target)
                reduction='batchmean',
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gating_net.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % args.log_steps == 0:
                lr_now     = scheduler.get_last_lr()[0]
                mean_coefs = predicted.float().detach().mean(0).tolist()
                coef_str   = ' '.join(f'w{k}={v:.3f}' for k, v in enumerate(mean_coefs))
                print(f'  [{pref_label} ep{epoch} s{global_step}] '
                      f'loss={loss.item():.5f} lr={lr_now:.2e}  {coef_str}', flush=True)

        avg_loss = epoch_loss / len(loader)
        print(f'  [{pref_label}] epoch {epoch}/{args.num_epochs}  '
              f'avg_loss={avg_loss:.5f}', flush=True)

    return gating_net.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser      = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

output_dir = os.path.join(script_args.output_dir, script_args.run_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(script_args.seed)
np.random.seed(script_args.seed)

if 'RANK' in os.environ:
    torch.distributed.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=600))
accelerator = Accelerator()
is_main     = accelerator.is_main_process
gpu_id      = (script_args.gpu_id if script_args.gpu_id >= 0
               else accelerator.local_process_index)
device      = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_experts    = len(reward_names)

if is_main and script_args.wandb_name:
    import wandb
    wandb.init(project='momoe', name=script_args.wandb_name)

# ── Load tokenizer + expert models (mirrors evolutionary_soups.py exactly) ──
sft_tokenizer = load_main_tokenizer(script_args.expert_model_paths[0])
sft_tokenizer.padding_side = 'left'

expert_models = load_shared_experts(
    script_args.base_model_name, script_args.expert_model_paths, device,
    tokenizer=sft_tokenizer)

lm_hidden_size = expert_models[0].config.hidden_size
num_layers     = len(expert_models[0].model.layers)
print(f'lm_hidden_size={lm_hidden_size}, num_layers={num_layers}')

# ── Load dataset ──
df       = pd.read_csv(script_args.gating_dataset_csv)
opt_cols = [f'optimal_w{k}' for k in range(n_experts)]
pref_cols = [f'pref_{r}' for r in reward_names]
for c in opt_cols:
    df[c] = df[c].astype(float)

# Collect unique (preference tuple) → group
# Each group shares the same prompt set; prompts are consistent across groups
unique_prefs: List[Tuple] = [
    tuple(row) for row in
    df[pref_cols].drop_duplicates().sort_values(pref_cols[0]).values.tolist()
]
print(f'\nFound {len(unique_prefs)} preference points, {df["prompt_idx"].nunique()} prompts.')

# Canonical prompt order (by first occurrence in dataset)
prompt_order = df.drop_duplicates('prompt_idx').sort_values('prompt_idx')
all_prompts  = prompt_order['prompt_text'].tolist()
prompt_idx_to_pos = {idx: pos for pos, idx in enumerate(prompt_order['prompt_idx'].tolist())}
N = len(all_prompts)

# ── Build hidden state cache (one forward pass via expert[0]) ──
print(f'\nBuilding hidden state cache for {N} prompts  (gating_type={script_args.gating_type}) …')
if script_args.gating_type == 'simple':
    hidden_cache = build_last_token_cache(
        prompts=all_prompts,
        expert_model=expert_models[0],
        tokenizer=sft_tokenizer,
        max_length=script_args.max_prompt_len,
        batch_size=script_args.batch_size,
        device=device,
    )  # (N, H)
    print(f'Hidden cache shape: {hidden_cache.shape}  (N, H)')
else:
    hidden_cache = build_hidden_cache(
        prompts=all_prompts,
        expert_models=expert_models,
        tokenizer=sft_tokenizer,
        max_length=script_args.max_prompt_len,
        batch_size=script_args.batch_size,
        device=device,
    )  # (N, num_layers, H)
    print(f'Hidden cache shape: {hidden_cache.shape}  (N, num_layers, H)')

# Free expert models from GPU after caching to save memory for training
del expert_models
torch.cuda.empty_cache()

# ── Train one GatingNetwork per preference ──
all_meta = []

for pref_idx, pref_tuple in enumerate(unique_prefs):
    pref_label = 'pref_' + '_'.join(f'{v:.1f}' for v in pref_tuple)
    print(f'\n[{pref_idx+1}/{len(unique_prefs)}] Training gating for {pref_label}')

    # Filter rows for this preference
    mask_pref = np.ones(len(df), dtype=bool)
    for col, val in zip(pref_cols, pref_tuple):
        mask_pref &= np.isclose(df[col].values, val)
    sub = df[mask_pref]

    # Build target tensor aligned to canonical prompt order
    targets = torch.zeros(N, n_experts, dtype=torch.float32)
    for _, row in sub.iterrows():
        pos = prompt_idx_to_pos.get(row['prompt_idx'])
        if pos is not None:
            targets[pos] = torch.tensor([row[c] for c in opt_cols], dtype=torch.float32)

    # Train
    net = train_one_gating(
        hidden_cache=hidden_cache,
        targets=targets,
        lm_hidden_size=lm_hidden_size,
        num_experts=n_experts,
        num_layers=num_layers,
        args=script_args,
        device=device,
        pref_label=pref_label,
    )

    # Save
    if is_main:
        save_dir = os.path.join(output_dir, pref_label)
        save_gating_network(net, save_dir)
        pref_meta = {
            'pref_idx':   pref_idx,
            'preference': {col: val for col, val in zip(pref_cols, pref_tuple)},
            'save_dir':   save_dir,
        }
        all_meta.append(pref_meta)
        print(f'  Saved → {save_dir}')

        if script_args.wandb_name:
            import wandb
            wandb.log({'pref_idx': pref_idx}, step=pref_idx)

# ── Write index of all saved networks ──
if is_main:
    meta = {
        'reward_names':       reward_names,
        'lm_hidden_size':     lm_hidden_size,
        'num_layers':         num_layers,
        'gating_hidden_size': script_args.gating_hidden_size,
        'fixed_alpha':        script_args.fixed_alpha,
        'num_epochs':         script_args.num_epochs,
        'dataset_csv':        script_args.gating_dataset_csv,
        'num_preferences':    len(unique_prefs),
        'networks':           all_meta,
    }
    with open(os.path.join(output_dir, 'pretrain_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nAll {len(unique_prefs)} gating networks saved under {output_dir}')
    print('Index written to pretrain_meta.json')
