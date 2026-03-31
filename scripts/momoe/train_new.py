"""Step 3: Train GatingNetwork on the supervised dataset from build_dataset_mg.py.

loss_mode='weight' (default):
    MSE between predicted weights and utility-optimal weights.
    Fast — no reward data needed beyond gating_dataset.csv.

loss_mode='reward':
    MSE between predicted reward and optimal reward.
    pred_r = pred_weights @ B  where B is a per-prompt linear reward model
    fit from collected_rewards.csv via least squares, making pred_r fully
    differentiable w.r.t. pred_weights.
    opt_r is looked up directly from collected_rewards.csv at the optimal
    grid-point weights (no interpolation needed).
    Requires --rewards_csv.
"""
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          HfArgumentParser)
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_config, load_main_tokenizer, print_trainable_parameters
from new_architecture import (GatingNetwork, GatingDataset, get_prompt_hidden,
                               get_prompt_hidden_from_reward_models, REWARD_PATHS)
from new_utils import load_base_model, save_gating_network, build_reward_maps


@dataclass
class ScriptArguments:
    sft_model_name:       str       = './models/sft/model/'
    expert_model_paths:   List[str] = field(default_factory=list)
    rewards_csv:          str       = ''       # required for loss_mode='reward'|'chebyshev'
    dataset_csv:          str       = './data/new/new_assistant/gating_dataset.csv'
    reward_names:         str       = 'harmless,helpful'
    block_mode:           str       = 'uniform'    # 'uniform' | 'custom'
    loss_mode:            str       = 'weight'     # 'weight' | 'reward' | 'chebyshev'
    use_reward_features:  bool      = False
    use_lora:             bool      = True    # True → expert paths are LoRA adapters
                                              # False → expert paths are full models on disk
    lr:                   float     = 1e-4
    weight_decay:         float     = 5e-4    # increased from 1e-4 for better regularisation
    dropout:              float     = 0.2     # increased from 0.1 for better regularisation
    hidden_dim:           int       = 256
    epochs:               int       = 100
    batch_size:           int       = 128
    val_frac:             float     = 0.15    # fraction of prompts held out for validation
    patience:             int       = 20      # early stopping patience (epochs)
    grad_clip:            float     = 1.0     # max gradient norm (0 = disabled)
    log_with:             str       = 'none'
    save_directory:       str       = './models/new/'
    wandb_name:           str       = 'new_assistant'


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.loss_mode in ('reward', 'chebyshev') and not script_args.rewards_csv:
    raise ValueError('--rewards_csv is required when --loss_mode reward|chebyshev')

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
gpu_id       = accelerator.local_process_index
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards  = len(reward_names)

tokenizer = load_main_tokenizer(script_args.sft_model_name)

# ── Prompt feature models (Solution 1: reward models OR expert LLMs) ─────────
feature_models      = []   # models used to encode prompts
feature_tokenizers  = []   # tokenizers matching feature_models (None for LLM path)
use_reward_features = script_args.use_reward_features

if use_reward_features:
    # Load the reward models as prompt encoders (frozen)
    print('Loading reward models for prompt feature extraction ...')
    for name in reward_names:
        path = REWARD_PATHS[name]
        m    = AutoModelForSequenceClassification.from_pretrained(
            path, torch_dtype=torch.bfloat16).to(f'cuda:{gpu_id}')
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:          # GPT-2 tokenisers lack a pad token
            tok.pad_token = tok.eos_token
            m.config.pad_token_id = tok.eos_token_id  # GPT-2 model config also needs this
        feature_models.append(m)
        feature_tokenizers.append(tok)
    # lm_hidden_size = SUM of each reward model's hidden size (we concatenate, not average)
    lm_hidden_size = 0
    with torch.no_grad():
        for _m, _tok in zip(feature_models, feature_tokenizers):
            _d   = _tok('hello', return_tensors='pt').to(f'cuda:{gpu_id}')
            _out = _m(**_d, output_hidden_states=True)
            if (hasattr(_out, 'encoder_last_hidden_state')
                    and _out.encoder_last_hidden_state is not None):
                lm_hidden_size += _out.encoder_last_hidden_state.shape[-1]
            else:
                lm_hidden_size += _out.hidden_states[-1].shape[-1]
else:
    # Original path: load frozen PPO expert LLMs as prompt encoders.
    # use_lora=True: expert paths are LoRA adapters → load base + merge.
    # use_lora=False: expert paths are full pre-merged models on disk.
    for path in script_args.expert_model_paths:
        if script_args.use_lora:
            from peft import PeftModel
            m = load_base_model(script_args.sft_model_name, target_device=f'cuda:{gpu_id}')
            m = PeftModel.from_pretrained(m, path)
            m = m.merge_and_unload()
        else:
            m = load_base_model(path, target_device=f'cuda:{gpu_id}')
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        feature_models.append(m)
    with torch.no_grad():
        dummy     = tokenizer('hello', return_tensors='pt').to(f'cuda:{gpu_id}')
        dummy_out = feature_models[0](**dummy, output_hidden_states=True)
        lm_hidden_size = dummy_out.hidden_states[-1].shape[-1]
print(f'lm_hidden_size = {lm_hidden_size}  '
      f'({"reward models" if use_reward_features else "expert LLMs"})')

# ── Precompute reward maps (loss_mode='reward' or 'chebyshev') ────────────────
reward_basis_map = opt_r_map = r_star_map = None

if script_args.loss_mode in ('reward', 'chebyshev'):
    print(f'Precomputing reward maps for loss_mode={script_args.loss_mode} ...')
    rewards_df   = pd.read_csv(script_args.rewards_csv)
    dataset_df_r = pd.read_csv(script_args.dataset_csv) if script_args.loss_mode == 'reward' else None
    reward_basis_map, opt_r_map, r_star_map = build_reward_maps(
        rewards_df, dataset_df_r, reward_names,
        script_args.block_mode, script_args.loss_mode)
    print(f'  Done. {len(reward_basis_map)} prompts.')

# ── Dataset and loader ────────────────────────────────────────────────────────
dataset_df = pd.read_csv(script_args.dataset_csv)

# Prompt-level train/val split — held-out prompts test generalisation to new inputs.
# Splitting at the prompt level (not sample level) prevents the model from seeing
# a prompt in training and then evaluating on a different preference for that same prompt.
all_prompt_ids = dataset_df['prompt_idx'].unique()
rng = np.random.default_rng(42)
rng.shuffle(all_prompt_ids)
n_val         = max(1, int(len(all_prompt_ids) * script_args.val_frac))
val_ids       = set(all_prompt_ids[:n_val].tolist())
train_ids_set = set(all_prompt_ids[n_val:].tolist())

train_df = dataset_df[dataset_df['prompt_idx'].isin(train_ids_set)].reset_index(drop=True)
val_df   = dataset_df[dataset_df['prompt_idx'].isin(val_ids)].reset_index(drop=True)
print(f'Prompts  — train: {len(train_ids_set)}, val: {len(val_ids)}')

train_dataset = GatingDataset(train_df, reward_names, tokenizer,
                               block_mode=script_args.block_mode,
                               reward_basis_map=reward_basis_map,
                               opt_r_map=opt_r_map,
                               r_star_map=r_star_map)
val_dataset   = GatingDataset(val_df, reward_names, tokenizer,
                               block_mode=script_args.block_mode,
                               reward_basis_map=reward_basis_map,
                               opt_r_map=opt_r_map,
                               r_star_map=r_star_map)
print(f'Samples  — train: {len(train_dataset)}, val: {len(val_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=script_args.batch_size,
                          shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=script_args.batch_size,
                          shuffle=False, drop_last=False)

# ── Gating network ────────────────────────────────────────────────────────────
gating_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=num_rewards,
    hidden_dim=script_args.hidden_dim,
    block_mode=script_args.block_mode,
    dropout=script_args.dropout,          # Solution 4
)
# Solution 4: AdamW with weight decay instead of plain Adam
optimizer = torch.optim.AdamW(gating_net.parameters(),
                               lr=script_args.lr,
                               weight_decay=script_args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=script_args.epochs, eta_min=script_args.lr * 0.01)
gating_net, optimizer, train_loader, val_loader = accelerator.prepare(
    gating_net, optimizer, train_loader, val_loader)

wandb_run = None
if accelerator.is_main_process and script_args.log_with == 'wandb':
    import wandb
    wandb.login(key="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")
    wandb_run = wandb.init(project='new_gating', name=script_args.wandb_name)

print_trainable_parameters(gating_net)
print(f'loss_mode = {script_args.loss_mode}')

# ── Training loop ─────────────────────────────────────────────────────────────

def compute_loss(batch, gating_net, is_val=False):
    """Forward pass + loss for a single batch. Shared by train and val loops."""
    input_ids       = batch['input_ids'].to(f'cuda:{gpu_id}')
    attention_mask  = batch['attention_mask'].to(f'cuda:{gpu_id}')
    preference      = batch['preference'].to(f'cuda:{gpu_id}')
    optimal_weights = batch['optimal_weights'].to(f'cuda:{gpu_id}')

    if use_reward_features:
        prompt_texts  = list(batch['prompt_text'])
        prompt_hidden = get_prompt_hidden_from_reward_models(
            feature_models, feature_tokenizers, prompt_texts, f'cuda:{gpu_id}')
    else:
        prompt_hidden = get_prompt_hidden(feature_models, input_ids, attention_mask)

    pred_weights = gating_net(prompt_hidden, preference)

    if is_val or script_args.loss_mode == 'weight':
        return F.mse_loss(pred_weights, optimal_weights)

    elif script_args.loss_mode == 'reward':
        reward_basis = batch['reward_basis'].to(f'cuda:{gpu_id}')
        opt_r        = batch['opt_r'].to(f'cuda:{gpu_id}')
        pred_r = torch.einsum('be,ber->br', pred_weights, reward_basis)
        return F.mse_loss(pred_r, opt_r)

    else:  # 'chebyshev'
        reward_basis = batch['reward_basis'].to(f'cuda:{gpu_id}')
        r_star       = batch['r_star'].to(f'cuda:{gpu_id}')
        pred_r = torch.einsum('be,ber->br', pred_weights, reward_basis)
        gaps   = preference * (pred_r - r_star).abs()
        return gaps.max(dim=1).values.mean()


global_step  = 0
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(script_args.epochs):
    # ── Train ──────────────────────────────────────────────────────────────────
    gating_net.train()
    epoch_losses = []
    progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{script_args.epochs}')

    for batch in progress:
        loss = compute_loss(batch, gating_net)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if script_args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(gating_net.parameters(), script_args.grad_clip)
        optimizer.step()

        loss_val = loss.item()
        epoch_losses.append(loss_val)
        global_step += 1
        progress.set_postfix(loss=f'{loss_val:.4f}')

        if wandb_run is not None:
            wandb_run.log({'train/loss': loss_val, 'train/step': global_step,
                           'train/epoch': epoch+1})

    scheduler.step()
    mean_train_loss = np.mean(epoch_losses)

    # ── Validation ─────────────────────────────────────────────────────────────
    gating_net.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            val_losses.append(compute_loss(batch, gating_net, is_val=True).item())
    mean_val_loss = np.mean(val_losses)

    print(f'Epoch {epoch+1}: train_loss={mean_train_loss:.4f}  val_loss={mean_val_loss:.4f}  '
          f'lr={scheduler.get_last_lr()[0]:.2e}')

    if wandb_run is not None:
        wandb_run.log({'val/loss': mean_val_loss, 'val/epoch': epoch+1})

    # ── Checkpoint + early stopping ────────────────────────────────────────────
    if accelerator.is_main_process:
        # Always save periodic checkpoint
        save_path = os.path.join(output_dir, f'epoch_{epoch+1}_step_{global_step}')
        save_gating_network(accelerator.unwrap_model(gating_net), save_path)
        print(f'  Saved checkpoint → {save_path}')

        # Save best-val checkpoint (overwrites previous best)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            epochs_no_improve = 0
            best_path = os.path.join(output_dir, 'best_val')
            save_gating_network(accelerator.unwrap_model(gating_net), best_path)
            print(f'  New best val_loss={best_val_loss:.4f} → saved to {best_path}')
        else:
            epochs_no_improve += 1
            print(f'  No improvement for {epochs_no_improve}/{script_args.patience} epochs')
            if epochs_no_improve >= script_args.patience:
                print(f'Early stopping at epoch {epoch+1}.')
                break

if wandb_run is not None:
    wandb_run.finish()
