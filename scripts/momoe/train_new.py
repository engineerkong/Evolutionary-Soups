"""Step 3: Train GatingNetwork on the supervised dataset from build_dataset.py.

The network learns f(prompt_hidden, preference) -> merging weights.
prompt_hidden is the mean-pooled last hidden state averaged over both expert models.
Loss is reward-space MSE: predicted_reward vs chebyshev_optimal_reward.
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
from transformers import HfArgumentParser
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_config, load_main_tokenizer, print_trainable_parameters
from new_architecture import GatingNetwork, chebyshev_optimal_weights, GatingDataset, get_prompt_hidden, SAMPLE_T_VALUES
from new_utils import load_base_model, save_gating_network


@dataclass
class ScriptArguments:
    sft_model_name: str = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    dataset_csv: str = './data/new/new_assistant/gating_dataset.csv'
    rewards_csv: str = './data/new/new_assistant/collected_rewards.csv'
    reward_names: str = 'harmless,helpful'
    log_with: str = 'none'
    save_directory: str = './models/new/'
    wandb_name: str = 'new_assistant'
    hidden_dim: int = 256
    lr: float = 1e-4
    epochs: int = 1000
    batch_size: int = 128


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
gpu_id = accelerator.local_process_index
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_experts = len(reward_names)

tokenizer = load_main_tokenizer(script_args.sft_model_name)

# Load frozen expert models for prompt encoding
expert_models = []
for path in script_args.expert_model_paths:
    m = load_base_model(path, target_device=f'cuda:{gpu_id}')
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    expert_models.append(m)

# Infer hidden size from first expert
with torch.no_grad():
    dummy = tokenizer('hello', return_tensors='pt').to(f'cuda:{gpu_id}')
    dummy_out = expert_models[0](**dummy, output_hidden_states=True)
    lm_hidden_size = dummy_out.hidden_states[-1].shape[-1]
print(f'Detected lm_hidden_size = {lm_hidden_size}')

# Load datasets
dataset_df = pd.read_csv(script_args.dataset_csv)
rewards_df = pd.read_csv(script_args.rewards_csv)
train_dataset = GatingDataset(dataset_df, rewards_df, reward_names, tokenizer)
print(f'Training dataset size: {len(train_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=script_args.batch_size,
                          shuffle=True, drop_last=True)

# Gating network
gating_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=num_experts,
    hidden_dim=script_args.hidden_dim,
)
optimizer = torch.optim.Adam(gating_net.parameters(), lr=script_args.lr)
gating_net, optimizer, train_loader = accelerator.prepare(gating_net, optimizer, train_loader)

wandb_run = None
if accelerator.is_main_process and script_args.log_with == 'wandb':
    import wandb
    wandb.login(key="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")
    wandb_run = wandb.init(project='new_gating', name=script_args.wandb_name)

print_trainable_parameters(gating_net)

global_step = 0
for epoch in range(script_args.epochs):
    gating_net.train()
    epoch_losses = []
    progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{script_args.epochs}')

    for batch in progress:
        input_ids      = batch['input_ids'].to(f'cuda:{gpu_id}')
        attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
        preference     = batch['preference'].to(f'cuda:{gpu_id}')
        optimal_weights= batch['optimal_weights'].to(f'cuda:{gpu_id}')
        reward_matrix  = batch['reward_matrix'].to(f'cuda:{gpu_id}')
        # reward_matrix: (B, num_t, num_rewards)

        # Frozen prompt representation
        prompt_hidden = get_prompt_hidden(expert_models, input_ids, attention_mask)

        # Predicted weights
        pred_weights = gating_net(prompt_hidden, preference)  # (B, num_experts)

        # Reward-space MSE loss
        # predicted reward = interpolate reward_matrix at pred_weights[:, 0]
        t_vals = torch.tensor(SAMPLE_T_VALUES, dtype=torch.float32, device=f'cuda:{gpu_id}')

        # Interpolate for predicted t and optimal t
        def interpolate_reward(weights_col0, reward_mat):
            """Linear interpolation of reward at given t values.
            weights_col0: (B,)  the t values to interpolate at
            reward_mat:   (B, num_t, num_rewards)
            returns:      (B, num_rewards)
            """
            B = weights_col0.shape[0]
            num_t = len(SAMPLE_T_VALUES)
            # Find surrounding indices
            t_vals_exp = t_vals.unsqueeze(0).expand(B, -1)  # (B, num_t)
            t_query = weights_col0.unsqueeze(1)              # (B, 1)
            # idx of the first t_val >= t_query
            idx_high = (t_vals_exp >= t_query).float().argmax(dim=1).clamp(1, num_t-1)  # (B,)
            idx_low  = (idx_high - 1).clamp(0, num_t-2)

            t_low  = t_vals[idx_low]                          # (B,)
            t_high = t_vals[idx_high]                         # (B,)
            span   = (t_high - t_low).clamp(min=1e-8)
            alpha  = ((t_query.squeeze(1) - t_low) / span).clamp(0, 1)  # (B,)

            r_low  = reward_mat[torch.arange(B), idx_low]    # (B, num_rewards)
            r_high = reward_mat[torch.arange(B), idx_high]   # (B, num_rewards)
            return r_low + alpha.unsqueeze(1) * (r_high - r_low)

        pred_t   = pred_weights[:, 0]                  # (B,)
        opt_t    = optimal_weights[:, 0]               # (B,)
        pred_r   = interpolate_reward(pred_t, reward_matrix)   # (B, num_rewards)
        opt_r    = interpolate_reward(opt_t,  reward_matrix)   # (B, num_rewards)

        loss = F.mse_loss(pred_r, opt_r)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        loss_val = loss.item()
        epoch_losses.append(loss_val)
        global_step += 1
        progress.set_postfix(loss=f'{loss_val:.4f}')

        if wandb_run is not None:
            wandb_run.log({'train/loss': loss_val, 'train/step': global_step,
                           'train/epoch': epoch+1})

    mean_loss = np.mean(epoch_losses)
    print(f'Epoch {epoch+1}: mean_loss={mean_loss:.4f}')

    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f'epoch_{epoch+1}_step_{global_step}')
        save_gating_network(accelerator.unwrap_model(gating_net), save_path)
        print(f'Saved checkpoint to {save_path}')

if wandb_run is not None:
    wandb_run.finish()