import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import get_clean_data


# ---------------------------------------------------------------------------
# Chebyshev optimal weight solver
# ---------------------------------------------------------------------------

def chebyshev_optimal_weights(reward_matrix, preference, sample_weights):
    """Find the optimal merging weight from sampled rewards using Chebyshev scalarization.

    Args:
        reward_matrix : np.ndarray shape (num_samples, num_rewards)
                        reward_matrix[i] = reward vector for merging weight sample_weights[i]
        preference    : list/array of length num_rewards, sums to 1
        sample_weights: list of floats, the t values used (e.g. [0.0,0.2,0.4,0.6,0.8,1.0])

    Returns:
        optimal_t : float, the merging weight t* that maximises Chebyshev utility
        optimal_weights : list of length num_experts = [t*, 1-t*]
    """
    preference = np.array(preference, dtype=np.float64)
    reward_matrix = np.array(reward_matrix, dtype=np.float64)

    # Per-prompt ideal point: best observed reward on each dimension
    r_star = reward_matrix.max(axis=0)          # shape (num_rewards,)

    # Chebyshev utility for each sample point
    utilities = []
    for rewards in reward_matrix:
        weighted_gaps = preference * np.abs(rewards - r_star)
        utilities.append(-np.max(weighted_gaps))

    best_idx = int(np.argmax(utilities))
    optimal_t = float(sample_weights[best_idx])
    return optimal_t, [optimal_t, 1.0 - optimal_t]


# ---------------------------------------------------------------------------
# Gating Network
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Maps (prompt_hidden, preference) -> merging weights over experts.

    prompt_hidden : mean-pooled hidden states averaged over the two expert models,
                    shape (batch, lm_hidden_size)
    preference    : shape (batch, num_experts)
    output        : shape (batch, num_experts), sums to 1 via softmax
    """

    def __init__(self, lm_hidden_size=4096, num_experts=2, hidden_dim=256):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.pref_proj = nn.Sequential(
            nn.Linear(num_experts, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim // 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def forward(self, prompt_hidden, preference):
        p = self.prompt_proj(prompt_hidden)
        r = self.pref_proj(preference)
        x = torch.cat([p, r], dim=-1)
        logits = self.fusion(x)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Gating Network Trainer
# ---------------------------------------------------------------------------

class GatingNetworkTrainer:
    """Supervised trainer for GatingNetwork using Chebyshev ground-truth weights."""

    def __init__(self, gating_net, expert_models, reward_models, instructions,
                 num_rewards=2, num_pref_samples=11, lr=1e-4, device='cuda'):
        self.gating_net = gating_net.to(device)
        self.expert_models = expert_models      # list of frozen LLMs
        self.reward_models = reward_models
        self.instructions = instructions
        self.num_rewards = num_rewards
        self.num_pref_samples = num_pref_samples
        self.device = device
        self.optimizer = torch.optim.Adam(gating_net.parameters(), lr=lr)

        for m in self.expert_models:
            for p in m.parameters():
                p.requires_grad = False

    def _get_prompt_hidden(self, input_ids, attention_mask):
        """Average hidden states from all expert models as prompt representation."""
        all_hidden = []
        with torch.no_grad():
            for model in self.expert_models:
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
                # mean pool last hidden state over sequence length
                h = out.hidden_states[-1]  # (batch, seq, hidden)
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1)
                all_hidden.append(pooled)
        return torch.stack(all_hidden, dim=0).mean(dim=0)  # (batch, hidden)

    def _score_responses(self, responses, prompts):
        """Return reward vectors shape (num_rewards, batch)."""
        prompts_clean, responses_clean = get_clean_data(responses, prompts)
        pairs = [(self.instructions.get_input(t), self.instructions.get_response(t))
                 for t in responses_clean]
        if hasattr(self.instructions, 'get_post'):
            return self.reward_models.get_reward_model_scores(
                pairs, self.instructions.get_post, normalize_rewards=False)
        return self.reward_models.get_reward_model_scores(pairs, normalize_rewards=False)

    def train_step(self, batch, tokenizer, precomputed_rewards, preferences):
        """One supervised training step.

        precomputed_rewards: dict mapping t_value -> list of reward vectors per prompt
                             shape per entry: (batch_size, num_rewards)
        preferences        : list of preference vectors, len = num_pref_samples
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        batch_size = input_ids.shape[0]

        # Get prompt representation (frozen)
        prompt_hidden = self._get_prompt_hidden(input_ids, attention_mask)  # (B, H)

        sample_weights = sorted(precomputed_rewards.keys())
        # reward_tensor: (num_t_samples, batch_size, num_rewards)
        reward_tensor = np.stack(
            [precomputed_rewards[t] for t in sample_weights], axis=0
        )

        total_loss = torch.tensor(0.0, device=self.device)
        num_pairs = 0

        self.optimizer.zero_grad()

        for preference in preferences:
            pref_tensor = torch.tensor(preference, dtype=torch.float32,
                                       device=self.device).unsqueeze(0).expand(batch_size, -1)

            # Ground truth: Chebyshev optimal t per prompt
            gt_weights = []
            for b in range(batch_size):
                reward_mat = reward_tensor[:, b, :]   # (num_t, num_rewards)
                opt_t, opt_w = chebyshev_optimal_weights(reward_mat, preference, sample_weights)
                gt_weights.append(opt_w)
            gt_weights = torch.tensor(gt_weights, dtype=torch.float32, device=self.device)  # (B, 2)

            # Predicted weights
            pred_weights = self.gating_net(prompt_hidden, pref_tensor)  # (B, 2)

            # Reward-space loss: compare predicted vs optimal reward vector
            # predicted_reward = sum_i w_i * r_i  for each reward dimension
            # reward_tensor at optimal_t vs predicted_t
            pred_t = pred_weights[:, 0].detach().cpu().numpy()
            opt_t_vals = gt_weights[:, 0].detach().cpu().numpy()

            pred_rewards_list = []
            opt_rewards_list = []
            for b in range(batch_size):
                # Interpolate reward at predicted t
                r_pred = np.interp(pred_t[b], sample_weights,
                                   [reward_tensor[i, b, :] for i in range(len(sample_weights))])
                r_opt = np.interp(opt_t_vals[b], sample_weights,
                                  [reward_tensor[i, b, :] for i in range(len(sample_weights))])
                pred_rewards_list.append(r_pred)
                opt_rewards_list.append(r_opt)

            pred_r = torch.tensor(np.array(pred_rewards_list), dtype=torch.float32, device=self.device)
            opt_r  = torch.tensor(np.array(opt_rewards_list),  dtype=torch.float32, device=self.device)
            loss = F.mse_loss(pred_r, opt_r)
            total_loss = total_loss + loss
            num_pairs += 1

        (total_loss / num_pairs).backward()
        self.optimizer.step()

        return (total_loss / num_pairs).item()


# ---------------------------------------------------------------------------
# Model merging (kept from qmo_architecture for reuse in eval)
# ---------------------------------------------------------------------------

def merge_model_weights(model, expert_state_dicts, expert_weights):
    """Merge expert state dicts in-place using param.data.copy_()."""
    param_map  = {name: p   for name, p   in model.named_parameters()}
    buffer_map = {name: buf for name, buf in model.named_buffers()}

    merged_state = {}
    for i, (sd, w) in enumerate(zip(expert_state_dicts, expert_weights)):
        for key, val in sd.items():
            if key not in param_map and key not in buffer_map:
                continue
            target = param_map.get(key) or buffer_map[key]
            v = val.float()
            if i == 0:
                merged_state[key] = w * v
            else:
                merged_state[key] += w * v

    for key, merged_val in merged_state.items():
        if key in param_map:
            param_map[key].data.copy_(
                merged_val.to(dtype=param_map[key].dtype, device=param_map[key].device))
        elif key in buffer_map:
            buffer_map[key].copy_(
                merged_val.to(dtype=buffer_map[key].dtype, device=buffer_map[key].device))


# ---------------------------------------------------------------------------
# Shared constants and dataset used by both train_new and test_gating_network
# ---------------------------------------------------------------------------

SAMPLE_T_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


class GatingDataset(torch.utils.data.Dataset):
    """Each item: one (prompt_text, preference, optimal_weights, reward_matrix) tuple."""

    def __init__(self, dataset_df, rewards_df, reward_names, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reward_names = reward_names
        self.items = []

        # Build reward lookup: prompt_idx -> {t_value: reward_vector}
        reward_lookup = {}
        for _, row in rewards_df.iterrows():
            idx = int(row['prompt_idx'])
            t = float(row['t_value'])
            rv = [float(row[f'reward_{n}']) for n in reward_names]
            if idx not in reward_lookup:
                reward_lookup[idx] = {}
            reward_lookup[idx][t] = rv

        for _, row in dataset_df.iterrows():
            idx = int(row['prompt_idx'])
            if idx not in reward_lookup:
                continue
            preference = [float(row[f'pref_{n}']) for n in reward_names]
            optimal_w  = [float(row[f'optimal_w{k}']) for k in range(len(reward_names))]
            reward_matrix = np.array([reward_lookup[idx][t] for t in SAMPLE_T_VALUES])
            self.items.append({
                'prompt_text': str(row['prompt_text']),
                'preference': preference,
                'optimal_weights': optimal_w,
                'reward_matrix': reward_matrix,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item['prompt_text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids':       enc['input_ids'].squeeze(0),
            'attention_mask':  enc['attention_mask'].squeeze(0),
            'preference':      torch.tensor(item['preference'],      dtype=torch.float32),
            'optimal_weights': torch.tensor(item['optimal_weights'], dtype=torch.float32),
            'reward_matrix':   torch.tensor(item['reward_matrix'],   dtype=torch.float32),
        }


def get_prompt_hidden(expert_models, input_ids, attention_mask):
    """Frozen mean-pool hidden states averaged over all expert models."""
    all_hidden = []
    with torch.no_grad():
        for model in expert_models:
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
            h = out.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1)
            all_hidden.append(pooled)
    return torch.stack(all_hidden).mean(0)