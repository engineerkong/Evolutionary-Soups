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

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
}


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

    prompt_hidden : last-token hidden states (from expert LLMs or reward models),
                    shape (batch, lm_hidden_size)
    preference    : shape (batch, num_experts)

    uniform  output : (batch, num_experts),     sums to 1 via softmax
    custom   output : (batch, num_experts * 3), each block of num_experts sums to 1

    Architecture uses FiLM (Feature-wise Linear Modulation): the preference vector
    generates per-channel scale and shift parameters that modulate the prompt
    representation before the output head. This allows the preference to
    meaningfully condition how the prompt features are interpreted, rather than
    simply being concatenated and processed in parallel.
    """

    def __init__(self, lm_hidden_size=4096, num_experts=2, hidden_dim=256,
                 block_mode='uniform', dropout=0.1):
        super().__init__()
        self.num_experts    = num_experts
        self.block_mode     = block_mode
        self.lm_hidden_size = lm_hidden_size   # saved for checkpoint config
        self.hidden_dim     = hidden_dim
        n_out = num_experts * 3 if block_mode == 'custom' else num_experts

        # Project prompt hidden states to a compact representation
        self.prompt_proj = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM generator: preference → (gamma, beta) for each hidden unit.
        # gamma and beta modulate the prompt features channel-wise, allowing
        # the preference to redirect which prompt features are amplified.
        self.pref_expand = nn.Sequential(
            nn.Linear(num_experts, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
        )
        self.film_gen = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

        # Output head applied to FiLM-modulated prompt features
        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_out),
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, prompt_hidden, preference):
        p = self.prompt_proj(prompt_hidden)          # (B, hidden_dim)

        # FiLM modulation: preference conditions the prompt representation
        preference = torch.log(preference.clamp(min=1e-6))
        pref = self.pref_expand(preference)          # (B, hidden_dim)
        film = self.film_gen(pref)                   # (B, hidden_dim * 2)
        gamma, beta = film.chunk(2, dim=-1)          # each (B, hidden_dim)
        p = p * (1.0 + gamma) + beta                 # channel-wise scale + shift

        logits = self.output_head(p)
        if self.block_mode == 'custom':
            B = logits.shape[0]
            logits = logits.view(B, 3, self.num_experts)
            return F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1).view(B, 3 * self.num_experts)
        return F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)


# ---------------------------------------------------------------------------
# Dataset — reads gating_dataset.csv produced by build_dataset_mg.py
# ---------------------------------------------------------------------------

class GatingDataset(torch.utils.data.Dataset):
    """Each item: (prompt_text, preference, optimal_weights[, opt_r, reward_basis, r_star]).

    uniform  optimal_weights : flat list of length num_experts
    custom   optimal_weights : flat list of length num_experts * 3
                               [early_w0, ..., mid_w0, ..., late_w0, ...]

    reward_basis (optional, for loss_mode='reward'|'chebyshev'):
        np.array (n_experts_total, n_rewards) — per-prompt linear reward model
        fit via lstsq on collected_rewards.csv so that r(w) ≈ w @ reward_basis.
        pred_r = pred_weights @ reward_basis  is then differentiable.
    opt_r (optional, for loss_mode='reward'):
        np.array (n_rewards,) — reward at the optimal weights, evaluated via RBF.
    r_star (optional, for loss_mode='chebyshev'):
        np.array (n_rewards,) — per-prompt ideal point (max reward per dimension).
    """

    def __init__(self, dataset_df, reward_names, tokenizer,
                 block_mode='uniform', max_length=256,
                 reward_basis_map=None, opt_r_map=None, r_star_map=None):
        """
        reward_basis_map : dict  prompt_idx -> np.array (n_experts_total, n_rewards)
        opt_r_map        : dict  (prompt_idx, pref_tuple) -> np.array (n_rewards,)
        r_star_map       : dict  prompt_idx -> np.array (n_rewards,)
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.items      = []
        n = len(reward_names)

        for _, row in dataset_df.iterrows():
            preference = [float(row[f'pref_{name}']) for name in reward_names]
            if block_mode == 'uniform':
                optimal_w = [float(row[f'optimal_w{k}']) for k in range(n)]
            else:
                optimal_w = (
                    [float(row[f'optimal_w{k}_early']) for k in range(n)] +
                    [float(row[f'optimal_w{k}_mid'])   for k in range(n)] +
                    [float(row[f'optimal_w{k}_late'])  for k in range(n)]
                )
            pidx     = int(row['prompt_idx'])
            pref_key = tuple(round(v, 6) for v in preference)
            item = {
                'prompt_idx':      pidx,
                'prompt_text':     str(row['prompt_text']),
                'preference':      preference,
                'optimal_weights': optimal_w,
                'reward_basis':    reward_basis_map[pidx]           if reward_basis_map else None,
                'opt_r':           opt_r_map.get((pidx, pref_key)) if opt_r_map        else None,
                'r_star':          r_star_map[pidx]                 if r_star_map       else None,
            }
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc  = self.tokenizer(
            item['prompt_text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        out = {
            'prompt_idx':      torch.tensor(item['prompt_idx'],      dtype=torch.long),
            'prompt_text':     item['prompt_text'],          # raw text for reward model re-tokenisation
            'input_ids':       enc['input_ids'].squeeze(0),
            'attention_mask':  enc['attention_mask'].squeeze(0),
            'preference':      torch.tensor(item['preference'],      dtype=torch.float32),
            'optimal_weights': torch.tensor(item['optimal_weights'], dtype=torch.float32),
        }
        if item['reward_basis'] is not None:
            out['reward_basis'] = torch.tensor(item['reward_basis'], dtype=torch.float32)
        if item['opt_r'] is not None:
            out['opt_r'] = torch.tensor(item['opt_r'], dtype=torch.float32)
        if item['r_star'] is not None:
            out['r_star'] = torch.tensor(item['r_star'], dtype=torch.float32)
        return out


def get_prompt_hidden(expert_models, input_ids, attention_mask):
    """Mean-pooled hidden states averaged over all expert LLMs.

    Uses attention-mask-weighted mean pooling over the sequence dimension.
    Mean pooling is preferred over last-token for causal LLMs used as general-purpose
    encoders: the last token is optimized for next-token prediction, not sequence
    summarization, so mean pooling gives a more stable sequence-level representation.

    All expert LLMs share the same hidden size, so we average across experts
    (keeping lm_hidden_size = single expert hidden size, not sum).
    """
    all_hidden = []
    with torch.no_grad():
        for model in expert_models:
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)
            h    = out.hidden_states[-1]                          # (B, seq, hidden)
            mask = attention_mask.unsqueeze(-1).float()           # (B, seq, 1)
            pooled = (h * mask).sum(1) / mask.sum(1)              # (B, hidden)
            all_hidden.append(pooled)
    return torch.stack(all_hidden).mean(0).float()


def get_prompt_hidden_from_reward_models(reward_models, reward_tokenizers,
                                         prompt_texts, device, max_length=256):
    """Solution 1: Use reward model hidden states as prompt features.

    Handles both decoder-only (GPT-2) and encoder-decoder (BART/T5) reward models:
    - Decoder-only: uses the last non-padded token's hidden state (aligned with how
      AutoModelForSequenceClassification computes its score for causal models).
    - Encoder-decoder: uses mean-pooled encoder_last_hidden_state.

    Args:
        reward_models      : list of AutoModelForSequenceClassification (frozen)
        reward_tokenizers  : list of corresponding AutoTokenizer objects
        prompt_texts       : list[str] — raw prompt strings for a batch
        device             : torch device string or object
        max_length         : tokenisation truncation length

    Returns:
        Tensor (batch, hidden_size) — concatenated pooled hidden states from all models
    """
    all_hidden = []
    with torch.no_grad():
        for model, tokenizer in zip(reward_models, reward_tokenizers):
            enc = tokenizer(
                prompt_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
            ).to(device)

            out  = model(**enc, output_hidden_states=True)
            mask = enc['attention_mask']                        # (B, seq)

            # Encoder-decoder models (BART, T5, etc.) — mean pool encoder states
            if (hasattr(out, 'encoder_last_hidden_state')
                    and out.encoder_last_hidden_state is not None):
                h = out.encoder_last_hidden_state              # (B, seq, hidden)
                pooled = (h * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).float()

            # Decoder-only models (GPT-2, etc.) — last non-padded token pooling.
            # This is aligned with how AutoModelForSequenceClassification scores
            # causal LMs: it uses the representation at the last non-padding position.
            elif (hasattr(out, 'hidden_states')
                  and out.hidden_states is not None):
                h = out.hidden_states[-1]                      # (B, seq, hidden)
                # Index of the last non-padding token per batch item
                last_idx = mask.sum(dim=1) - 1                 # (B,)
                pooled   = h[torch.arange(h.shape[0], device=device), last_idx]  # (B, hidden)

            else:
                raise ValueError(
                    f'Cannot extract hidden states from {type(out).__name__}. '
                    'Ensure the model supports output_hidden_states=True.')

            all_hidden.append(pooled)
    # Concatenate along feature dim so different hidden sizes are handled correctly.
    # lm_hidden_size = sum of each model's hidden_size.
    # Cast to float32: reward models are loaded in bfloat16 but GatingNetwork
    # and loss computations expect float32.
    return torch.cat(all_hidden, dim=-1).float()


# ---------------------------------------------------------------------------
# Gating Network Trainer (not in use)
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
# Model merging (not in use)
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

