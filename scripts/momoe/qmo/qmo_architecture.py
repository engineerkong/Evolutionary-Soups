import sys
from pathlib import Path

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.utils import get_clean_data, sample_preferences_uniform


class QTableGating:
    """Q-Learning gating: state=discretized preference, action=discretized expert weights."""

    def __init__(self, num_experts=2, num_pref_bins=11, num_action_bins=11, alpha=0.1, epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.1):
        self.num_experts = num_experts
        self.num_pref_bins = num_pref_bins
        self.num_action_bins = num_action_bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Enumerate all valid simplex grid points for state and action spaces.
        # For N experts, there are N-1 free dimensions; the last = 1 - sum(others).
        self._state_pts  = self._simplex_grid(num_pref_bins)
        self._action_pts = self._simplex_grid(num_action_bins)
        self._state_coords  = np.array(self._state_pts)  / max(num_pref_bins  - 1, 1)
        self._action_coords = np.array(self._action_pts) / max(num_action_bins - 1, 1)
        self.q_table = np.zeros((len(self._state_pts), len(self._action_pts)))

    def _simplex_grid(self, num_bins):
        from itertools import product
        n_free = self.num_experts - 1
        pts = [c for c in product(range(num_bins), repeat=n_free) if sum(c) <= num_bins - 1]
        return np.array(pts, dtype=np.float32) if pts else np.zeros((1, n_free), dtype=np.float32)

    def discretize_preference(self, preference):
        free = np.array(preference[:self.num_experts - 1], dtype=np.float32)
        return int(np.argmin(np.linalg.norm(self._state_coords - free, axis=1)))

    def select_action(self, pref_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self._action_pts))
        return int(np.argmax(self.q_table[pref_idx]))

    def update(self, pref_idx, action_idx, reward):
        self.q_table[pref_idx, action_idx] += self.alpha * (reward - self.q_table[pref_idx, action_idx])

    def get_weights(self, pref_idx, action_idx):
        free = self._action_coords[action_idx].tolist()
        return free + [max(0.0, 1.0 - sum(free))]

    def best_weights(self, preference):
        pref_idx = self.discretize_preference(preference)
        action_idx = int(np.argmax(self.q_table[pref_idx]))
        return self.get_weights(pref_idx, action_idx)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MoEGatingTrainer:
    def __init__(self, moe_model, expert_state_dicts, reward_models, instructions,
                 num_rewards=2, num_pref_samples=10,
                 num_pref_bins=11, num_action_bins=11, alpha=0.1, epsilon=0.3):
        self.model = moe_model
        self.expert_state_dicts = expert_state_dicts
        self.reward_models = reward_models
        self.instructions = instructions
        self.num_rewards = num_rewards
        self.num_pref_samples = num_pref_samples
        self.q_table = QTableGating(
            num_experts=num_rewards,
            num_pref_bins=num_pref_bins,
            num_action_bins=num_action_bins,
            alpha=alpha,
            epsilon=epsilon,
        )
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def core_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def sample_preferences(self):
        return sample_preferences_uniform(self.num_rewards, self.num_pref_samples)
        # return [np.random.dirichlet(np.ones(self.num_rewards)).tolist() for _ in range(self.num_pref_samples)]

    def _pad_inputs(self, batch, tokenizer):
        items = [ids.clone().detach() if isinstance(ids, torch.Tensor) else torch.tensor(ids) for ids in batch["input_ids"]]
        max_length = max(len(ids) for ids in items)
        padded_ids, masks = [], []
        for ids in items:
            pad = max_length - len(ids)
            padded_ids.append(torch.cat([torch.full((pad,), tokenizer.pad_token_id, dtype=ids.dtype), ids]))
            masks.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))
        device = next(self.model.parameters()).device
        return torch.stack(padded_ids).to(device), torch.stack(masks).to(device)

    def _score_generations(self, generated_sequences, prompts, preference, tokenizer):
        responses = tokenizer.batch_decode(generated_sequences)
        prompt_text = tokenizer.batch_decode(prompts)
        prompt_text, responses = get_clean_data(responses, prompt_text)
        query_response_pairs = [
            (self.instructions.get_input(text), self.instructions.get_response(text))
            for text in responses
        ]
        if hasattr(self.instructions, "get_post"):
            rewards_list = self.reward_models.get_reward_model_scores(
                query_response_pairs, self.instructions.get_post, normalize_rewards=False
            )
        else:
            rewards_list = self.reward_models.get_reward_model_scores(
                query_response_pairs, normalize_rewards=False
            )
        return [
            float(sum(preference[k] * rewards_list[k][idx] for k in range(self.num_rewards)))
            for idx in range(len(query_response_pairs))
        ]

    def train_step_qlearning(self, batch, tokenizer):
        input_ids, attention_mask = self._pad_inputs(batch, tokenizer)
        all_rewards = []
        all_expert_weights = []
        preferences = self.sample_preferences()

        self.model.eval()
        with torch.no_grad():
            for preference in preferences:
                pref_idx = self.q_table.discretize_preference(preference)
                action_idx = self.q_table.select_action(pref_idx)
                weights = self.q_table.get_weights(pref_idx, action_idx)

                merge_model_weights(self.core_model, self.expert_state_dicts, weights)
                generated = self.core_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                all_expert_weights.append(weights)
                rewards = self._score_generations(generated, input_ids, preference, tokenizer)
                mean_reward = float(np.mean(rewards))
                all_rewards.append(mean_reward)
                self.q_table.update(pref_idx, action_idx, mean_reward)

        self.q_table.decay_epsilon()
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        mean_expert_weights = np.mean(all_expert_weights, axis=0).tolist() if all_expert_weights else []

        return {
            "mean_reward": rewards_tensor.mean().item(),
            "std_reward": rewards_tensor.std(unbiased=False).item(),
            "mean_expert_weights": mean_expert_weights,
            "epsilon": self.q_table.epsilon,
        }


def merge_model_weights(model, expert_state_dicts, expert_weights):
    """Merge full model state dicts in-place, exactly matching get_average_state_dict in utils.py.

    Replicates the original behavior precisely:
      i=0: merged[key] = w0 * expert0[key]
      i>0: merged[key] += wi * experti[key]
    All arithmetic stays in the original dtype (bfloat16) with no float32 cast,
    matching eval_ppo_rs results exactly.
    """
    device = next(model.parameters()).device
    merged_state = {}
    for i, (state_dict, weight) in enumerate(zip(expert_state_dicts, expert_weights)):
        for key, val in state_dict.items():
            val_on_device = val.to(device=device)
            if i == 0:
                merged_state[key] = weight * val_on_device
            else:
                merged_state[key] = merged_state[key] + weight * val_on_device

    model.load_state_dict(merged_state, strict=False)
    return model