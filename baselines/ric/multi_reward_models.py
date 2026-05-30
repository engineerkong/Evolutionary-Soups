"""RiC reward models — copied from RiC/ric/multi_reward_models.py.

Changes vs original (RiC/ric/multi_reward_models.py):
  1. _encode_beaver (new helper): formats (query, response) as
     'BEGINNING OF CONVERSATION: USER: <q> ASSISTANT: <r>' for beaver reward models.
  2. get_reward_model_scores — beaver branch: uses _encode_beaver for tokenization,
     calls get_rewards with sub_position=-100 so it reads .end_scores[0].
  3. get_reward_model_scores — cost negation: beaver_cost rewards are negated
     (the cost model scores unsafe content positively; negating converts to a reward).
  4. load_reward_model / get_rewards: imported from local utils.py which was
     extended to handle beaver (AutoModelForScore, sub_position=-100 branch).
"""

from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd
from utils import load_reward_model, get_rewards


def _encode_beaver(q: str, r: str) -> str:
    """Format (query, response) for BeaverTails reward/cost model input."""
    q_clean = q.split('\n\nAssistant:')[0].split('\n\nHuman:')[-1].strip()
    return 'BEGINNING OF CONVERSATION: USER: ' + q_clean + ' ASSISTANT: ' + r.strip()


class RewardModels():
    def __init__(self, reward_model_path_list, rm_tokenizer_path_list, gpu_id_list, reward_stats_path=None):
        assert len(reward_model_path_list) == len(rm_tokenizer_path_list)
        self.reward_model_path_list = reward_model_path_list
        self.rm_tokenizer_path_list = rm_tokenizer_path_list
        self.num_rewards = len(reward_model_path_list)
        self.reward_stats = np.load(reward_stats_path) if reward_stats_path is not None else None
        self.reward_models = []
        self.rm_tokenizers = []
        if type(gpu_id_list) != list:
            gpu_id_list = [gpu_id_list, gpu_id_list, gpu_id_list]

        print('Loading reward models .....')
        for i in range(self.num_rewards):
            self.reward_models.append(load_reward_model(self.reward_model_path_list[i], gpu_id_list[i]))
            self.rm_tokenizers.append(AutoTokenizer.from_pretrained(self.rm_tokenizer_path_list[i]))

    def get_reward_model_scores(self, queries_responses, summary_fun=None):
        texts_for_rewards = []
        for i in range(self.num_rewards):
            if i >= 1 and self.rm_tokenizer_path_list[i] == self.rm_tokenizer_path_list[i-1]:
                texts_for_rewards.append(texts_for_rewards[-1])
            elif 'beaver' in self.reward_model_path_list[i]:
                # Added: BeaverTails reward model expects a specific conversation format
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [
                    self.rm_tokenizers[i](
                        _encode_beaver(q, r),
                        return_tensors='pt', truncation=True, max_length=max_length,
                    )
                    for q, r in queries_responses
                ]
                texts_for_rewards.append(temp_encoded_texts)
            elif 'faithful' in self.reward_model_path_list[i]:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](text=r, text_pair=summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            elif 'summary' in self.reward_model_path_list[i] or 'summarization' in self.reward_model_path_list[i]:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r + " " + self.rm_tokenizers[i].bos_token + " " + summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            elif 'humor' in self.reward_model_path_list[i]:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            else:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](q, r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)

        rewards = []
        for i in range(self.num_rewards):
            if self.reward_stats is not None:
                if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
                    reward_mean_std = (self.reward_stats[2*i], self.reward_stats[2*i+1])
                else:
                    reward_mean_std = self.reward_stats[i]
            else:
                reward_mean_std = None

            if 'humor' in self.reward_model_path_list[i] or 'faithful' in self.reward_model_path_list[i]:
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std, sub_position=1)
            elif 'beaver' in self.reward_model_path_list[i]:
                # Added: AutoModelForScore returns end_scores; cost is negated to be a reward
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std, sub_position=-100)
                if 'cost' in self.reward_model_path_list[i]:
                    temp_reward = [-r for r in temp_reward]
            else:
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std)
            rewards.append(temp_reward)
        return rewards
