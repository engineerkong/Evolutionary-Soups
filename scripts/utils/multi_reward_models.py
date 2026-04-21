from transformers import AutoTokenizer
import numpy as np
import torch
from .utils import load_reward_model, get_rewards


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _actual_path(path: str) -> str:
    return path.split('#')[0]


def _is_armorm(path: str) -> bool:
    return _actual_path(path).startswith('armorm://')


def _parse_armorm(path: str):
    """Return (model_path, dim_index) from 'armorm://model_id#N'."""
    body = path[len('armorm://'):]
    if '#' in body:
        model_path, dim = body.rsplit('#', 1)
        return model_path, int(dim)
    return body, 0


def _sub_position(path: str) -> int:
    """Return sub_position: -100 for beaver, int(N) for #N suffix, 0 otherwise."""
    if 'beaver' in _actual_path(path):
        return -100
    if '#' in path:
        return int(path.split('#')[1])
    return 0


# ---------------------------------------------------------------------------
# ArmoRM local multi-objective reward model
# RLHFlow/ArmoRM-Llama3-8B-v0.1 — loaded directly on GPU via HuggingFace.
# Outputs 19 reward objectives; first 5 map to HelpSteer dimensions.
# Path format: 'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1#N'
#   N >= 0 → select rewards[:, N] (specific objective)
#   N < 0  → return aggregated preference score (output.score)
# ---------------------------------------------------------------------------

ARMORM_ATTRIBUTES = [
    'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
    'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
    'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
    'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
    'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm',
    'code-complexity', 'code-style', 'code-explanation',
    'code-instruction-following', 'code-readability',
]

_ARMORM_BATCH = 4  # max (q, r) pairs per GPU forward pass


class ArmoRMClient:
    """Local ArmoRM multi-objective reward model (RLHFlow/ArmoRM-Llama3-8B-v0.1).

    Loaded once per unique model path; shared across multiple reward heads that
    differ only in the selected dimension index.
    """

    def __init__(self, model_path: str, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model.eval()

    def score_batch(self, queries_responses: list, dim: int) -> list:
        """Score (q, r) pairs; return float list for objective dim (or preference score if dim < 0)."""
        results = []
        for start in range(0, len(queries_responses), _ARMORM_BATCH):
            chunk = queries_responses[start:start + _ARMORM_BATCH]
            ids_list = []
            for q, r in chunk:
                messages = [{"role": "user", "content": q},
                            {"role": "assistant", "content": r}]
                ids = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt", truncation=True, max_length=4096,
                ).squeeze(0)
                ids_list.append(ids)
            # Left-pad: decoder model classifies from the last token
            max_len = max(t.size(0) for t in ids_list)
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            padded = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
            for j, ids in enumerate(ids_list):
                padded[j, max_len - ids.size(0):] = ids
                mask[j, max_len - ids.size(0):] = 1
            with torch.no_grad():
                out = self.model(padded.to(self.device), attention_mask=mask.to(self.device))
            if dim < 0:
                results.extend(out.score.cpu().float().tolist())
            else:
                results.extend(out.rewards.cpu().float()[:, dim].tolist())
        return results


# ---------------------------------------------------------------------------
# Standard (non-ArmoRM) helpers
# ---------------------------------------------------------------------------

def _encode_beaver(q: str, r: str) -> str:
    q_clean = q.split('\n\nAssistant:')[0].split('\n\nHuman:')[-1].strip()
    return 'BEGINNING OF CONVERSATION: USER: ' + q_clean + ' ASSISTANT: ' + r.strip()


# ---------------------------------------------------------------------------
# RewardModels
# ---------------------------------------------------------------------------

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
        _model_cache = {}
        _tokenizer_cache = {}
        for i in range(self.num_rewards):
            ap = _actual_path(self.reward_model_path_list[i])

            if _is_armorm(ap):
                model_path = ap[len('armorm://'):]
                if model_path not in _model_cache:
                    print(f'Loading ArmoRM from {model_path}')
                    _model_cache[model_path] = ArmoRMClient(model_path, gpu_id_list[i])
                self.reward_models.append(_model_cache[model_path])
                self.rm_tokenizers.append(None)  # tokeniser is internal to ArmoRMClient
            else:
                tok_ap = _actual_path(self.rm_tokenizer_path_list[i])
                if ap not in _model_cache:
                    _model_cache[ap] = load_reward_model(ap, gpu_id_list[i])
                if tok_ap not in _tokenizer_cache:
                    _tokenizer_cache[tok_ap] = AutoTokenizer.from_pretrained(tok_ap)
                self.reward_models.append(_model_cache[ap])
                self.rm_tokenizers.append(_tokenizer_cache[tok_ap])

    def get_reward_model_scores(self, queries_responses, summary_fun=None, normalize_rewards=False, round_digits=None):
        texts_for_rewards = []
        for i in range(self.num_rewards):
            path = self.reward_model_path_list[i]
            ap = _actual_path(path)

            if _is_armorm(ap):
                texts_for_rewards.append(queries_responses)  # raw (q, r) pairs
                continue

            tok = self.rm_tokenizers[i]
            max_length = min(tok.model_max_length, 1024)

            prev_tok_ap = _actual_path(self.rm_tokenizer_path_list[i - 1]) if i >= 1 else None
            cur_tok_ap = _actual_path(self.rm_tokenizer_path_list[i])
            if i >= 1 and cur_tok_ap == prev_tok_ap and not _is_armorm(_actual_path(self.reward_model_path_list[i - 1])):
                texts_for_rewards.append(texts_for_rewards[-1])
            elif 'beaver' in ap:
                encoded = [
                    tok(_encode_beaver(q, r), return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ]
                texts_for_rewards.append(encoded)
            elif 'faithful' in ap:
                encoded = [
                    tok(text=r, text_pair=summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ]
                texts_for_rewards.append(encoded)
            elif 'summary' in ap or 'summarization' in ap:
                encoded = [
                    tok(r + ' ' + tok.bos_token + ' ' + summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ]
                texts_for_rewards.append(encoded)
            elif 'humor' in ap:
                encoded = [
                    tok(r, return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ]
                texts_for_rewards.append(encoded)
            else:
                encoded = [
                    tok(q, r, return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ]
                texts_for_rewards.append(encoded)

        rewards = []
        for i in range(self.num_rewards):
            path = self.reward_model_path_list[i]
            ap = _actual_path(path)

            if _is_armorm(ap):
                _, dim = _parse_armorm(path)
                temp_reward = self.reward_models[i].score_batch(texts_for_rewards[i], dim)
                if round_digits is not None:
                    temp_reward = [np.round(r, round_digits) for r in temp_reward]
                rewards.append(temp_reward)
                continue

            sub_pos = _sub_position(path)

            if self.reward_stats is not None:
                if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
                    reward_mean_std = (self.reward_stats[2 * i], self.reward_stats[2 * i + 1])
                else:
                    reward_mean_std = self.reward_stats[i]
            else:
                reward_mean_std = None

            if not normalize_rewards:
                reward_mean_std = None

            if 'humor' in ap or 'faithful' in ap:
                sub_pos = 1

            temp_reward = get_rewards(
                self.reward_models[i],
                texts_for_rewards[i],
                reward_mean_std=reward_mean_std,
                sub_position=sub_pos,
                round_digits=round_digits,
            )

            if 'beaver' in ap and 'cost' in ap:
                temp_reward = [-r for r in temp_reward]

            rewards.append(temp_reward)
        return rewards
