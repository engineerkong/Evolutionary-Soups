"""es_utils.py — Self-contained utilities for the ES package.

Combines what ES scripts need into one module so they don't have to reach into
`baselines.utils.*`. Covers:

  - Reward-model registry (REWARD_PATHS)
  - Simplex sampling
  - GatingNetwork save/load helpers
  - Tokenizer loader, response cleaner
  - Dataset builders for hh-rlhf / summarize_from_feedback / PKU-SafeRLHF-10K
    in three flavours each: _sft (full text), _ppo (prompt-only), _eval (test split)
  - Prompt-formatting classes (Instructions, Instructions_summary)
  - Reward-model loader (standard HF SequenceClassification + Beaver AutoModelForScore)
  - RewardModels (multi-objective scoring wrapper)
"""

import copy
import json
import os
import re
from itertools import product
from typing import List

import numpy as np
import torch
from datasets import disable_caching, load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from es_architecture import GatingNetwork, SimpleGatingNetwork

disable_caching()


# ---------------------------------------------------------------------------
# Reward model paths
# ---------------------------------------------------------------------------
REWARD_PATHS = {
    'harmless':      'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':       'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':       'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':       'Tristan/gpt2_reward_summarization',
    'faithful':      'CogComp/bart-faithful-summary-detector',
    'humor':         'mohameddhiab/humor-no-humor',
    'beaver_reward': 'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':   'PKU-Alignment/beaver-7b-v1.0-cost',
}


# ---------------------------------------------------------------------------
# Simplex sampling
# ---------------------------------------------------------------------------

def get_simplex_samples(n_objectives: int, step: float = 0.2) -> List[List[float]]:
    steps = round(1.0 / step)
    vals  = [round(i * step, 8) for i in range(steps + 1)]
    return [list(c) for c in product(vals, repeat=n_objectives)
            if abs(sum(c) - 1.0) < 1e-6]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_main_tokenizer(tokenizer_name):
    """Load the main LM tokenizer (slow if possible, fall back to fast for Qwen2)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ---------------------------------------------------------------------------
# Prompt / response post-processing
# ---------------------------------------------------------------------------

def get_clean_data(full_responses, full_prompts, remove_bad=False):
    """Strip pad/bos/eos and any model-emitted continuation suffixes."""
    full_prompts_clean = []
    full_responses_clean = []
    for i, response in enumerate(full_responses):
        full_prompts[i] = full_prompts[i].strip('[PAD] ').strip('[PAD]').strip('<s>').strip('</s>').strip()
        response = response.strip('[PAD] ').strip('[PAD]').strip('<s>').strip('</s>')
        temp_resp = response.replace(full_prompts[i], '').strip().strip('\n\n----').strip('\n\n----- ').strip()
        if '</s>' in temp_resp:
            temp_resp = temp_resp[:temp_resp.rindex('</s>')]
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        clean_resp = full_prompts[i] + ' ' + temp_resp
        if remove_bad and (('.....' in clean_resp) or (clean_resp.count(':)') >= 3)):
            continue
        full_responses_clean.append(clean_resp)
        full_prompts_clean.append(full_prompts[i])
    return full_prompts_clean, full_responses_clean


# ---------------------------------------------------------------------------
# Prompt-format helpers
# ---------------------------------------------------------------------------

class Instructions:
    response_split = "\n\nAssistant:"
    input_split    = "\n\nHuman:"

    @staticmethod
    def get_input(query):
        before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip() + ' ' + Instructions.response_split

    @staticmethod
    def get_response(response):
        return response.split(Instructions.response_split)[-1].strip()


class Instructions_summary():
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split      = "### Response:"
    input_split         = "### Input:"
    instruction_split   = "### Instruction:"

    @classmethod
    def prompt_input(self, input):
        return f"### Instruction: {Instructions_summary.instruction_summary} ### Input: {input} ### Response: "

    def get_prompt(self, query):
        before_response = self.response_split.join(query.split(self.response_split)[:-1])
        return before_response.rstrip()

    def get_post(self, query):
        before_response = self.get_prompt(query)
        return before_response.split(self.input_split)[1].strip()

    def get_input(self, query):
        return self.get_prompt(query) + ' ' + self.response_split

    def get_response(self, response):
        return response.split(self.response_split)[-1].strip()


# ---------------------------------------------------------------------------
# Anthropic/hh-rlhf dataset builders
# ---------------------------------------------------------------------------

def build_dataset_sft(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        sample['text'] = sample['chosen']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt']    = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response']  = split_text[-1].strip()
        sample['input_ids'] = tokenizer.encode(sample['text']) + [tokenizer.eos_token_id]
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns(['chosen', 'rejected'])
    ds.set_format(type='torch')
    return ds


def build_dataset_ppo(path, tokenizer, rm_tokenizer=None, split='train', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        sample['text'] = sample['chosen']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt']    = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['input_ids'] = tokenizer.encode(sample['prompt'])
        sample['query']     = tokenizer.decode(sample['input_ids'])
        if rm_tokenizer is not None:
            sample['reward_ids'] = rm_tokenizer.encode(sample['text'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256
                             and 8 <= len(x['reward_ids']) <= 256)
    ds = ds.remove_columns(['rejected', 'chosen', 'reward_ids', 'text'])
    ds.set_format(type='torch')
    return ds


def build_dataset_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, len(ds), 4))

    rm_tokenizer1, rm_tokenizer2 = rm_tokenizers_list[:2]

    def tokenize(sample):
        sample['text'] = sample['chosen']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt']        = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response']      = split_text[-1].strip()
        sample['input_ids']     = tokenizer.encode(sample['prompt'])
        sample['query']         = tokenizer.decode(sample['input_ids'])
        sample['input_ids_rm1'] = rm_tokenizer1.encode(sample['prompt'])
        sample['input_ids_rm2'] = rm_tokenizer2.encode(sample['prompt'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=20)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids'])     <= 512
                             and 8 <= len(x['input_ids_rm1']) <= 512
                             and 8 <= len(x['input_ids_rm2']) <= 512)
    ds = ds.remove_columns(['chosen', 'rejected', 'input_ids_rm1', 'input_ids_rm2',
                            'text', 'prompt', 'response', 'query'])
    ds.set_format(type='torch')
    return ds


# ---------------------------------------------------------------------------
# openai/summarize_from_feedback dataset builders
# ---------------------------------------------------------------------------

def build_dataset_summary_sft(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x['info']['post'] is not None
                             and 100 < len(x['info']['post']) < 1200,
                   batched=False, num_proc=20)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post      = sample['info']['post'].replace('\n', ' ')
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample['prompt']    = prompt_summary
        choice              = sample['choice']
        sample['response']  = sample['summaries'][choice]['text'].replace('\n', ' ').strip()
        sample['input_ids'] = tokenizer.encode(prompt_summary + sample['response']) + [tokenizer.eos_token_id]
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns(['info', 'summaries', 'choice', 'worker',
                            'batch', 'split', 'extra'])
    ds.set_format(type='torch')
    return ds


def build_dataset_summary_ppo(path, tokenizer, rm_tokenizer, split='train', size=None):
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x['info']['post'] is not None
                             and 100 < len(x['info']['post']) < 1200,
                   batched=False, num_proc=30)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post      = sample['info']['post'].replace('\n', ' ')
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample['prompt']    = prompt_summary
        sample['input_ids'] = tokenizer.encode(sample['prompt'])
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns(['info', 'summaries', 'choice', 'worker',
                            'batch', 'split', 'extra'])
    ds.set_format(type='torch')
    return ds


def build_dataset_summary_eval(path, tokenizer, rm_tokenizers, split='test', size=None):
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x['info']['post'] is not None
                             and 100 < len(x['info']['post']) < 1200,
                   batched=False, num_proc=30)

    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']['id'] is not None)
        initial_list = duplicated_dataset.map(lambda x: {'id': x['info']['id']})
        _, unique_indices = np.unique(initial_list['id'], return_index=True, axis=0)
        return duplicated_dataset.select(unique_indices.tolist())

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post      = sample['info']['post'].replace('\n', ' ')
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample['prompt']    = prompt_summary
        sample['input_ids'] = tokenizer.encode(prompt_summary)
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns(['info', 'summaries', 'choice', 'worker',
                            'batch', 'split', 'extra'])
    ds.set_format(type='torch')
    return ds


# ---------------------------------------------------------------------------
# PKU-Alignment/PKU-SafeRLHF-10K dataset builders
# ---------------------------------------------------------------------------

def build_dataset_beaver_sft(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, split='train')
    if split == 'test':
        ds = ds.select(range(0, len(ds), 12))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def tokenize(sample):
        chosen = sample[f"response_{sample['better_response_id']}"]
        text   = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:' + chosen
        sample['text']      = text
        sample['input_ids'] = tokenizer.encode(text) + [tokenizer.eos_token_id]
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    keep = {'input_ids', 'query', 'text'}
    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type='torch')
    return ds


def build_dataset_beaver_ppo(path, tokenizer, rm_tokenizer=None, split='train', size=None):
    ds = load_dataset(path, split='train')
    if split == 'test':
        ds = ds.select(range(0, len(ds), 12))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def tokenize(sample):
        prompt = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:'
        sample['prompt']    = prompt
        sample['input_ids'] = tokenizer.encode(prompt)
        sample['query']     = tokenizer.decode(sample['input_ids'])
        if rm_tokenizer is not None:
            sample['reward_ids'] = rm_tokenizer.encode(prompt)
        return sample

    drop = ['response_0', 'response_1', 'is_response_0_safe',
            'is_response_1_safe', 'better_response_id', 'safer_response_id']
    ds = ds.map(tokenize, batched=False, num_proc=30)
    if rm_tokenizer is not None:
        ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256
                                 and 8 <= len(x['reward_ids']) <= 256)
        ds = ds.remove_columns(['reward_ids'] + [c for c in drop if c in ds.column_names])
    else:
        ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256)
        ds = ds.remove_columns([c for c in drop if c in ds.column_names])
    ds.set_format(type='torch')
    return ds


def build_dataset_beaver_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None):
    ds = load_dataset(path, split='train')
    ds = ds.select(range(0, len(ds), 12))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def tokenize(sample):
        prompt = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:'
        sample['input_ids'] = tokenizer.encode(prompt)
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    drop = ['prompt', 'response_0', 'response_1', 'is_response_0_safe',
            'is_response_1_safe', 'better_response_id', 'safer_response_id']
    ds = ds.map(tokenize, batched=False, num_proc=20)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256)
    ds = ds.remove_columns([c for c in drop if c in ds.column_names])
    ds.set_format(type='torch')
    return ds


# ---------------------------------------------------------------------------
# Reward model loading
# ---------------------------------------------------------------------------

def _check_lora_in_model_path(model, path):
    if os.path.exists(path):
        if 'adapter_config.json' in os.listdir(path):
            return True
        for key in model.state_dict().keys():
            if 'lora' in key:
                return True
    return False


def load_reward_model(reward_peft_path, gpu_id):
    """Load a reward model (Beaver `AutoModelForScore` or standard `AutoModelForSequenceClassification`)."""
    actual_path = reward_peft_path.split('#')[0]
    if 'beaver' in actual_path:
        try:
            from safe_rlhf.models import AutoModelForScore
        except ImportError:
            raise ImportError(
                'safe_rlhf is required for BeaverTails reward models. '
                'Install from: https://github.com/PKU-Alignment/safe-rlhf')
        reward_model = AutoModelForScore.from_pretrained(
            actual_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
        return reward_model.to(gpu_id)
    num_labels   = 2 if ('humor' in actual_path or 'faithful' in actual_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        actual_path, num_labels=num_labels, torch_dtype=torch.bfloat16, device_map=gpu_id)
    if _check_lora_in_model_path(reward_model, actual_path):
        reward_model = PeftModel.from_pretrained(reward_model, actual_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload()
    return reward_model.to(gpu_id)


def get_rewards(reward_model, texts_for_rewards, reward_mean_std=None,
                sub_position=0, round_digits=1):
    """Forward a batch of pre-tokenised inputs through `reward_model` and return floats."""
    rewards = []
    with torch.no_grad():
        for inputs in texts_for_rewards:
            if sub_position == -100:                              # Beaver AutoModelForScore
                rewards.append(reward_model(**(inputs.to(reward_model.device))).end_scores[0])
            else:
                logits = reward_model(**(inputs.to(reward_model.device))).logits[0]
                if logits.dim() > 0 and logits.numel() > 1:
                    rewards.append(logits[sub_position])
                else:
                    rewards.append(logits)

    if reward_mean_std is None:
        rewards = [r.cpu().detach().item() for r in rewards]
    else:
        mean_reward, std_reward = reward_mean_std
        rewards = [(r.cpu().detach().item() - mean_reward) / std_reward for r in rewards]
    if round_digits is not None:
        rewards = [np.round(r, round_digits) for r in rewards]
    return rewards


# ---------------------------------------------------------------------------
# Multi-reward wrapper
# ---------------------------------------------------------------------------

def _actual_path(path: str) -> str:
    return path.split('#')[0]


def _sub_position(path: str) -> int:
    """-100 for Beaver, int(N) for any `#N` suffix, otherwise 0."""
    if 'beaver' in _actual_path(path):
        return -100
    if '#' in path:
        return int(path.split('#')[1])
    return 0


def _encode_beaver(q: str, r: str) -> str:
    q_clean = q.split('\n\nAssistant:')[0].split('\n\nHuman:')[-1].strip()
    return 'BEGINNING OF CONVERSATION: USER: ' + q_clean + ' ASSISTANT: ' + r.strip()


class RewardModels:
    """Load one or more reward models and score (query, response) pairs against all of them."""

    def __init__(self, reward_model_path_list, rm_tokenizer_path_list, gpu_id_list,
                 reward_stats_path=None):
        assert len(reward_model_path_list) == len(rm_tokenizer_path_list)
        self.reward_model_path_list = reward_model_path_list
        self.rm_tokenizer_path_list = rm_tokenizer_path_list
        self.num_rewards            = len(reward_model_path_list)
        self.reward_stats           = (np.load(reward_stats_path)
                                       if reward_stats_path is not None else None)
        self.reward_models          = []
        self.rm_tokenizers          = []
        if not isinstance(gpu_id_list, list):
            gpu_id_list = [gpu_id_list] * self.num_rewards

        print('Loading reward models …')
        _model_cache, _tok_cache = {}, {}
        for i in range(self.num_rewards):
            ap     = _actual_path(self.reward_model_path_list[i])
            tok_ap = _actual_path(self.rm_tokenizer_path_list[i])
            if ap not in _model_cache:
                _model_cache[ap] = load_reward_model(ap, gpu_id_list[i])
            if tok_ap not in _tok_cache:
                _tok_cache[tok_ap] = AutoTokenizer.from_pretrained(tok_ap)
            self.reward_models.append(_model_cache[ap])
            self.rm_tokenizers.append(_tok_cache[tok_ap])

    def to_device(self, device):
        """Move every distinct underlying model to `device`."""
        seen = set()
        for model in self.reward_models:
            if id(model) in seen or model is None:
                continue
            seen.add(id(model))
            model.to(device)

    def get_reward_model_scores(self, queries_responses, summary_fun=None,
                                normalize_rewards=False, round_digits=None):
        """Return a list of `num_rewards` lists, each of length len(queries_responses)."""
        texts_for_rewards = []
        for i in range(self.num_rewards):
            ap  = _actual_path(self.reward_model_path_list[i])
            tok = self.rm_tokenizers[i]
            max_length = min(tok.model_max_length, 1024)

            prev_tok_ap = _actual_path(self.rm_tokenizer_path_list[i - 1]) if i >= 1 else None
            cur_tok_ap  = _actual_path(self.rm_tokenizer_path_list[i])
            if i >= 1 and cur_tok_ap == prev_tok_ap:
                texts_for_rewards.append(texts_for_rewards[-1])
            elif 'beaver' in ap:
                texts_for_rewards.append([
                    tok(_encode_beaver(q, r), return_tensors='pt',
                        truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ])
            elif 'faithful' in ap:
                texts_for_rewards.append([
                    tok(text=r, text_pair=summary_fun(q), return_tensors='pt',
                        truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ])
            elif 'summary' in ap or 'summarization' in ap:
                texts_for_rewards.append([
                    tok(r + ' ' + tok.bos_token + ' ' + summary_fun(q),
                        return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ])
            elif 'humor' in ap:
                texts_for_rewards.append([
                    tok(r, return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ])
            else:
                texts_for_rewards.append([
                    tok(q, r, return_tensors='pt', truncation=True, max_length=max_length)
                    for q, r in queries_responses
                ])

        rewards = []
        for i in range(self.num_rewards):
            ap      = _actual_path(self.reward_model_path_list[i])
            sub_pos = _sub_position(self.reward_model_path_list[i])
            if 'humor' in ap or 'faithful' in ap:
                sub_pos = 1

            reward_mean_std = None
            if normalize_rewards and self.reward_stats is not None:
                if isinstance(self.reward_stats, list) or len(self.reward_stats) == 2 * self.num_rewards:
                    reward_mean_std = (self.reward_stats[2 * i], self.reward_stats[2 * i + 1])
                else:
                    reward_mean_std = self.reward_stats[i]

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


# ---------------------------------------------------------------------------
# Gating network save / load
# ---------------------------------------------------------------------------

def save_gating_network(gating_net, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    torch.save(gating_net.state_dict(), os.path.join(save_path, 'gating_network.pt'))
    config = {
        'type':           type(gating_net).__name__,
        'lm_hidden_size': gating_net.lm_hidden_size,
        'num_experts':    gating_net.num_experts,
        'hidden_size':    gating_net.hidden_size,
        'fixed_alpha':    gating_net.fixed_alpha,
    }
    if isinstance(gating_net, GatingNetwork):
        config['num_layers'] = gating_net.num_layers
    with open(os.path.join(save_path, 'gating_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def load_gating_network(save_path: str, lm_hidden_size: int = 4096,
                        num_experts: int = 2, num_layers: int = 32,
                        device: str = 'cuda'):
    resolved  = _resolve_checkpoint(save_path, 'gating_network.pt')
    ckpt_file = os.path.join(resolved, 'gating_network.pt')
    if not os.path.exists(ckpt_file):
        return None

    net_type    = 'GatingNetwork'
    hidden_size = 256
    fixed_alpha = None
    cfg_file = os.path.join(resolved, 'gating_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        net_type       = cfg.get('type',           net_type)
        lm_hidden_size = cfg.get('lm_hidden_size', lm_hidden_size)
        num_experts    = cfg.get('num_experts',     num_experts)
        hidden_size    = cfg.get('hidden_size',     hidden_size)
        num_layers     = cfg.get('num_layers',      num_layers)
        fixed_alpha    = cfg.get('fixed_alpha',     None)

    net = GatingNetwork(lm_hidden_size=lm_hidden_size, num_experts=num_experts,
                        hidden_size=hidden_size, num_layers=num_layers,
                        fixed_alpha=fixed_alpha)
    # strict=False allows loading old checkpoints that lack the alpha parameter
    net.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
    return net.to(device).bfloat16()


def load_simple_gating_network(save_path: str, lm_hidden_size: int = 4096,
                                num_experts: int = 2, device: str = 'cuda'):
    """Load a SimpleGatingNetwork checkpoint saved by save_gating_network."""
    resolved  = _resolve_checkpoint(save_path, 'gating_network.pt')
    ckpt_file = os.path.join(resolved, 'gating_network.pt')
    if not os.path.exists(ckpt_file):
        return None

    hidden_size = 256
    fixed_alpha = 1.0
    cfg_file = os.path.join(resolved, 'gating_config.json')
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            cfg = json.load(f)
        lm_hidden_size = cfg.get('lm_hidden_size', lm_hidden_size)
        num_experts    = cfg.get('num_experts',     num_experts)
        hidden_size    = cfg.get('hidden_size',     hidden_size)
        fixed_alpha    = cfg.get('fixed_alpha',     fixed_alpha)

    net = SimpleGatingNetwork(lm_hidden_size=lm_hidden_size, num_experts=num_experts,
                              hidden_size=hidden_size, fixed_alpha=fixed_alpha)
    net.load_state_dict(torch.load(ckpt_file, map_location=device), strict=False)
    return net.to(device).bfloat16()


def _resolve_checkpoint(checkpoint_path: str, filename: str) -> str:
    if not checkpoint_path:
        return checkpoint_path
    if os.path.exists(os.path.join(checkpoint_path, filename)):
        return checkpoint_path
    if not os.path.isdir(checkpoint_path):
        return checkpoint_path
    candidates = []
    for entry in os.listdir(checkpoint_path):
        subdir = os.path.join(checkpoint_path, entry)
        if not os.path.exists(os.path.join(subdir, filename)):
            continue
        key = (
            int(re.search(r'step_(\d+)',  entry).group(1)) if re.search(r'step_(\d+)',  entry) else -1,
            int(re.search(r'epoch_(\d+)', entry).group(1)) if re.search(r'epoch_(\d+)', entry) else -1,
            os.path.getmtime(subdir),
        )
        candidates.append((key, subdir))
    return sorted(candidates, reverse=True)[0][1] if candidates else checkpoint_path


# ---------------------------------------------------------------------------
# GatingNetwork ↔ flat parameter vector
# ---------------------------------------------------------------------------

def net_to_params(net) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().float().numpy().ravel() for p in net.parameters()]
    )


def params_to_net(params: np.ndarray, template, device: str = 'cpu'):
    net = copy.deepcopy(template).to(device)
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(params[offset:offset + n].reshape(p.shape),
                                 dtype=p.dtype, device=device))
            offset += n
    return net


def make_onehot_params(template, expert_idx: int) -> np.ndarray:
    """Flat params that force gating to output one-hot[expert_idx] for any input.

    Zeroes the last linear weight so only the bias matters, then sets
    bias[expert_idx]=+100 and all others=-100. Entmax → exactly one-hot.
    """
    net = copy.deepcopy(template)
    with torch.no_grad():
        last = net.net[-1]              # nn.Linear(hidden_size, num_experts)
        last.weight.zero_()
        last.bias.fill_(-100.0)
        last.bias[expert_idx] = 100.0
    return net_to_params(net)


# ---------------------------------------------------------------------------
# Pareto / NSGA-II / NSGA-III / greedy-HVC selection
# ---------------------------------------------------------------------------

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(rewards: np.ndarray) -> List[List[int]]:
    P = len(rewards)
    dom_count    = np.zeros(P, dtype=int)
    dominated_by = [[] for _ in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            if dominates(rewards[i], rewards[j]):
                dominated_by[i].append(j); dom_count[j] += 1
            elif dominates(rewards[j], rewards[i]):
                dominated_by[j].append(i); dom_count[i] += 1
    fronts = [[i for i in range(P) if dom_count[i] == 0]]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(rewards: np.ndarray, front: List[int]) -> np.ndarray:
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    r    = rewards[front]
    for m in range(r.shape[1]):
        order = np.argsort(r[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = r[order[-1], m] - r[order[0], m]
        if span == 0:
            continue
        for idx in range(1, n - 1):
            dist[order[idx]] += (r[order[idx + 1], m] - r[order[idx - 1], m]) / span
    return dist


def nsga2_select(rewards: np.ndarray, pop_size: int) -> List[int]:
    fronts   = non_dominated_sort(rewards)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            remaining = pop_size - len(selected)
            dist      = crowding_distance(rewards, front)
            ranked    = [x for _, x in sorted(zip(-dist, front))]
            selected.extend(ranked[:remaining])
            break
    return selected


def generate_reference_points(n_objectives: int, n_divisions: int) -> np.ndarray:
    """Das-Dennis structured reference points on the unit hyperplane (sum=1, ≥0)."""
    def _gen(n_obj: int, n_div: int, cur: list, result: list) -> None:
        if n_obj == 1:
            result.append(cur + [n_div])
        else:
            for i in range(n_div + 1):
                _gen(n_obj - 1, n_div - i, cur + [i], result)
    pts: list = []
    _gen(n_objectives, n_divisions, [], pts)
    return np.array(pts, dtype=np.float32) / n_divisions


def nsga3_select(rewards: np.ndarray, pop_size: int,
                 reference_points: np.ndarray) -> List[int]:
    """NSGA-III selection: non-dominated sort + niche preservation on critical front."""
    fronts   = non_dominated_sort(rewards)
    selected: List[int] = []
    last_front: List[int] = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            last_front = list(front)
            break

    n_needed = pop_size - len(selected)
    if n_needed == 0 or not last_front:
        return selected[:pop_size]

    all_idx    = selected + last_front
    fit_all    = rewards[all_idx]
    ideal      = fit_all.min(axis=0)
    translated = fit_all - ideal

    if selected:
        nadir = (rewards[selected] - ideal).max(axis=0)
    else:
        nadir = translated.max(axis=0)
    nadir      = np.where(nadir < 1e-10, 1.0, nadir)
    normalized = translated / nadir

    R       = reference_points
    r_norms = np.linalg.norm(R, axis=1, keepdims=True)
    r_norms = np.where(r_norms < 1e-10, 1.0, r_norms)
    R_hat   = R / r_norms

    dot       = normalized @ R_hat.T
    proj      = dot[:, :, None] * R_hat[None, :, :]
    diff      = normalized[:, None, :] - proj
    perp_dist = np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))

    assoc_ref  = perp_dist.argmin(axis=1)
    assoc_dist = perp_dist.min(axis=1)

    n_sel       = len(selected)
    niche_count = np.zeros(len(R), dtype=int)
    for j in range(n_sel):
        niche_count[assoc_ref[j]] += 1

    lf_assoc = assoc_ref[n_sel:]
    lf_dist  = assoc_dist[n_sel:]

    remaining = list(range(len(last_front)))
    chosen_from_last: List[int] = []

    for _ in range(n_needed):
        ref_to_cands: dict = {}
        for loc in remaining:
            ref_to_cands.setdefault(lf_assoc[loc], []).append(loc)
        if not ref_to_cands:
            break
        min_nc   = min(niche_count[r] for r in ref_to_cands)
        min_refs = [r for r in ref_to_cands if niche_count[r] == min_nc]
        chosen_r = min_refs[np.random.randint(len(min_refs))]
        cands    = ref_to_cands[chosen_r]
        if niche_count[chosen_r] == 0:
            loc = cands[int(np.argmin(lf_dist[cands]))]
        else:
            loc = cands[np.random.randint(len(cands))]
        chosen_from_last.append(last_front[loc])
        niche_count[chosen_r] += 1
        remaining.remove(loc)

    return selected + chosen_from_last


try:
    from pymoo.indicators.hv import HV as _PymooHV
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


def hv(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume (maximisation) via pymoo — negate to convert to min convention."""
    ind = _PymooHV(ref_point=-ref)
    return float(ind(-points))


def hypervolume_contribution(candidates: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Exclusive hypervolume contribution HVC[i] = HV(C) - HV(C \\ {i})."""
    n        = len(candidates)
    total_hv = hv(candidates, ref)
    hvc      = np.zeros(n)
    for i in range(n):
        hvc[i] = total_hv - hv(np.delete(candidates, i, axis=0), ref)
    return hvc


def greedy_hvc_select(rewards: np.ndarray, pop_size: int) -> List[int]:
    """Non-dominated sort + sequential greedy HVC fill on the critical front."""
    fronts   = non_dominated_sort(rewards)
    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            n_needed  = pop_size - len(selected)
            all_pts   = rewards[np.array(selected + front)] if selected else rewards[np.array(front)]
            ref       = all_pts.min(axis=0) - 0.1
            chosen    = [rewards[k] for k in selected]
            hv_base   = hv(np.array(chosen), ref) if chosen else 0.0
            remaining = list(range(len(front)))
            for _ in range(n_needed):
                gains    = [hv(np.array(chosen + [rewards[front[loc]]]), ref) - hv_base
                            for loc in remaining]
                best_pos = int(np.argmax(gains))
                best_loc = remaining[best_pos]
                hv_base += gains[best_pos]
                chosen.append(rewards[front[best_loc]])
                selected.append(front[best_loc])
                remaining.remove(best_loc)
            break
    return selected


# ---------------------------------------------------------------------------
# MoE rollout + multi-objective scoring
# ---------------------------------------------------------------------------

def generate_and_score(moe_model, prompt_input_ids, prompt_attention,
                       sft_tokenizer, reward_models, instructions,
                       generation_kwargs, gpu_id, num_continuations=1,
                       normalize_rewards=False, sample_writer=None):
    """Roll out responses with `moe_model`, score with `reward_models`.

    Returns the mean reward vector (M,) over the batch and continuations.

    sample_writer: optional callable invoked once per (cont_idx, prompt_idx,
        prompt_clean, response_clean, [reward_0, …]).  Used to dump per-sample
        rows to a CSV without changing aggregation.
    """
    device          = f'cuda:{gpu_id}'
    accumulated     = None
    prompts_decoded = sft_tokenizer.batch_decode(prompt_input_ids.cpu())

    for cont_idx in range(num_continuations):
        outputs = moe_model.generate(
            prompt_input_ids.to(device),
            attention_mask=prompt_attention.to(device),
            **generation_kwargs,
        )
        responses = sft_tokenizer.batch_decode(outputs.cpu())
        del outputs

        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(instructions.get_input(r), instructions.get_response(r))
                 for r in responses_clean]
        if hasattr(instructions, 'get_post'):
            scores = reward_models.get_reward_model_scores(
                pairs, instructions.get_post,
                normalize_rewards=normalize_rewards, round_digits=None)
        else:
            scores = reward_models.get_reward_model_scores(
                pairs, normalize_rewards=normalize_rewards, round_digits=None)

        n_prompts, n_rewards = len(prompts_clean), len(scores)
        if accumulated is None:
            accumulated = [[[] for _ in range(n_rewards)] for _ in range(n_prompts)]
        for p in range(n_prompts):
            for k in range(n_rewards):
                accumulated[p][k].append(scores[k][p])
            if sample_writer is not None:
                sample_writer(cont_idx, p, prompts_clean[p], responses_clean[p],
                              [float(scores[k][p]) for k in range(n_rewards)])
        torch.cuda.empty_cache()

    per_prompt = np.array([[np.mean(accumulated[p][k]) for k in range(n_rewards)]
                           for p in range(n_prompts)])
    return per_prompt.mean(axis=0)
