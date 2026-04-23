import os
import gc
import shutil
import yaml
from dataclasses import dataclass, field
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from datasets import load_dataset, disable_caching
import numpy as np
disable_caching()

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    

def sample_preferences_uniform(num_rewards, num_samples):
    """Sample preferences uniformly from the simplex using a grid."""
    if num_rewards == 2:
        # For 2D, use evenly spaced points
        preferences = []
        for i in range(num_samples):
            w = i / (num_samples - 1) if num_samples > 1 else 0.5
            preferences.append([w, 1 - w])
        return preferences
    else:
        # For higher dimensions, generate grid points on simplex
        # Find appropriate grid resolution to get ~num_samples points
        from itertools import combinations_with_replacement
        
        def simplex_grid(dim, resolution):
            """Generate evenly spaced points on a simplex."""
            points = []
            # Generate all combinations that sum to resolution
            for combo in combinations_with_replacement(range(resolution + 1), dim - 1):
                # Convert to differences to get partition
                prev = 0
                parts = []
                for val in combo:
                    parts.append(val - prev)
                    prev = val
                parts.append(resolution - prev)
                # Normalize to get point on simplex
                point = [p / resolution for p in parts]
                points.append(point)
            return points
        
        # Binary search for resolution that gives ~num_samples points
        # Number of points for dim d and resolution r is C(r + d - 1, d - 1)
        from math import comb
        resolution = 1
        while comb(resolution + num_rewards - 1, num_rewards - 1) < num_samples:
            resolution += 1
        
        points = simplex_grid(num_rewards, resolution)
        
        # If we have more points than needed, subsample evenly
        if len(points) > num_samples:
            indices = np.linspace(0, len(points) - 1, num_samples, dtype=int)
            points = [points[i] for i in indices]
        
        return points
    

class Instructions:
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"

    @staticmethod
    def get_input(query):
        before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip() + ' ' + Instructions.response_split
        
    @staticmethod
    def get_response(response):
        return response.split(Instructions.response_split)[-1].strip()


class Instructions_summary():
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"

    @classmethod
    def prompt_input(self, input):
        # formulate the news
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

# Anthropic HH-RLHF dataset builders
def build_dataset_sft(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=30)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected'])

    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_ppo(path, tokenizer, rm_tokenizer=None, split='train', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    
    def tokenize(sample, reject=False):
        if not reject:
            sample['text'] = sample['chosen'] 
        else:
            sample['text'] = sample['rejected'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        if rm_tokenizer is not None:
            sample['reward_ids'] = rm_tokenizer.encode(sample['text']) # for data filter
        return sample

    ds_concat = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=30)
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 256 and len(x["input_ids"]) >= 8 and len(x['reward_ids']) <= 256 and len(x['reward_ids']) >= 8)
    ds_concat = ds_concat.remove_columns(['rejected', 'chosen', 'reward_ids', 'text'])

    ds_concat.set_format(type="torch")
    return ds_concat

def build_dataset_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, len(ds), 4))

    rm_tokenizer1, rm_tokenizer2 = rm_tokenizers_list[:2]
    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["input_ids_rm1"] = rm_tokenizer1.encode(sample["prompt"])
        sample["input_ids_rm2"] = rm_tokenizer2.encode(sample["prompt"])
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=20)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8 \
                                 and len(x["input_ids_rm1"]) <= 512 and len(x["input_ids_rm1"]) >= 8
                                 and len(x["input_ids_rm2"]) <= 512 and len(x["input_ids_rm2"]) >= 8 )
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected','input_ids_rm1', 'input_ids_rm2', 'text', 'prompt', 'response', 'query'])
    ds_concat.set_format(type="torch")
    return ds_concat


# openai summarize_from_feedback dataset builders
def build_dataset_summary_sft(path, tokenizer, split='train', size=None):
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=20)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        choice = sample["choice"] # select the best summary
        sample["response"] = sample["summaries"][choice]["text"].replace("\n", " ").strip()
        sample["input_ids"] = tokenizer.encode(prompt_summary + sample["response"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def build_dataset_summary_ppo(path, tokenizer, rm_tokenizer, split='train', size=None):
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def build_dataset_summary_eval(path, tokenizer, rm_tokenizers, split='test', size=None):
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    # need to remove duplicated prompts for evaluation
    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(prompt_summary)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


# argilla news-summary dataset builders
def build_dataset_news_summary_sft(path, tokenizer, split='train', size=None):
    """SFT dataset builder for argilla/news-summary (full prompt+summary pairs)."""
    ds = load_dataset(path, split=split)
    ds = ds.filter(
        lambda x: x['text'] is not None and 100 < len(x['text']) < 2000
                  and x['prediction'] is not None and len(x['prediction']) > 0,
        batched=False, num_proc=20)

    if size is not None:
        ds = ds.select(range(size))

    def _extract_summary(pred_item: dict) -> str:
        # argilla datasets use 'label' for text generation predictions;
        # fall back to 'value' or 'text' for other possible schemas
        for key in ('label', 'value', 'text'):
            v = pred_item.get(key)
            if isinstance(v, str) and v.strip():
                return v.replace('\n', ' ').strip()
        return ''

    def tokenize(sample):
        article = sample['text'].replace('\n', ' ')
        prompt_summary = Instructions_summary.prompt_input(article)
        summary = _extract_summary(sample['prediction'][0])
        sample['prompt'] = prompt_summary
        sample['response'] = summary
        sample['input_ids'] = tokenizer.encode(prompt_summary + summary) + [tokenizer.eos_token_id]
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: len(x['input_ids']) <= 512 and len(x['input_ids']) >= 8
                             and len(x['response']) > 0)
    keep = {'input_ids', 'query', 'prompt', 'response'}
    remove = [c for c in ds.column_names if c not in keep]
    ds = ds.remove_columns(remove)
    ds.set_format(type='torch')
    return ds


def build_dataset_news_summary_ppo(path, tokenizer, rm_tokenizer, split='train', size=None):
    """Dataset builder for argilla/news-summary (PPO / NSGA-II prompt-only format)."""
    ds = load_dataset(path, split=split)
    ds = ds.filter(lambda x: x['text'] is not None and 100 < len(x['text']) < 2000,
                   batched=False, num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        article = sample['text'].replace('\n', ' ')
        prompt_summary = Instructions_summary.prompt_input(article)
        sample['prompt'] = prompt_summary
        sample['input_ids'] = tokenizer.encode(prompt_summary)
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: len(x['input_ids']) <= 512 and len(x['input_ids']) >= 8)
    keep = {'input_ids', 'query', 'prompt'}
    remove = [c for c in ds.column_names if c not in keep]
    ds = ds.remove_columns(remove)
    ds.set_format(type='torch')
    return ds


def check_lora_in_model_path(model, path):
    if os.path.exists(path):
        dirnames = os.listdir(path)
        if 'adapter_config.json' in dirnames:
            return True
        
        state_dict_keys = model.state_dict().keys()
        for key in state_dict_keys:
            if 'lora' in key:
                return True
    return False


def load_reward_model(reward_peft_path, gpu_id):
    actual_path = reward_peft_path.split('#')[0]  # strip steer dimension suffix (e.g. 'nvidia/HelpSteer#0')
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
    num_labels = 2 if ('humor' in actual_path or 'faithful' in actual_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    actual_path,
                    num_labels=num_labels, torch_dtype=torch.bfloat16,
                    device_map=gpu_id,
                    )
    if check_lora_in_model_path(reward_model, actual_path):
        reward_model = PeftModel.from_pretrained(reward_model, actual_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload()
    return reward_model.to(gpu_id)

# Load main tokenizer without adding special tokens
def load_main_tokenizer(tokenizer_name):
    # use_fast=False avoids tokenizers-version mismatch when tokenizer.json was
    # saved with a newer tokenizers library than is installed (ModelWrapper parse error)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_rewards(reward_model, texts_for_rewards, reward_mean_std=None, sub_position=0, round_digits=1):
    rewards = []
    with torch.no_grad():
        for inputs in texts_for_rewards:
            if sub_position == -100:  # BeaverTails AutoModelForScore
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


def save_configs(config, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'training_config.txt'), 'w+') as f:
        if type(config) == dict:
            lines = [key + ' : ' + config[key] + '\n' for key in config.keys()]
            f.writelines(lines)
        else:
            f.writelines(str(config))


def get_average_state_dict(state_dicts, coefficients):
    i = 0
    for state_dict, coefficient in zip(state_dicts, coefficients):
        current_weights = state_dict
        for key in list(current_weights.keys()):
            if i == 0:
                state_dicts[0][key] = coefficient * current_weights[key]
            else:
                state_dicts[0][key] += coefficient * current_weights[key]
        i += 1
    return state_dicts[0]


def merge_weights_with_preference(base_model_names, preference, temp_save_path):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    model_1 = models[0]
    model_1.load_state_dict(average_weights, strict=False)
    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path, ignore_errors=True)
    model_1.save_pretrained(temp_save_path)

    while len(models):
        del models[0]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()


def merge_lora_weight(model, path):
    if check_lora_in_model_path(model, path):
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    return model


# ---------------------------------------------------------------------------
# BeaverTails (PKU-Alignment/PKU-SafeRLHF-10K) dataset builders
# BeaverTails uses the same \n\nHuman: / \n\nAssistant: format as hh-rlhf.
# The dataset only has a 'train' split; split='test' subsamples it.
# ---------------------------------------------------------------------------

def build_dataset_beaver_sft(path, tokenizer, split='train', size=None):
    """SFT dataset for PKU-Alignment/PKU-SafeRLHF-10K (uses better_response)."""
    ds = load_dataset(path, split='train')
    if split == 'test':
        ds = ds.select(range(0, len(ds), 12))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def tokenize(sample):
        chosen = sample[f"response_{sample['better_response_id']}"]
        text = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:' + chosen
        sample['text'] = text
        sample['input_ids'] = tokenizer.encode(text) + [tokenizer.eos_token_id]
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    keep = {'input_ids', 'query', 'text'}
    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type='torch')
    return ds


def build_dataset_beaver_ppo(path, tokenizer, rm_tokenizer=None, split='train', size=None):
    """PPO/NSGA-II prompt-only dataset for PKU-Alignment/PKU-SafeRLHF-10K."""
    ds = load_dataset(path, split='train')
    if split == 'test':
        ds = ds.select(range(0, len(ds), 12))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def tokenize(sample):
        prompt = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:'
        sample['prompt'] = prompt
        sample['input_ids'] = tokenizer.encode(prompt)
        sample['query'] = tokenizer.decode(sample['input_ids'])
        if rm_tokenizer is not None:
            sample['reward_ids'] = rm_tokenizer.encode(prompt)
        return sample

    drop = ['response_0', 'response_1', 'is_response_0_safe',
            'is_response_1_safe', 'better_response_id', 'safer_response_id']
    ds = ds.map(tokenize, batched=False, num_proc=30)
    if rm_tokenizer is not None:
        ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256 and 8 <= len(x['reward_ids']) <= 256)
        ds = ds.remove_columns(['reward_ids'] + [c for c in drop if c in ds.column_names])
    else:
        ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256)
        ds = ds.remove_columns([c for c in drop if c in ds.column_names])
    ds.set_format(type='torch')
    return ds


def build_dataset_beaver_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None):
    """Eval dataset for PKU-SafeRLHF-10K (subsamples train as test set)."""
    ds = load_dataset(path, split='train')
    ds = ds.select(range(0, len(ds), 12))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def tokenize(sample):
        prompt = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:'
        sample['input_ids'] = tokenizer.encode(prompt)
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    drop = ['prompt', 'response_0', 'response_1', 'is_response_0_safe',
            'is_response_1_safe', 'better_response_id', 'safer_response_id']
    ds = ds.map(tokenize, batched=False, num_proc=20)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 256)
    ds = ds.remove_columns([c for c in drop if c in ds.column_names])
    ds.set_format(type='torch')
    return ds


# ---------------------------------------------------------------------------
# HelpSteer / HelpSteer2 (nvidia/HelpSteer, nvidia/HelpSteer2) dataset builders
# HelpSteer uses \n\nHuman: prefix; HelpSteer2 uses \n\nuser: (lowercase).
# ---------------------------------------------------------------------------

def _steer_prompt_prefix(path: str) -> str:
    return '\n\nuser: ' if 'HelpSteer2' in path else '\n\nHuman: '


def build_dataset_steer_sft(path, tokenizer, split='train', size=None):
    """SFT dataset for nvidia/HelpSteer or nvidia/HelpSteer2."""
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))
    prefix = _steer_prompt_prefix(path)

    def tokenize(sample):
        text = prefix + sample['prompt'] + ' \n\nAssistant: ' + sample['response']
        sample['text'] = text
        sample['input_ids'] = tokenizer.encode(text) + [tokenizer.eos_token_id]
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    keep = {'input_ids', 'query', 'text'}
    ds = ds.map(tokenize, batched=False, num_proc=30)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type='torch')
    return ds


def build_dataset_steer_ppo(path, tokenizer, rm_tokenizer=None, split='train', size=None):
    """PPO/NSGA-II prompt-only dataset for nvidia/HelpSteer or HelpSteer2."""
    import pandas as pd
    from datasets import Dataset as HFDataset
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, split=split)
    import pandas as pd
    df = pd.DataFrame(ds)
    ds = HFDataset.from_pandas(df.drop_duplicates('prompt').reset_index(drop=True))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))
    prefix = _steer_prompt_prefix(path)

    def tokenize(sample):
        prompt = prefix + sample['prompt'] + ' \n\nAssistant: '
        sample['prompt'] = prompt
        sample['input_ids'] = tokenizer.encode(prompt)
        sample['query'] = tokenizer.decode(sample['input_ids'])
        if rm_tokenizer is not None:
            sample['reward_ids'] = rm_tokenizer.encode(prompt)
        return sample

    drop = ['response', 'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    ds = ds.map(tokenize, batched=False, num_proc=30)
    if rm_tokenizer is not None:
        ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512 and 8 <= len(x['reward_ids']) <= 512)
        ds = ds.remove_columns(['reward_ids'] + [c for c in drop if c in ds.column_names])
    else:
        ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
        ds = ds.remove_columns([c for c in drop if c in ds.column_names])
    ds.set_format(type='torch')
    return ds


def build_dataset_steer_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None):
    """Eval dataset for nvidia/HelpSteer or HelpSteer2."""
    from datasets import Dataset as HFDataset
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, split=split)
    import pandas as pd
    df = pd.DataFrame(ds)
    ds = HFDataset.from_pandas(df.drop_duplicates('prompt').reset_index(drop=True))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))
    ds = ds.select(range(0, min(len(ds), 2000)))
    prefix = _steer_prompt_prefix(path)

    def tokenize(sample):
        prompt = prefix + sample['prompt'] + ' \n\nAssistant: '
        sample['input_ids'] = tokenizer.encode(prompt)
        sample['query'] = tokenizer.decode(sample['input_ids'])
        return sample

    drop = ['prompt', 'response', 'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']
    ds = ds.map(tokenize, batched=False, num_proc=20)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    ds = ds.remove_columns([c for c in drop if c in ds.column_names])
    ds.set_format(type='torch')
    return ds


def get_clean_data(full_responses, full_prompts, remove_bad=False):
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
            ## pass bad sample
            continue
        full_responses_clean.append(clean_resp)
        full_prompts_clean.append(full_prompts[i])
    return full_prompts_clean, full_responses_clean

