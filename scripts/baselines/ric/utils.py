"""RiC utilities — copied from RiC/ric/utils.py.

Changes vs original (RiC/ric/utils.py):
  1. load_reward_model: added beaver branch — loads PKU-Alignment/beaver-7b-v1.0-{reward,cost}
     via safe_rlhf.models.AutoModelForScore instead of AutoModelForSequenceClassification.
  2. get_rewards: added sub_position=-100 branch that reads .end_scores[0] (BeaverTails
     models return end_scores rather than logits).
  3. build_beaver_dataset_with_preference_n (new): eval dataset for PKU-SafeRLHF-10K
     with score-conditioned prompts; mirrors build_dataset_with_preference_n but loads
     from the beaver dataset and uses response_{better_response_id} columns.
  4. reset_score_in_dataset, dataset_from_csv_n, balancing_rewards: exp_type checks
     extended from =='assistant' to in ('assistant','beaver') since both use
     Instructions_n with the same \n\nAssistant: response template.
"""

import os
import gc
import copy
import shutil
from typing import Optional
from peft import PeftModel
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk, disable_caching
import numpy as np
import pandas as pd
from tqdm import tqdm
disable_caching()

def clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class Instructions():
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"
    score_split1 = "<rm1_score>"
    score_split2 = "<rm2_score>"

    @staticmethod
    def get_prompt(query):
        if Instructions.score_split1 in query:
            before_response = query.split(Instructions.score_split1)[0]
        else:
            before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip()

    @staticmethod
    def get_input(query):
        if Instructions.score_split1 in query:
            before_response = query.split(Instructions.score_split1)[0]
        else:
            before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip() + ' ' + Instructions.response_split

    @staticmethod
    def get_response(response):
        return response.split(Instructions.response_split)[-1].strip()

    @staticmethod
    def get_score1(query):
        after_score = " ".join(query.split(Instructions.score_split1)[1:]).strip()
        before_response = after_score.split(Instructions.score_split2)[0]
        return before_response

    @staticmethod
    def get_score2(query):
        after_score = " ".join(query.split(Instructions.score_split2)[1:]).strip()
        before_response = after_score.split(Instructions.response_split)[0]
        return before_response


class Instructions_n():
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"
    def __init__(self, num_rewards):
        self.num_rewards = num_rewards
        self.score_splits = []
        for i in range(num_rewards):
            self.score_splits.append("<rm{}_score>".format(i+1))

    def get_prompt(self, query):
        if self.score_splits[0] in query:
            before_response = query.split(self.score_splits[0])[0]
        else:
            before_response = self.response_split.join(query.split(self.response_split)[:-1])
        return before_response.rstrip()

    def get_input(self, query):
        return self.get_prompt(query) + ' ' + self.response_split

    def get_response(self, response):
        return response.split(self.response_split)[-1].strip()

    def get_score_index(self, query, index):
        after_score = " ".join(query.split(self.score_splits[index])[1:]).strip()
        if index >= self.num_rewards - 1:
            before_response = after_score.split(self.response_split)[0]
        else:
            before_response = after_score.split(self.score_splits[index+1])[0]
        return before_response.strip()

    def get_scores(self, query):
        scores = []
        for i in range(self.num_rewards):
            scores.append(self.get_score_index(query, i))
        return scores


class Instructions_summary_n():
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"
    def __init__(self, num_rewards=2):
        self.num_rewards = num_rewards
        self.score_splits = []
        for i in range(num_rewards):
            self.score_splits.append("<rm{}_score>".format(i+1))

    @classmethod
    def prompt_input_noscore(self, input):
        return f"### Instruction: {Instructions_summary_n.instruction_summary} ### Input: {input} ### Response: "

    def prompt_input_score(self, input, scores):
        text = f"### Instruction: {Instructions_summary_n.instruction_summary} ### Input: {input} "
        for i in range(self.num_rewards):
            text += self.score_splits[i] + ' ' + str(round(scores[i], 1)) + ' '
        text += "### Response: "
        return text

    def get_prompt(self, query):
        if self.score_splits[0] in query:
            before_response = query.split(self.score_splits[0])[0]
        else:
            before_response = self.response_split.join(query.split(self.response_split)[:-1])
        return before_response.rstrip()

    def get_post(self, query):
        before_response = self.get_prompt(query)
        return before_response.split(self.input_split)[1].strip()

    def get_input(self, query):
        return self.get_prompt(query) + ' ' + self.response_split

    def get_response(self, response):
        return response.split(self.response_split)[-1].strip()

    def get_score_index(self, query, index):
        after_score = " ".join(query.split(self.score_splits[index])[1:]).strip()
        if index >= self.num_rewards - 1:
            before_response = after_score.split(self.response_split)[0]
        else:
            before_response = after_score.split(self.score_splits[index+1])[0]
        return before_response.strip()

    def get_scores(self, query):
        scores = []
        for i in range(self.num_rewards):
            scores.append(self.get_score_index(query, i))
        return scores


def build_dataset_with_preference(path, tokenizer, rm_tokenizer1, rm_tokenizer2, preference, split='test',  size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, len(ds), 4))

    def tokenize(sample, reject=False):
        if not reject:
            sample['text'] = sample['chosen']
        else:
            sample['text'] = sample['rejected']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample['reward_ids1'] = rm_tokenizer1.encode(sample['text'])
        sample['reward_ids2'] = rm_tokenizer2.encode(sample['text'])
        return sample

    ds = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=20)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8 \
                            and len(x['reward_ids1']) <= 512 and len(x['reward_ids1']) >= 8\
                            and len(x['reward_ids2']) <= 512 and len(x['reward_ids2']) >= 8)
    ds = ds.remove_columns(['rejected', 'chosen', 'reward_ids1', 'reward_ids2'])

    ds = ds.add_column('score1', np.zeros(len(ds)) + preference[0])
    ds = ds.add_column('score2', np.zeros(len(ds)) + preference[1])

    def add_score(sample):
        sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' ' + Instructions.score_split1 + ' ' + str(sample['score1']) + ' '  \
                                                + ' ' + Instructions.score_split2 + ' ' + str(sample['score2']) + ' ' + '\n\nAssistant:'
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(add_score, batched=False, num_proc=20)
    ds.set_format(type="torch")
    return ds


def build_dataset_with_preference_n(path, tokenizer, rm_tokenizers, preference, split='test',  size=None):
    ds = load_dataset(path, split=split)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, len(ds), 4))
    n = len(rm_tokenizers)
    instructions = Instructions_n(n)

    def tokenize(sample, reject=False):
        if not reject:
            sample['text'] = sample['chosen']
        else:
            sample['text'] = sample['rejected']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        for i in range(n):
            if type(rm_tokenizers[i]) != str:
                sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(sample['text'])
        return sample

    ds = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=20)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['rejected', 'chosen']
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds = ds.filter(lambda x: len(x['reward_ids{}'.format(1+i)]) <= 512 and len(x['reward_ids{}'.format(1+i)]) >= 8)
            remove_columns.append('reward_ids{}'.format(1+i))

    ds = ds.remove_columns(remove_columns)
    for i in range(n):
        ds = ds.add_column('score{}'.format(i+1), np.zeros(len(ds)) + preference[i])

    def add_score(sample):
        sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' '
        for i in range(n):
            sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(sample['score{}'.format(i+1)]) + ' '
        sample['prompt_with_score'] += '\n\nAssistant:'
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(add_score, batched=False, num_proc=20)
    ds.set_format(type="torch")
    return ds


def build_summary_dataset_with_preference_n(path, tokenizer, rm_tokenizers, preference, split='test',  size=None):
    if split == 'test':
        split = 'validation'
    ds = load_dataset(path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=20)

    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))
    ds = ds.select(range(0, min(len(ds), 2000)))

    n = len(rm_tokenizers)
    instructions = Instructions_summary_n(n)

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary_n.prompt_input_noscore(info_post)
        sample["prompt"] = prompt_summary
        sample["response"] = sample["summaries"][sample["choice"]]["text"].replace("\n", " ").strip()
        sample["input_ids"] = tokenizer.encode(prompt_summary + sample["response"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        for i in range(n):
            if type(rm_tokenizers[i]) != str:
                sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(sample["query"])
        return sample

    ds = ds.map(tokenize, batched=False, num_proc=20)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds = ds.filter(lambda x: len(x['reward_ids{}'.format(1+i)]) <= 512 and len(x['reward_ids{}'.format(1+i)]) >= 8)
            remove_columns.append('reward_ids{}'.format(1+i))
    ds = ds.remove_columns(remove_columns)

    for i in range(n):
        ds = ds.add_column('score{}'.format(i+1), np.zeros(len(ds)) + preference[i])

    def add_score(sample):
        sample['prompt_with_score'] = sample['prompt'].rstrip('### Response:') + ' '
        for i in range(n):
            sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(sample['score{}'.format(i+1)]) + ' '
        sample['prompt_with_score'] += '### Response:'
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(add_score, batched=False, num_proc=20)
    ds.set_format(type="torch")
    return ds


def build_beaver_dataset_with_preference_n(path, tokenizer, rm_tokenizers, preference, split='test', size=None):
    """Eval dataset for PKU-SafeRLHF-10K with score-conditioned prompts for RiC.

    Added for beaver support: builds prompt_with_score_ids that RiC's evaluation
    loop uses to condition the model on target reward levels.
    """
    ds = load_dataset(path, split='train')
    ds = ds.select(range(0, len(ds), 12))  # subsample as test set
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))
    n = len(rm_tokenizers)
    instructions = Instructions_n(n)

    def tokenize(sample):
        prompt_formatted = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:'
        chosen = sample[f'response_{sample["better_response_id"]}']
        text = prompt_formatted + chosen
        sample['text']     = text
        sample['prompt']   = prompt_formatted
        sample['response'] = chosen
        sample['input_ids'] = tokenizer.encode(text)
        sample['query']     = tokenizer.decode(sample['input_ids'])
        for i in range(n):
            if type(rm_tokenizers[i]) != str:
                sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(text)
        return sample

    drop = ['response_0', 'response_1', 'is_response_0_safe',
            'is_response_1_safe', 'better_response_id', 'safer_response_id']
    ds = ds.map(tokenize, batched=False, num_proc=20)
    ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    remove_columns = [c for c in drop if c in ds.column_names]
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds = ds.filter(lambda x: 8 <= len(x['reward_ids{}'.format(1+i)]) <= 512)
            remove_columns.append('reward_ids{}'.format(1+i))
    ds = ds.remove_columns(remove_columns)

    for i in range(n):
        ds = ds.add_column('score{}'.format(i+1), np.zeros(len(ds)) + preference[i])

    def add_score(sample):
        sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' '
        for i in range(n):
            sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(sample['score{}'.format(i+1)]) + ' '
        sample['prompt_with_score'] += '\n\nAssistant:'
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample['input_ids'] = tokenizer.encode(sample['prompt_with_score'] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample['query']     = tokenizer.decode(sample['input_ids'])
        return sample

    ds = ds.map(add_score, batched=False, num_proc=20)
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
    # Added: beaver models use AutoModelForScore from safe-rlhf
    if 'beaver' in reward_peft_path:
        try:
            from safe_rlhf.models import AutoModelForScore
        except ImportError:
            raise ImportError(
                'safe_rlhf is required for BeaverTails reward models. '
                'Install from: https://github.com/PKU-Alignment/safe-rlhf')
        reward_model = AutoModelForScore.from_pretrained(
            reward_peft_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
        return reward_model.to(gpu_id)

    num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_peft_path,
                    num_labels=num_labels, torch_dtype=torch.bfloat16,
                    device_map=gpu_id,
                    )
    if check_lora_in_model_path(reward_model, reward_peft_path):
        reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload()
    return reward_model.to(gpu_id)


def load_main_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_rewards(reward_model, texts_for_rewards, reward_mean_std=None, sub_position=0):
    rewards = []
    print('log: reward model forwarding ...')
    with torch.no_grad():
        pbar = tqdm(total=len(texts_for_rewards))
        for inputs in texts_for_rewards:
            if sub_position == -100:  # BeaverTails AutoModelForScore — added for beaver
                rewards.append(reward_model(**(inputs.to(reward_model.device))).end_scores[0])
            elif sub_position != 0:
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0][sub_position])
            else:
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0])
            pbar.update(1)

    if reward_mean_std is None:
        rewards = [np.round(r.cpu().detach().item(), 1) for r in rewards]
    else:
        mean_reward, std_reward = reward_mean_std
        rewards = [np.round((r.cpu().detach().item() - mean_reward) / std_reward, 1) for r in rewards]
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
    from transformers import AutoModelForCausalLM
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='cpu')
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
            continue
        full_responses_clean.append(clean_resp)
        full_prompts_clean.append(full_prompts[i])
    return full_prompts_clean, full_responses_clean


def map_rewards_from_preference(rewards_list, preference, method='linf'):
    n = len(rewards_list)
    target_rewards = np.zeros(n)
    if method == 'linf':
        max_id = np.argmax(preference)
        target_rewards[max_id] = np.round(np.quantile(rewards_list[max_id], 1), 1)
        for i in range(n):
            if i != max_id:
                low, high = np.quantile(rewards_list[i], 0), np.quantile(rewards_list[i], 1)
                target_rewards[i] = np.round(min(n * preference[i], 1) * (high - low) + low, 1)
    elif method == 'l2':
        for i in range(n):
            low, high = np.quantile(rewards_list[i], 0), np.quantile(rewards_list[i], 1)
            target_rewards[i] = np.round(preference[i] / np.sqrt((np.power(preference, 2).sum())) * (high - low) + low, 1)
    elif method == 'linear':
        for i in range(n):
            low, high = np.quantile(rewards_list[i], 0), np.quantile(rewards_list[i], 1)
            target_rewards[i] = np.round(preference[i] * (high - low) + low, 1)
    else:
        raise NotImplementedError
    return target_rewards


def sample_goals(size, num_rewards=2, rewards_list=None, maximum=0.9999):
    if rewards_list is None:
        samples = np.random.normal(0, 1, 100000)
        low, high = np.round(np.quantile(samples, 0), 1), np.round(np.quantile(samples, 1), 1)
        preferences = np.round(np.random.random((size, num_rewards)) * (high - low) + low, 1)
        for k in range(len(preferences)):
            min_index = np.argmin(preferences[k])
            for j in range(num_rewards):
                if j != min_index:
                    preferences[k][j] = high
    else:
        raise NotImplementedError
    return np.round(preferences, 1)


def reset_score_in_dataset(dataset, tokenizer, rewards_list=None, exp_type='assistant'):
    n = 0
    for name in dataset.column_names:
        if name.startswith('score'):
            n += 1
    preferences = sample_goals(len(dataset), n, rewards_list)

    if exp_type in ('assistant', 'beaver'):
        instructions = Instructions_n(n)
    else:
        instructions = Instructions_summary_n(n)

    for i in range(n):
        if 'score{}'.format(i+1) in dataset.column_names:
            if len(preferences.shape) >= 2:
                desired_score = preferences[:,i]
            else:
                desired_score = np.zeros(len(dataset)) + preferences
            dataset = dataset.remove_columns('score{}'.format(i+1)).add_column('score{}'.format(i+1), desired_score)

    def add_score(sample):
        sample['prompt_with_score'] = sample['prompt'].rstrip(instructions.response_split) + ' '
        for i in range(n):
            sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(np.round(sample['score{}'.format(i+1)].item(), 1)) + ' '
        sample['prompt_with_score'] += instructions.response_split
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        return sample
    return dataset.map(add_score, batched=False, num_proc=20)


def dataset_from_csv(checkpoint_path, tokenizer):
    generated_dataset = pd.read_csv(checkpoint_path + '/data.csv')
    generated_dataset = Dataset.from_pandas(generated_dataset)
    generated_dataset = generated_dataset.rename_column('response', 'query').rename_column('obtained_score1', 'score1').rename_column('obtained_score2', 'score2')
    generated_dataset = generated_dataset.remove_columns(['desired_score1', 'desired_score2', 'Unnamed: 0'])

    def process(sample):
        sample['prompt'] = Instructions.get_input(sample['query'])
        sample['response'] = Instructions.get_response(sample['query'])
        sample['text'] = sample['prompt'] + ' ' + sample['response']
        sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' ' + Instructions.score_split1 + ' ' + str(sample['score1']) + ' '  \
                                                + ' ' + Instructions.score_split2 + ' ' + str(sample['score2']) + ' ' + '\n\nAssistant:'
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    threshold1, threshold2 = np.quantile(generated_dataset['score1'], 0.7), np.quantile(generated_dataset['score2'], 0.7)
    generated_dataset = generated_dataset.filter(lambda x: x["score1"] >= threshold1 or x["score2"] >= threshold2)
    generated_dataset = generated_dataset.map(process, batched=False, num_proc=20)
    return generated_dataset


def select_data_with_quantile(dataset, quantile_threshold, num_scores=None):
    if num_scores == None:
        num_scores = 0
        for key in dataset.column_names:
            if key.startswith('score'):
                num_scores += 1
    thresholds = [np.quantile(dataset['score{}'.format(i+1)], quantile_threshold) for i in range(num_scores)]
    return dataset.filter(lambda x: np.any([x["score{}".format(i+1)] >= thresholds[i] for i in range(num_scores)]))


def dataset_from_csv_n(checkpoint_path, tokenizer, exp_type='assistant', quantile_threshold=0.7):
    generated_dataset = pd.read_csv(checkpoint_path + '/data.csv')
    generated_dataset = Dataset.from_pandas(generated_dataset)
    generated_dataset = generated_dataset.rename_column('response', 'query')
    num_scores = 0
    for key in generated_dataset.column_names:
        if key.startswith('obtained_score'):
            i = int(key.strip('obtained_score'))
            generated_dataset = generated_dataset.rename_column('obtained_score{}'.format(i), 'score{}'.format(i))
            num_scores += 1

    generated_dataset = generated_dataset.remove_columns(['desired_score{}'.format(i+1) for i in range(num_scores)] + ['Unnamed: 0'])
    if exp_type in ('assistant', 'beaver'):
        instructions = Instructions_n(num_scores)
    else:
        instructions = Instructions_summary_n(num_scores)

    def process(sample):
        sample['prompt'] = instructions.get_input(sample['query'])
        sample['response'] = instructions.get_response(sample['query'])
        sample['text'] = sample['prompt'] + ' ' + sample['response']
        sample['prompt_with_score'] = sample['prompt'].rstrip(instructions.response_split)
        for i in range(num_scores):
            sample['prompt_with_score'] += ' ' + instructions.score_splits[i] + ' ' + str(sample['score{}'.format(i+1)])
        sample['prompt_with_score'] += ' ' + instructions.response_split
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    generated_dataset = select_data_with_quantile(generated_dataset, quantile_threshold, num_scores)
    generated_dataset = generated_dataset.map(process, batched=False, num_proc=20)
    return generated_dataset


def merge_dataset(dataset, online_dataset, save_path, tokenizer_name, info_path=None, sample_origin=10000, exp_type='assistant', quantile_threshold=0.7):
    tokenizer = load_main_tokenizer(tokenizer_name)
    if type(dataset) == str:
        dataset = load_from_disk(dataset)

    dataset = select_data_with_quantile(dataset, quantile_threshold)
    if sample_origin is not None:
        selected_index = np.arange(0, len(dataset))
        np.random.shuffle(selected_index)
        selected_dataset = dataset.select(selected_index[:sample_origin])
    else:
        selected_dataset = dataset

    generated_dataset = dataset_from_csv_n(save_path, tokenizer, exp_type=exp_type, quantile_threshold=quantile_threshold)
    if online_dataset is None:
        online_dataset = generated_dataset
    else:
        online_dataset = concatenate_datasets([online_dataset, generated_dataset])

    if len(selected_dataset):
        merged_dataset = concatenate_datasets([selected_dataset, online_dataset])
    else:
        merged_dataset = online_dataset
    print('Size of merged dataset: {}'.format(len(merged_dataset)))
    return merged_dataset, online_dataset


def balancing_rewards(train_dataset, tokenizer, info_path=None, exp_type='assistant'):
    info = {}
    names = train_dataset.column_names
    if info_path is None:
        for key in names:
            if key.startswith('score'):
                scores = train_dataset[key]
                scores = np.clip(scores, -10, 10)
                score_max, score_min = np.quantile(scores, 0.9999), np.quantile(scores, 0.0001)
                ratio1, ratio2 = 4.0 / score_max, -4.0 / score_min
                info[key] = [ratio1, ratio2]
    else:
        info = np.load(info_path, allow_pickle=True).item()

    n = len(info.keys())
    instructions = Instructions_n(n) if exp_type in ('assistant', 'beaver') else Instructions_summary_n(n)
    def add_score(sample):
        sample['prompt_with_score'] = sample['prompt'].rstrip(instructions.response_split) + ' '
        for i in range(n):
            key = 'score{}'.format(i+1)
            ratio1, ratio2 = info[key]
            if sample[key] >= 0:
                sample[key] = np.round(np.clip(ratio1 * sample[key], -4, 4), 1)
            else:
                sample[key] = np.round(np.clip(ratio2 * sample[key], -4, 4), 1)
            sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(np.round(sample['score{}'.format(i+1)].item(), 1)) + ' '
        sample['prompt_with_score'] += instructions.response_split
        sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
        sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    train_dataset = train_dataset.map(add_score, batched=False, num_proc=20)
    return train_dataset, info
