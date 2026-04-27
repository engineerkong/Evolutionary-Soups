"""RiC — prepare training dataset with reward scores.

Adapted from RiC/ric/prepare_dataset_with_rewards.py.

Changes vs original (RiC/ric/prepare_dataset_with_rewards.py):
  1. build_dataset_beaver (new): builds the RiC scored training dataset from
     PKU-Alignment/PKU-SafeRLHF-10K. Uses response_{better_response_id} as the
     chosen response and response_{1-better_response_id} as the rejected response,
     formatted as '\n\nHuman:<prompt> \n\nAssistant:<response>'.
  2. add_score_beaver: alias for add_score_assistant (beaver uses the same
     \n\nAssistant: prompt template, so score injection is identical).
  3. exp_type='beaver' handling in main block: selects Instructions_n (not
     Instructions_summary_n) and calls build_dataset_beaver.
  4. reward_path_tokenizer_dict: extended with beaver_reward / beaver_cost entries.
"""

from dataclasses import dataclass, field
from typing import Optional
import datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk
from accelerate import Accelerator
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from utils import Instructions_n, load_main_tokenizer, Instructions_summary_n
from multi_reward_models import RewardModels

hhrlhf_dataset_path  = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
beaver_dataset_path  = 'PKU-Alignment/PKU-SafeRLHF-10K'
tokenizer_name       = 'meta-llama/Llama-2-7b-hf'


@dataclass
class ScriptArguments:
    reward_names:   Optional[str] = field(default='harmless,helpful')
    save_directory: Optional[str] = field(default='./datasets/all_full_train_harmhelp.hf')
    exp_type:       Optional[str] = field(default='assistant', metadata={"help": "assistant | summary | beaver"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)

reward_path_tokenizer_dict = {
    'harmless':     ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful':      ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta':      ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary':      ['Tristan/gpt2_reward_summarization'],
    'faithful':     ['CogComp/bart-faithful-summary-detector'],
    'humor':        ['mohameddhiab/humor-no-humor'],
    'beaver_reward':['PKU-Alignment/beaver-7b-v1.0-reward'],
    'beaver_cost':  ['PKU-Alignment/beaver-7b-v1.0-cost'],
}

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f'Unknown reward: {name}')
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

tokenizer    = load_main_tokenizer(tokenizer_name)
gpu_id       = Accelerator().local_process_index
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizers = reward_models.rm_tokenizers


def build_dataset(index, tokenizer, rm_tokenizers, split='train'):
    ds = load_dataset(hhrlhf_dataset_path, split=split)
    n = len(rm_tokenizers)

    num_proc = Accelerator().num_processes
    if num_proc > 1:
        start = index * len(ds) // num_proc
        end = (index+1) * len(ds) // num_proc if index < num_proc - 1 else len(ds)
        print(start, end)
        ds = ds.select(range(start, end))

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

    ds_chosen   = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=10)
    ds_rejected = ds.map(tokenize, batched=False, fn_kwargs={"reject": True},  num_proc=10)
    ds_concat   = concatenate_datasets([ds_chosen, ds_rejected])
    ds_concat   = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    print(ds_concat)
    remove_columns = ['rejected', 'chosen']
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds_concat = ds_concat.filter(lambda x: len(x['reward_ids{}'.format(1+i)]) <= 512 and len(x['reward_ids{}'.format(1+i)]) >= 8)
            remove_columns.append('reward_ids{}'.format(1+i))
    ds_concat = ds_concat.remove_columns(remove_columns)

    queries_responses = [(q, r) for q, r in zip(ds_concat['prompt'], ds_concat['response'])]
    rewards_list = reward_models.get_reward_model_scores(queries_responses)
    for i in range(len(rewards_list)):
        ds_concat = ds_concat.add_column('score{}'.format(i+1), rewards_list[i])
    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary(index, tokenizer, rm_tokenizers, split='train'):
    ds = load_dataset(summary_dataset_path, 'comparisons')
    ds = ds[split]
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=20)
    n = len(rm_tokenizers)

    num_proc = Accelerator().num_processes
    if num_proc > 1:
        start = index * len(ds) // num_proc
        end = (index+1) * len(ds) // num_proc if index < num_proc - 1 else len(ds)
        print(start, end)
        ds = ds.select(range(start, end))

    def tokenize(sample, chosen=True):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = instructions.prompt_input_noscore(info_post)
        sample["prompt"] = prompt_summary
        if chosen:
            choice = sample["choice"]
        else:
            choice = 0 if sample["choice"] != 0 else 1
        sample["response"] = sample["summaries"][choice]["text"].replace("\n", " ").strip()
        sample["input_ids"] = tokenizer.encode(prompt_summary + sample["response"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        for i in range(n):
            if type(rm_tokenizers[i]) != str:
                sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(sample["query"])
        return sample

    ds_chosen   = ds.map(tokenize, batched=False, fn_kwargs={'chosen': True},  num_proc=20)
    ds_rejected = ds.map(tokenize, batched=False, fn_kwargs={'chosen': False}, num_proc=20)
    ds = concatenate_datasets([ds_chosen, ds_rejected])
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds = ds.filter(lambda x: len(x['reward_ids{}'.format(1+i)]) <= 512 and len(x['reward_ids{}'.format(1+i)]) >= 8)
            remove_columns.append('reward_ids{}'.format(1+i))
    ds = ds.remove_columns(remove_columns)

    queries_responses = [(q, r) for q, r in zip(ds['prompt'], ds['response'])]
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    for i in range(len(rewards_list)):
        ds = ds.add_column('score{}'.format(i+1), rewards_list[i])
    ds.set_format(type="torch")
    return ds


def build_dataset_beaver(index, tokenizer, rm_tokenizers, split='train'):
    """Added: build RiC training dataset from PKU-Alignment/PKU-SafeRLHF-10K.

    Beaver uses the same \n\nHuman:/\n\nAssistant: format as hh-rlhf.
    Both the better and worse responses are included (same as chosen/rejected for hh-rlhf).
    """
    ds = load_dataset(beaver_dataset_path, split='train')  # only has train split
    n = len(rm_tokenizers)

    num_proc = Accelerator().num_processes
    if num_proc > 1:
        start = index * len(ds) // num_proc
        end = (index+1) * len(ds) // num_proc if index < num_proc - 1 else len(ds)
        print(start, end)
        ds = ds.select(range(start, end))

    def tokenize(sample, chosen=True):
        if chosen:
            resp = sample[f'response_{sample["better_response_id"]}']
        else:
            worse_id = 1 - sample['better_response_id']
            resp = sample[f'response_{worse_id}']
        prompt_formatted = '\n\nHuman:' + sample['prompt'] + ' \n\nAssistant:'
        text = prompt_formatted + resp
        # overwrite 'prompt' with the formatted version (original 'prompt' column is raw text)
        sample['text']     = text
        sample['prompt']   = prompt_formatted
        sample['response'] = resp
        sample['input_ids'] = tokenizer.encode(text)
        sample['query']     = tokenizer.decode(sample['input_ids'])
        for i in range(n):
            if type(rm_tokenizers[i]) != str:
                sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(text)
        return sample

    ds_chosen   = ds.map(tokenize, batched=False, fn_kwargs={'chosen': True},  num_proc=10)
    ds_rejected = ds.map(tokenize, batched=False, fn_kwargs={'chosen': False}, num_proc=10)
    ds_concat   = concatenate_datasets([ds_chosen, ds_rejected])
    ds_concat   = ds_concat.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
    print(ds_concat)
    drop = ['response_0', 'response_1', 'is_response_0_safe',
            'is_response_1_safe', 'better_response_id', 'safer_response_id']
    remove_columns = [c for c in drop if c in ds_concat.column_names]
    for i in range(n):
        if type(rm_tokenizers[i]) != str:
            ds_concat = ds_concat.filter(lambda x: 8 <= len(x['reward_ids{}'.format(1+i)]) <= 512)
            remove_columns.append('reward_ids{}'.format(1+i))
    ds_concat = ds_concat.remove_columns(remove_columns)

    queries_responses = [(q, r) for q, r in zip(ds_concat['prompt'], ds_concat['response'])]
    rewards_list = reward_models.get_reward_model_scores(queries_responses)
    for i in range(len(rewards_list)):
        ds_concat = ds_concat.add_column('score{}'.format(i+1), rewards_list[i])
    ds_concat.set_format(type='torch')
    return ds_concat


def add_score_summary(sample):
    sample['prompt_with_score'] = sample['prompt'].rstrip('### Response:') + ' '
    for i in range(instructions.num_rewards):
        sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(round(sample['score{}'.format(i+1)], 1)) + ' '
    sample['prompt_with_score'] += '### Response:'
    sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


def add_score_assistant(sample):
    sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' '
    for i in range(instructions.num_rewards):
        sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(round(sample['score{}'.format(i+1)], 1)) + ' '
    sample['prompt_with_score'] += '\n\nAssistant:'
    sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
    sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response']) + [tokenizer.eos_token_id]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


# add_score_beaver is identical to add_score_assistant (same format)
add_score_beaver = add_score_assistant


n = len(rm_tokenizers)
if script_args.exp_type == 'assistant':
    instructions = Instructions_n(n)
    train_data = build_dataset(gpu_id, tokenizer, rm_tokenizers, 'train')
elif script_args.exp_type == 'summary':
    instructions = Instructions_summary_n(n)
    train_data = build_dataset_summary(gpu_id, tokenizer, rm_tokenizers, 'train')
else:  # beaver
    instructions = Instructions_n(n)
    train_data = build_dataset_beaver(gpu_id, tokenizer, rm_tokenizers, 'train')


# normalize dataset and save information
if Accelerator().num_processes == 1:
    mean_reward_lis = []
    std_reward_lis  = []
    train_data.set_format()
    for i in range(reward_models.num_rewards):
        mean_reward = np.mean(train_data['score{}'.format(i+1)])
        std_reward  = np.std(train_data['score{}'.format(i+1)])
        mean_reward_lis.append(mean_reward)
        std_reward_lis.append(std_reward)
        norm_tensor = (np.array(train_data['score{}'.format(i+1)]) - mean_reward) / std_reward
        train_data = train_data.remove_columns('score{}'.format(i+1)).add_column('score{}'.format(i+1), list(norm_tensor))

    if script_args.exp_type == 'summary':
        add_score = add_score_summary
    elif script_args.exp_type == 'beaver':
        add_score = add_score_beaver
    else:
        add_score = add_score_assistant

    train_data = train_data.map(add_score, batched=False, num_proc=20)
    train_data.set_format(type="torch")
    train_data.save_to_disk(script_args.save_directory)
    print(np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
    np.save(script_args.save_directory + '/all_reward_stat.npy', np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
else:
    train_data.save_to_disk(script_args.save_directory + '/ind{}'.format(gpu_id))
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    if gpu_id == 0:
        train_data_all = concatenate_datasets([
            load_from_disk(script_args.save_directory + '/ind{}'.format(i))
            for i in range(accelerator.num_processes)
        ])
        mean_reward_lis = []
        std_reward_lis  = []
        train_data_all.set_format()
        for i in range(reward_models.num_rewards):
            mean_reward = np.mean(train_data_all['score{}'.format(i+1)])
            std_reward  = np.std(train_data_all['score{}'.format(i+1)])
            mean_reward_lis.append(mean_reward)
            std_reward_lis.append(std_reward)
            norm_tensor = (np.array(train_data_all['score{}'.format(i+1)]) - mean_reward) / std_reward
            train_data_all = train_data_all.remove_columns('score{}'.format(i+1)).add_column('score{}'.format(i+1), list(norm_tensor))

        if script_args.exp_type == 'summary':
            add_score = add_score_summary
        elif script_args.exp_type == 'beaver':
            add_score = add_score_beaver
        else:
            add_score = add_score_assistant

        train_data_all = train_data_all.map(add_score, batched=False, num_proc=20)
        train_data_all.set_format(type="torch")
        train_data_all.save_to_disk(script_args.save_directory)
        print(np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
        np.save(script_args.save_directory + '/all_reward_stat.npy', np.array([mean_reward_lis, std_reward_lis]).reshape(2, -1).T)
