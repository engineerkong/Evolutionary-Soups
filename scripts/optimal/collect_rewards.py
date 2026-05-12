"""Step 1: For each prompt in the training/test set, run inference with every
merging weight on the simplex and record the reward vectors.  Results are saved
as a CSV that becomes the input to build_dataset.py.

Rank 0 pre-merges all weight combinations to disk; all ranks load from disk.
"""
import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import datetime

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_ppo, build_dataset_summary_ppo,
    build_dataset_eval, build_dataset_summary_eval,
    build_dataset_beaver_ppo, build_dataset_beaver_eval,
    get_clean_data, load_main_tokenizer
)
from optimal_utils import get_simplex_samples, merge_lora_and_save

REWARD_PATHS = {
    'harmless':     'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':      'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':      'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':      'Tristan/gpt2_reward_summarization',
    'faithful':     'CogComp/bart-faithful-summary-detector',
    'humor':        'mohameddhiab/humor-no-humor',
    'beaver_reward':'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':  'PKU-Alignment/beaver-7b-v1.0-cost',
}


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:    str       = 'meta-llama/Llama-2-7b-hf'
    sft_model_name:     str       = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    exp_type:           str       = 'assistant'
    batch_size:         int       = 64
    split:              str       = 'test'
    simplex_step:       float     = 0.1
    do_sample:          bool      = False
    save_directory:     str       = './results/optimal/'
    wandb_name:         str       = 'optimal_assistant'


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir  = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id

reward_names       = [x.strip() for x in script_args.reward_names.split(',')]
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

n_objectives   = len(reward_names)
SAMPLE_WEIGHTS = get_simplex_samples(n_objectives, step=script_args.simplex_step)

print(f'simplex_step={script_args.simplex_step} | total combinations={len(SAMPLE_WEIGHTS)}')


# ---------------------------------------------------------------------------
# Dataset / dataloader
# ---------------------------------------------------------------------------

tokenizer = load_main_tokenizer(script_args.expert_model_paths[0])
tokenizer.padding_side = 'left'

if script_args.exp_type == 'assistant':
    if script_args.split == 'test':
        dataset = build_dataset_eval(
            'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_ppo(
            'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.exp_type == 'beaver':
    if script_args.split == 'test':
        dataset = build_dataset_beaver_eval(
            'PKU-Alignment/PKU-SafeRLHF-10K', tokenizer, reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_beaver_ppo(
            'PKU-Alignment/PKU-SafeRLHF-10K', tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
else:
    if script_args.split == 'test':
        dataset = build_dataset_summary_eval(
            'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_summary_ppo(
            'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()

_full_size = len(dataset)
if accelerator.num_processes > 1:
    _div, _mod = divmod(_full_size, accelerator.num_processes)
    shard_start = _div * process_id + min(process_id, _mod)
    print(f'[Rank {process_id}] Full dataset size: {_full_size} | Shard start index: {shard_start}')
    dataset = dataset.shard(num_shards=accelerator.num_processes,
                            index=process_id, contiguous=True)
else:
    shard_start = 0

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader    = DataLoader(dataset, batch_size=script_args.batch_size,
                           drop_last=False, collate_fn=data_collator)

generation_kwargs = {
    'max_new_tokens': 48 if script_args.exp_type == 'summary' else 128,
    'do_sample': script_args.do_sample
}
print(f'do_sample={script_args.do_sample}')

shard_path          = os.path.join(output_dir, f'collected_rewards_rank{process_id}.csv')
_csv_header_written = os.path.exists(shard_path)


# ---------------------------------------------------------------------------
# Phase 1 — rank 0 pre-merges all combinations to disk
# ---------------------------------------------------------------------------

if process_id == 0:
    for sample in SAMPLE_WEIGHTS:
        weights_str = '_'.join(f'{w:.2f}' for w in sample)
        temp_path   = os.path.join(output_dir, f'temp_model_w{weights_str}')
        if os.path.exists(temp_path):
            print(f'  Skipping (already exists): {temp_path}')
        else:
            merge_lora_and_save(
                base_model_name=script_args.base_model_name,
                expert_model_paths=script_args.expert_model_paths,
                weights=sample, save_path=temp_path)
    print(f'\n[Rank 0] All {len(SAMPLE_WEIGHTS)} models merged and saved to disk.')
accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Phase 2 — inference sweep
# ---------------------------------------------------------------------------

for sample in SAMPLE_WEIGHTS:
    weights_str = '_'.join(f'{w:.2f}' for w in sample)
    print(f'\n[Rank {process_id}] weights={weights_str}')

    temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')
    model = AutoModelForCausalLM.from_pretrained(
        temp_path, torch_dtype=torch.bfloat16, device_map=f'cuda:{gpu_id}')
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    full_responses       = []
    full_prompts_decoded = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=weights_str):
            input_ids      = batch['input_ids'].to(f'cuda:{gpu_id}')
            attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            outputs = model.generate(
                input_ids, attention_mask=attention_mask, **generation_kwargs)
            full_responses.extend(tokenizer.batch_decode(outputs.cpu()))
            full_prompts_decoded.extend(tokenizer.batch_decode(input_ids.cpu()))
            del outputs, input_ids, attention_mask

    full_prompts_decoded, full_responses = get_clean_data(full_responses, full_prompts_decoded)

    queries_responses = [
        (instructions.get_input(r), instructions.get_response(r))
        for r in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(
            queries_responses, instructions.get_post,
            normalize_rewards=False, round_digits=None)
    else:
        rewards_list = reward_models.get_reward_model_scores(
            queries_responses, normalize_rewards=False, round_digits=None)

    torch.cuda.empty_cache()

    rows_this_combination = []
    for idx in range(len(full_prompts_decoded)):
        row = {'prompt_idx':  shard_start + idx,
               'prompt_text': full_prompts_decoded[idx].replace('\r', '').replace('\n', ' ')}
        for k, w in enumerate(sample):
            row[f'w{k}'] = w
        for k, name in enumerate(reward_names):
            row[f'reward_{name}'] = float(rewards_list[k][idx])
        rows_this_combination.append(row)

    df_chunk = pd.DataFrame(rows_this_combination)
    df_chunk.to_csv(shard_path, mode='a', index=False,
                    header=not _csv_header_written, escapechar='\\')
    _csv_header_written = True
    print(f'[Rank {process_id}] Appended {len(rows_this_combination)} rows → {shard_path}')

    del model
    gc.collect()
    torch.cuda.empty_cache()


print(f'[Rank {process_id}] Shard complete → {shard_path}')

accelerator.wait_for_everyone()

if process_id == 0:
    weight_cols = [f'w{k}' for k in range(n_objectives)]
    shards = [
        pd.read_csv(os.path.join(output_dir, f'collected_rewards_rank{rank}.csv'))
        for rank in range(accelerator.num_processes)
    ]
    df = pd.concat(shards, ignore_index=True).sort_values(
        weight_cols + ['prompt_idx']).reset_index(drop=True)
    out_path = os.path.join(output_dir, 'collected_rewards.csv')
    df.to_csv(out_path, index=False, escapechar='\\')
    for rank in range(accelerator.num_processes):
        os.remove(os.path.join(output_dir, f'collected_rewards_rank{rank}.csv'))
    print(f'\nRewards saved → {out_path}')
    print(f'Total rows: {len(df)} | '
          f'{df["prompt_idx"].nunique()} prompts × {len(SAMPLE_WEIGHTS)} weight combos')
    print(df.groupby(weight_cols)[[f'reward_{n}' for n in reward_names]].mean().round(4))
