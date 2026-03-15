"""Step 1: For each prompt in the training set, run inference with 6 sampled merging
weights and record the reward vectors. Results are saved as a CSV that becomes the
input to build_dataset.py.

Output CSV columns:
    prompt_idx, t_value, reward_0, reward_1, ..., prompt_text
"""
import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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
    get_clean_data, load_main_tokenizer,
)
from new_utils import merge_and_save_weights, load_base_model

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}

# 6 sampled merging weights — avoid uniform spacing, favour middle range
SAMPLE_T_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class ScriptArguments:
    sft_model_name: str = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names: str = 'harmless,helpful'
    exp_type: str = 'assistant'
    save_directory: str = './results/new/data/'
    wandb_name: str = 'new_assistant'
    mini_batch_size: int = 64


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

tokenizer = load_main_tokenizer(script_args.sft_model_name)
tokenizer.padding_side = 'left'

if script_args.exp_type == 'assistant':
    dataset = build_dataset_ppo(
        'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers[0], split='train') # hardcoded for testing
    instructions = Instructions()
else:
    dataset = build_dataset_summary_ppo(
        'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()

# Shard across processes
if accelerator.num_processes > 1:
    dataset = dataset.shard(num_shards=accelerator.num_processes,
                            index=process_id, contiguous=True)

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=script_args.mini_batch_size,
                        drop_last=False, collate_fn=data_collator)

generation_kwargs = {
    'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
    'min_length': -1,
    'top_k': 0.0,
    'top_p': 0.9,
    'do_sample': False,   # greedy — eliminates sampling variance
}

all_rows = []
prompt_offset = process_id * (len(dataset))  # global prompt index offset

for t_val in SAMPLE_T_VALUES:
    print(f'\n[Rank {process_id}] Collecting rewards for t={t_val:.1f}')

    # Merge and save model for this t
    expert_weights = [t_val, 1.0 - t_val]
    temp_path = os.path.join(output_dir, f'temp_model_t{t_val:.2f}')

    if process_id == 0:
        merge_and_save_weights(script_args.expert_model_paths, expert_weights, temp_path)
    accelerator.wait_for_everyone()

    # Load merged model fresh (identical to eval_ppo_rs pattern)
    model = AutoModelForCausalLM.from_pretrained(
        temp_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
    model.resize_token_embeddings(len(tokenizer))
    model, dataloader_prepared = accelerator.prepare(model, dataloader)
    model.eval()

    full_responses = []
    full_prompts = []
    prompt_ids_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader_prepared, desc=f't={t_val}'):
            outputs = accelerator.unwrap_model(model).generate(
                batch['input_ids'], attention_mask=batch['attention_mask'],
                **generation_kwargs)
            full_responses.extend(outputs)
            full_prompts.extend(batch['input_ids'])

    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts_decoded = tokenizer.batch_decode(full_prompts)
    full_prompts_decoded, full_responses = get_clean_data(full_responses, full_prompts_decoded)

    queries_responses = [
        (instructions.get_input(r), instructions.get_response(r))
        for r in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(
            queries_responses, instructions.get_post, normalize_rewards=False)
    else:
        rewards_list = reward_models.get_reward_model_scores(
            queries_responses, normalize_rewards=False)

    # Gather across processes
    rewards_gathered = [accelerator.gather_for_metrics(r) for r in rewards_list]
    prompts_gathered = accelerator.gather_for_metrics(full_prompts_decoded)

    if process_id == 0:
        for idx in range(len(prompts_gathered)):
            row = {
                'prompt_idx': idx,
                't_value': t_val,
                'prompt_text': prompts_gathered[idx],
            }
            for k, name in enumerate(reward_names):
                row[f'reward_{name}'] = rewards_gathered[k][idx]
            all_rows.append(row)

    # Clean up merged model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        # Remove temp merged model to save disk space
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)

if process_id == 0:
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(output_dir, 'collected_rewards.csv')
    df.to_csv(out_path, index=False, escapechar='\\')
    print(f'\nRewards saved to {out_path}')
    print(f'Total rows: {len(df)} ({len(df) // len(SAMPLE_T_VALUES)} prompts × {len(SAMPLE_T_VALUES)} t values)')
    print(df.groupby('t_value')[
        [f'reward_{n}' for n in reward_names]].mean().round(4))
