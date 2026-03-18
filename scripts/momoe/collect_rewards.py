"""Step 1: For each prompt in the training set, run inference with x sampled merging
weights and record the reward vectors. Results are saved as a CSV that becomes the
input to build_dataset.py.

Output CSV columns:
    prompt_idx, w0, w1, ..., reward_0, reward_1, ..., prompt_text
"""
import gc
import os
import sys
from dataclasses import dataclass, field
from itertools import product
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
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
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


def get_simplex_samples(n_objectives: int, step: float = 0.2) -> List[List[float]]:
    """Generate all weight vectors on a uniform simplex grid.

    Works for any number of objectives:
      - n_objectives=2, step=0.2 → 6 vectors
      - n_objectives=3, step=0.2 → 21 vectors
    """
    steps = round(1.0 / step)
    vals = [round(i * step, 8) for i in range(steps + 1)]
    samples = []
    for combo in product(vals, repeat=n_objectives):
        if abs(sum(combo) - 1.0) < 1e-6:
            samples.append(list(combo))
    return samples


@dataclass
class ScriptArguments:
    sft_model_name: str = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names: str = 'harmless,helpful'
    exp_type: str = 'assistant'
    save_directory: str = './results/new/'
    wandb_name: str = 'new_assistant'
    batch_size: int = 64
    split: str = 'train'  # 'train' or 'test'


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

# Derive simplex samples from the number of reward/expert models
n_objectives = len(reward_names)
SAMPLE_WEIGHTS = get_simplex_samples(n_objectives, step=0.2)

tokenizer = load_main_tokenizer(script_args.sft_model_name)
tokenizer.padding_side = 'left'

if script_args.exp_type == 'assistant':
    if script_args.split == 'test':
        dataset = build_dataset_eval_ppo(
            'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_ppo(
            'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
else:
    if script_args.split == 'test':
        dataset = build_dataset_summary_eval_ppo(
            'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers, split='test')
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
# drop_last=True: avoids gather_for_metrics padding duplicates when
# dataset_size % (num_gpus * batch_size) != 0
dataloader = DataLoader(dataset, batch_size=script_args.batch_size,
                        drop_last=True, collate_fn=data_collator)

generation_kwargs = {
    'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
    'min_length': -1,
    'top_k': 0.0,
    'top_p': 0.9,
    'do_sample': False,   # greedy — eliminates sampling variance
}

all_rows = []
# Number of complete batches × batch_size = exact prompts evaluated per process
# (drop_last=True guarantees no padding duplicates)
num_prompts_per_process = (len(dataset) // script_args.batch_size) * script_args.batch_size

for weights in SAMPLE_WEIGHTS:
    weights_str = '_'.join(f'{w:.2f}' for w in weights)
    print(f'\n[Rank {process_id}] Collecting rewards for weights={weights}')

    temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')

    if process_id == 0:
        merge_and_save_weights(script_args.expert_model_paths, weights, temp_path)
    accelerator.wait_for_everyone()

    # Load merged model fresh (identical to eval_ppo_rs pattern)
    model = AutoModelForCausalLM.from_pretrained(
        temp_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
    model.resize_token_embeddings(len(tokenizer))
    # Only prepare model — dataloader is already sharded via dataset.shard() above
    model = accelerator.prepare(model)
    model.eval()

    full_responses      = []
    full_prompts_decoded = []    # decode immediately, no tensor accumulation

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'weights={weights}'):
            input_ids      = batch['input_ids'].to(f'cuda:{gpu_id}')
            attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            outputs = accelerator.unwrap_model(model).generate(
                input_ids, attention_mask=attention_mask,
                **generation_kwargs)
            full_responses.extend(outputs.cpu())
            full_prompts_decoded.extend(tokenizer.batch_decode(input_ids.cpu()))

    full_responses = tokenizer.batch_decode(full_responses)
    # full_prompts_decoded already decoded above — no second decode needed
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

    # Each process saves its own shard — no gather needed, no duplication risk
    shard_start = process_id * num_prompts_per_process
    for idx in range(len(full_prompts_decoded)):
        row = {
            'prompt_idx': shard_start + idx,
            'prompt_text': full_prompts_decoded[idx],
        }
        # Store each weight component as its own column (w0, w1, ...)
        for k, w in enumerate(weights):
            row[f'w{k}'] = w
        for k, name in enumerate(reward_names):
            row[f'reward_{name}'] = rewards_list[k][idx]
        all_rows.append(row)

    # Clean up merged model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        # Remove temp merged model to save disk space
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)

# Save per-rank shard
shard_path = os.path.join(output_dir, f'collected_rewards_rank{process_id}.csv')
pd.DataFrame(all_rows).to_csv(shard_path, index=False, escapechar='\\')
print(f'[Rank {process_id}] Saved {len(all_rows)} rows to {shard_path}')

accelerator.wait_for_everyone()

# Rank 0 merges all shards into final CSV
if process_id == 0:
    shards = []
    for rank in range(accelerator.num_processes):
        p = os.path.join(output_dir, f'collected_rewards_rank{rank}.csv')
        shards.append(pd.read_csv(p))
    weight_cols = [f'w{k}' for k in range(n_objectives)]
    df = pd.concat(shards, ignore_index=True).sort_values(
        weight_cols + ['prompt_idx']).reset_index(drop=True)
    out_path = os.path.join(output_dir, 'collected_rewards.csv')
    df.to_csv(out_path, index=False, escapechar='\\')
    # Remove shards
    for rank in range(accelerator.num_processes):
        os.remove(os.path.join(output_dir, f'collected_rewards_rank{rank}.csv'))
    print(f'\nRewards saved to {out_path}')
    print(f'Total rows: {len(df)} ({df["prompt_idx"].nunique()} prompts × {len(SAMPLE_WEIGHTS)} weight vectors)')
    print(df.groupby(weight_cols)[[f'reward_{n}' for n in reward_names]].mean().round(4))