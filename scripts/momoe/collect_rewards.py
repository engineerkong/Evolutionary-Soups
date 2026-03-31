"""Step 1: For each prompt in the training set, run inference with x sampled merging
weights and record the reward vectors. Results are saved as a CSV that becomes the
input to build_dataset.py.

Extends collect_rewards.py with blockwise (early/mid/late) layer-wise merging.

LoRA path  (--use_lora True):
  - Base model loaded ONCE per rank
  - Adapter weights interpolated in-place per weight combination → no disk I/O
  - No rank-0-only merge phase → barrier timeout issue eliminated

Disk path  (--use_lora False):
  - Original behaviour preserved exactly
"""
import gc
import os
import shutil
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
from peft import PeftModel

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_ppo, build_dataset_summary_ppo,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
    get_clean_data, load_main_tokenizer
)
from new_utils import (
    EARLY_FRAC, LATE_FRAC, get_simplex_samples,
    merge_and_save_weights, merge_and_save_weights_blockwise,
    load_lora_adapters, apply_merged_lora, apply_merged_lora_blockwise,
)

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    sft_model_name:     str       = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    exp_type:           str       = 'assistant'
    batch_size:         int       = 64
    split:              str       = 'train'
    block_mode:         str       = 'uniform'
    simplex_step:       float     = 0.1
    use_lora:           bool      = True    # True → in-memory LoRA swap
                                            # False → original disk merge
    do_sample:          bool      = True    # passed to generate(); if False, rewards will be deterministic but less smooth
    num_continuations:  int       = 3       # K continuations per (prompt, weight); rewards averaged
    save_directory:     str       = './results/new/'
    wandb_name:         str       = 'new_assistant'


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
SAMPLE_WEIGHTS = get_simplex_samples(n_objectives,
                                     step=script_args.simplex_step,
                                     block_mode=script_args.block_mode)

print(f'block_mode={script_args.block_mode} | '
      f'simplex_step={script_args.simplex_step} | '
      f'use_lora={script_args.use_lora} | '
      f'total combinations={len(SAMPLE_WEIGHTS)}')


# ---------------------------------------------------------------------------
# Dataset / dataloader setup
# ---------------------------------------------------------------------------

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

# Compute true shard start in the full dataset before sharding.
# HuggingFace shard(contiguous=True) splits N items into n shards where
# shard k starts at:  (N // n) * k + min(k, N % n)
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
    'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
    'min_length': -1,
    'top_k': 0,
    'top_p': 0.9,
    'temperature': 0.7,
    'do_sample': script_args.do_sample
}
print(f'do_sample={script_args.do_sample}  num_continuations={script_args.num_continuations if script_args.do_sample else 1}')

# ---------------------------------------------------------------------------
# Initialise shard CSV path and header-written flag here,
# before the inference loop, instead of accumulating all_rows.
# ---------------------------------------------------------------------------
shard_path = os.path.join(output_dir, f'collected_rewards_rank{process_id}.csv')
_csv_header_written = False


# ---------------------------------------------------------------------------
# Phase 1 — model setup
#
#   LoRA path  : every rank independently loads base + caches adapter dicts.
#                No inter-rank coordination → barrier timeout impossible.
#
#   Disk path  : rank 0 merges all combinations to disk; all ranks wait at
#                barrier (original behaviour, unchanged).
# ---------------------------------------------------------------------------

if script_args.use_lora:
    print(f'[Rank {process_id}] Loading base model ...')
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_name, torch_dtype=torch.bfloat16,
        device_map=f'cuda:{gpu_id}')
    base_model.resize_token_embeddings(len(tokenizer))

    # Wrap once with the first expert's LoRA config so the model is PEFT-aware
    base_model = PeftModel.from_pretrained(
        base_model, script_args.expert_model_paths[0], is_trainable=False)

    # Cache all adapter state dicts on CPU — only delta weights, not full models
    adapter_state_dicts = load_lora_adapters(base_model, script_args.expert_model_paths)
    n_layers = base_model.config.num_hidden_layers

    base_model = accelerator.prepare(base_model)
    # No wait_for_everyone() — each rank is fully independent here

else:
    # ── Original disk-based pre-merge (rank 0 only) ──────────────────────
    if process_id == 0:
        for sample in SAMPLE_WEIGHTS:
            if script_args.block_mode == 'uniform':
                weights_str = '_'.join(f'{w:.2f}' for w in sample)
                temp_path   = os.path.join(output_dir, f'temp_model_w{weights_str}')
                if os.path.exists(temp_path):
                    print(f'  Skipping (already exists): {temp_path}')
                else:
                    merge_and_save_weights(
                        expert_model_paths=script_args.expert_model_paths,
                        weights=sample, save_path=temp_path)
            else:
                early_w, mid_w, late_w = sample
                weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                               '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                               '_L' + '_'.join(f'{w:.2f}' for w in late_w))
                temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')
                if os.path.exists(temp_path):
                    print(f'  Skipping (already exists): {temp_path}')
                else:
                    merge_and_save_weights_blockwise(
                        expert_model_paths=script_args.expert_model_paths,
                        early_weights=early_w, mid_weights=mid_w, late_weights=late_w,
                        save_path=temp_path)
        print(f'\n[Rank 0] All {len(SAMPLE_WEIGHTS)} models merged and saved to disk.')
    accelerator.wait_for_everyone()     # only barrier in the whole script; disk path only


# ---------------------------------------------------------------------------
# Phase 2 — inference sweep
#
#   LoRA path : hot-swap adapter weights in-place before each inference pass.
#               Base model stays loaded; no disk reads, no re-prepare.
#
#   Disk path : load each pre-merged model from disk (original behaviour).
# ---------------------------------------------------------------------------

for sample in SAMPLE_WEIGHTS:

    if script_args.block_mode == 'uniform':
        early_w = mid_w = late_w = sample
        weights_str = '_'.join(f'{w:.2f}' for w in sample)
    else:
        early_w, mid_w, late_w = sample
        weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                       '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                       '_L' + '_'.join(f'{w:.2f}' for w in late_w))

    print(f'\n[Rank {process_id}] weights={weights_str}')

    if script_args.use_lora:
        # ── Hot-swap: interpolate adapters in-place, no disk I/O ─────────
        unwrapped = accelerator.unwrap_model(base_model)
        if script_args.block_mode == 'uniform':
            apply_merged_lora(unwrapped, adapter_state_dicts, sample)
        else:
            apply_merged_lora_blockwise(
                unwrapped, adapter_state_dicts,
                early_w, mid_w, late_w, n_layers)
        model = base_model      # alias — no reload, no re-prepare

    else:
        # ── Original: load pre-merged model from disk ─────────────────────
        temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')
        model = AutoModelForCausalLM.from_pretrained(
            temp_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
        model.resize_token_embeddings(len(tokenizer))
        model = accelerator.prepare(model)

    model.eval()

    # ── K-continuation reward averaging ──────────────────────────────────────
    # do_sample=True : run num_continuations independent stochastic passes and
    #                  average rewards (GRPO / RLOO style variance reduction).
    # do_sample=False: greedy decoding is deterministic — one pass is enough.
    # ─────────────────────────────────────────────────────────────────────────
    n_continuations = script_args.num_continuations if script_args.do_sample else 1

    # On the first pass we also record prompt texts (they don't change across continuations).
    all_prompts_decoded = None
    # accumulated_rewards: list of length N_prompts_this_shard, each element is a
    # list of K reward vectors (one per continuation).
    accumulated_rewards = None   # initialised after first pass

    for cont_idx in range(n_continuations):
        full_responses       = []
        full_prompts_decoded = []

        with torch.no_grad():
            for batch in tqdm(dataloader,
                              desc=f'{weights_str} cont={cont_idx+1}/{n_continuations}'):
                input_ids      = batch['input_ids'].to(f'cuda:{gpu_id}')
                attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
                outputs = accelerator.unwrap_model(model).generate(
                    input_ids, attention_mask=attention_mask, **generation_kwargs)
                full_responses.extend(tokenizer.batch_decode(outputs.cpu()))
                full_prompts_decoded.extend(tokenizer.batch_decode(input_ids.cpu()))
                del outputs, input_ids, attention_mask

        full_prompts_decoded, full_responses = get_clean_data(full_responses, full_prompts_decoded)

        # Record prompt texts once (same across all continuations)
        if all_prompts_decoded is None:
            all_prompts_decoded = full_prompts_decoded

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
        # rewards_list: list of n_objectives lists, each of length N_prompts

        n_prompts = len(all_prompts_decoded)
        if accumulated_rewards is None:
            accumulated_rewards = [[[] for _ in range(n_objectives)]
                                   for _ in range(n_prompts)]

        for idx in range(n_prompts):
            for k in range(n_objectives):
                accumulated_rewards[idx][k].append(rewards_list[k][idx])

        torch.cuda.empty_cache()

    # Average rewards across K continuations (full float precision)
    # ---------------------------------------------------------------------------
    # Build rows for this combination and append to CSV immediately,
    # instead of accumulating in all_rows for a single end-of-script write.
    # ---------------------------------------------------------------------------
    rows_this_combination = []
    for idx in range(len(all_prompts_decoded)):
        row = {'prompt_idx': shard_start + idx,
               'prompt_text': all_prompts_decoded[idx].replace('\r', '').replace('\n', ' ')}
        if script_args.block_mode == 'uniform':
            for k, w in enumerate(sample):
                row[f'w{k}'] = w
        else:
            for k, w in enumerate(early_w):
                row[f'w{k}_early'] = w
            for k, w in enumerate(mid_w):
                row[f'w{k}_mid'] = w
            for k, w in enumerate(late_w):
                row[f'w{k}_late'] = w
        for k, name in enumerate(reward_names):
            row[f'reward_{name}'] = float(np.mean(accumulated_rewards[idx][k]))
        rows_this_combination.append(row)

    df_chunk = pd.DataFrame(rows_this_combination)
    df_chunk.to_csv(shard_path, mode='a', index=False,
                    header=not _csv_header_written, escapechar='\\')
    _csv_header_written = True
    print(f'[Rank {process_id}] Appended {len(rows_this_combination)} rows → {shard_path}')
    # ---------------------------------------------------------------------------

    if not script_args.use_lora:
        # Disk path only — LoRA path reuses base_model across all iterations
        del model
        gc.collect()
        torch.cuda.empty_cache()


# # ---------------------------------------------------------------------------
# # Cleanup temp models — disk path only
# # ---------------------------------------------------------------------------

# if not script_args.use_lora and process_id == 0:
#     for sample in SAMPLE_WEIGHTS:
#         if script_args.block_mode == 'uniform':
#             weights_str = '_'.join(f'{w:.2f}' for w in sample)
#         else:
#             early_w, mid_w, late_w = sample
#             weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
#                            '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
#                            '_L' + '_'.join(f'{w:.2f}' for w in late_w))
#         shutil.rmtree(os.path.join(output_dir, f'temp_model_w{weights_str}'),
#                       ignore_errors=True)


# ---------------------------------------------------------------------------
# Removed the original one-shot pd.DataFrame(all_rows).to_csv(...)
# block here — shard CSV is already fully written by this point.
# ---------------------------------------------------------------------------
print(f'[Rank {process_id}] Shard complete → {shard_path}')

accelerator.wait_for_everyone()     # safe: all ranks reach here symmetrically

if process_id == 0:
    shards = [
        pd.read_csv(os.path.join(output_dir, f'collected_rewards_rank{rank}.csv'))
        for rank in range(accelerator.num_processes)
    ]
    if script_args.block_mode == 'uniform':
        weight_cols = [f'w{k}' for k in range(n_objectives)]
    else:
        weight_cols = ([f'w{k}_early' for k in range(n_objectives)] +
                       [f'w{k}_mid'   for k in range(n_objectives)] +
                       [f'w{k}_late'  for k in range(n_objectives)])

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