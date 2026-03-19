"""Step 1: For each prompt in the training set, run inference with x sampled merging
weights and record the reward vectors. Results are saved as a CSV that becomes the
input to build_dataset.py.

Extends collect_rewards.py with blockwise (early/mid/late) layer-wise merging.

Output CSV columns:
    prompt_idx, w0, w1, ..., reward_0, reward_1, ..., prompt_text   (uniform mode)
    prompt_idx, w0_early, w1_early, w0_mid, w1_mid, w0_late, w1_late,
    reward_0, reward_1, ..., prompt_text                             (custom mode)

Block weight strategy (--block_mode):
  'uniform' — one simplex point applied to all blocks (same as original collect_rewards.py)
  'custom'  — three independent simplex points sampled for early / mid / late blocks;
              all combinations are enumerated (use --simplex_step 0.5 to limit combos)
"""
import gc
import os
import shutil
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

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

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}

EARLY_FRAC = 1 / 3
LATE_FRAC  = 1 / 3


# ---------------------------------------------------------------------------
# Simplex sampling — uniform and custom (blockwise) modes
# ---------------------------------------------------------------------------

def get_simplex_samples(
    n_objectives: int,
    step: float = 0.2,
    block_mode: str = 'uniform',
) -> List[Union[List[float], Tuple[List[float], List[float], List[float]]]]:
    """
    Generate weight samples on the simplex.

    uniform : returns List[List[float]]
                each entry is one weight vector applied to all three blocks.
                Identical to the original collect_rewards.py behaviour.

    custom  : returns List[Tuple[List[float], List[float], List[float]]]
                each entry is (early_w, mid_w, late_w) — three independent
                simplex points, one per block. All combinations are enumerated.
                With n_objectives=2, step=0.5 → 3^3 = 27 combos.
                With n_objectives=2, step=0.2 → 6^3 = 216 combos.
    """
    steps = round(1.0 / step)
    vals  = [round(i * step, 8) for i in range(steps + 1)]
    base  = [
        list(combo)
        for combo in product(vals, repeat=n_objectives)
        if abs(sum(combo) - 1.0) < 1e-6
    ]

    if block_mode == 'uniform':
        return base

    if block_mode == 'custom':
        return list(product(base, base, base))

    raise ValueError(f"Unknown block_mode '{block_mode}'. Choose: uniform | custom")


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_and_save_weights(
    expert_model_paths: List[str],
    weights: List[float],
    save_path: str,
):
    """Flat merge — same weight vector applied to every tensor (uniform mode)."""
    n_experts = len(expert_model_paths)
    assert len(weights) == n_experts
    assert abs(sum(weights) - 1.0) < 1e-6

    print(f"  Loading {n_experts} expert models for flat merge...")
    models      = [AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
                   for p in expert_model_paths]
    state_dicts = [m.state_dict() for m in models]

    merged = {
        key: sum(weights[k] * state_dicts[k][key].float() for k in range(n_experts))
        for key in state_dicts[0]
    }

    models[0].load_state_dict(merged)
    models[0].half()
    models[0].save_pretrained(save_path)
    AutoTokenizer.from_pretrained(expert_model_paths[0]).save_pretrained(save_path)
    print(f"  Saved flat-merged model → {save_path}")


# ---------------------------------------------------------------------------
# Blockwise merge helpers
# ---------------------------------------------------------------------------

def _get_layer_idx(key: str) -> Optional[int]:
    """Extract layer index from keys like 'model.layers.7.self_attn.q_proj.weight'."""
    parts = key.split('.')
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def _is_head_tensor(key: str) -> bool:
    return any(k in key for k in ('lm_head', 'model.norm'))


def _is_embed_tensor(key: str) -> bool:
    return 'embed_tokens' in key


def merge_and_save_weights_blockwise(
    expert_model_paths: List[str],
    early_weights: List[float],
    mid_weights:   List[float],
    late_weights:  List[float],
    save_path: str,
    early_frac: float = EARLY_FRAC,
    late_frac:  float = LATE_FRAC,
):
    """
    Merge expert models with different weights per layer block.

    Tensor assignment:
      embed_tokens                     → early_weights  (input side)
      layers 0 .. early_end-1          → early_weights
      layers early_end .. late_start-1 → mid_weights
      layers late_start .. n_layers-1  → late_weights
      model.norm + lm_head             → late_weights   (output side)
    """
    n_experts = len(expert_model_paths)
    for name, w in [('early', early_weights), ('mid', mid_weights), ('late', late_weights)]:
        assert len(w) == n_experts, \
            f"{name}_weights has {len(w)} entries but there are {n_experts} experts"
        assert abs(sum(w) - 1.0) < 1e-6, \
            f"{name}_weights must sum to 1.0, got {sum(w):.6f}"

    print(f"  Loading {n_experts} expert models for blockwise merge...")
    models      = [AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32)
                   for p in expert_model_paths]
    state_dicts = [m.state_dict() for m in models]

    n_layers   = models[0].config.num_hidden_layers
    early_end  = int(n_layers * early_frac)
    late_start = n_layers - int(n_layers * late_frac)

    print(f"  Layers: {n_layers} total | "
          f"early 0–{early_end-1} {early_weights} | "
          f"mid {early_end}–{late_start-1} {mid_weights} | "
          f"late {late_start}–{n_layers-1} {late_weights} | "
          f"embed {early_weights} | head/norm {late_weights}")

    def _block_weights(key: str) -> List[float]:
        if _is_embed_tensor(key):
            return early_weights
        if _is_head_tensor(key):
            return late_weights
        idx = _get_layer_idx(key)
        if idx is None:
            return mid_weights
        if idx < early_end:
            return early_weights
        if idx >= late_start:
            return late_weights
        return mid_weights

    merged = {
        key: sum(_block_weights(key)[k] * state_dicts[k][key].float()
                 for k in range(n_experts))
        for key in state_dicts[0]
    }

    models[0].load_state_dict(merged)
    models[0].half()
    models[0].save_pretrained(save_path)
    AutoTokenizer.from_pretrained(expert_model_paths[0]).save_pretrained(save_path)
    print(f"  Saved blockwise-merged model → {save_path}")


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    sft_model_name:     str       = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    exp_type:           str       = 'assistant'
    save_directory:     str       = './results/new/'
    wandb_name:         str       = 'new_assistant'
    batch_size:         int       = 64
    split:              str       = 'train'    # 'train' | 'test'
    block_mode:         str       = 'uniform'  # 'uniform' | 'custom'
    simplex_step:       float     = 0.2        # coarsen to 0.5 when block_mode='custom'


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir  = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
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
      f'total combinations={len(SAMPLE_WEIGHTS)}')

# ---------------------------------------------------------------------------
# Dataset / dataloader setup  (unchanged from collect_rewards.py)
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

if accelerator.num_processes > 1:
    dataset = dataset.shard(num_shards=accelerator.num_processes,
                            index=process_id, contiguous=True)

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataloader    = DataLoader(dataset, batch_size=script_args.batch_size,
                           drop_last=True, collate_fn=data_collator)

generation_kwargs = {
    'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
    'min_length': -1,
    'top_k': 0.0,
    'top_p': 0.9,
    'do_sample': False,
}

all_rows = []
num_prompts_per_process = (len(dataset) // script_args.batch_size) * script_args.batch_size

# ---------------------------------------------------------------------------
# Phase 1: pre-merge all weight combinations to disk (rank 0, CPU only)
# ---------------------------------------------------------------------------

if process_id == 0:
    for sample in SAMPLE_WEIGHTS:
        if script_args.block_mode == 'uniform':
            weights_str = '_'.join(f'{w:.2f}' for w in sample)
            temp_path   = os.path.join(output_dir, f'temp_model_w{weights_str}')
            merge_and_save_weights(
                expert_model_paths = script_args.expert_model_paths,
                weights            = sample,
                save_path          = temp_path,
            )
        else:
            early_w, mid_w, late_w = sample
            weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                           '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                           '_L' + '_'.join(f'{w:.2f}' for w in late_w))
            temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')
            merge_and_save_weights_blockwise(
                expert_model_paths = script_args.expert_model_paths,
                early_weights      = early_w,
                mid_weights        = mid_w,
                late_weights       = late_w,
                save_path          = temp_path,
            )
    print(f'\n[Rank 0] All {len(SAMPLE_WEIGHTS)} models merged and saved to disk.')

accelerator.wait_for_everyone()

# ---------------------------------------------------------------------------
# Phase 2: inference sweep — load each pre-merged model and collect rewards
# ---------------------------------------------------------------------------

for sample in SAMPLE_WEIGHTS:

    if script_args.block_mode == 'uniform':
        early_w = mid_w = late_w = sample
        weights_str = '_'.join(f'{w:.2f}' for w in sample)
    else:  # custom
        early_w, mid_w, late_w = sample
        weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                       '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                       '_L' + '_'.join(f'{w:.2f}' for w in late_w))

    print(f'\n[Rank {process_id}] block_mode={script_args.block_mode} | '
          f'early={early_w} mid={mid_w} late={late_w}')

    temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')

    model = AutoModelForCausalLM.from_pretrained(
        temp_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
    model.resize_token_embeddings(len(tokenizer))
    model = accelerator.prepare(model)
    model.eval()

    full_responses      = []
    full_prompts_decoded = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=weights_str):
            input_ids      = batch['input_ids'].to(f'cuda:{gpu_id}')
            attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            outputs = accelerator.unwrap_model(model).generate(
                input_ids, attention_mask=attention_mask,
                **generation_kwargs)
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
            queries_responses, instructions.get_post, normalize_rewards=False)
    else:
        rewards_list = reward_models.get_reward_model_scores(
            queries_responses, normalize_rewards=False)

    shard_start = process_id * num_prompts_per_process
    for idx in range(len(full_prompts_decoded)):
        row = {'prompt_idx': shard_start + idx, 'prompt_text': full_prompts_decoded[idx]}
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
            row[f'reward_{name}'] = rewards_list[k][idx]
        all_rows.append(row)

    del model
    gc.collect()
    torch.cuda.empty_cache()

# Clean up all temp models after inference is complete
if process_id == 0:
    for sample in SAMPLE_WEIGHTS:
        if script_args.block_mode == 'uniform':
            weights_str = '_'.join(f'{w:.2f}' for w in sample)
        else:
            early_w, mid_w, late_w = sample
            weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                           '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                           '_L' + '_'.join(f'{w:.2f}' for w in late_w))
        shutil.rmtree(os.path.join(output_dir, f'temp_model_w{weights_str}'), ignore_errors=True)

# ---------------------------------------------------------------------------
# Save shards and merge  (unchanged from collect_rewards.py)
# ---------------------------------------------------------------------------

shard_path = os.path.join(output_dir, f'collected_rewards_rank{process_id}.csv')
pd.DataFrame(all_rows).to_csv(shard_path, index=False, escapechar='\\')
print(f'[Rank {process_id}] Saved {len(all_rows)} rows → {shard_path}')

accelerator.wait_for_everyone()

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
    print(f'Total rows: {len(df)} '
          f'({df["prompt_idx"].nunique()} prompts × {len(SAMPLE_WEIGHTS)} weight combinations)')
    print(df.groupby(weight_cols)[[f'reward_{n}' for n in reward_names]].mean().round(4))