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
from peft import PeftModel, set_peft_model_state_dict

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
# Simplex sampling — unchanged
# ---------------------------------------------------------------------------

def get_simplex_samples(
    n_objectives: int,
    step: float = 0.2,
    block_mode: str = 'uniform',
) -> List[Union[List[float], Tuple[List[float], List[float], List[float]]]]:
    """
    Generate weight samples on the simplex.

    uniform : returns List[List[float]]
    custom  : returns List[Tuple[List[float], List[float], List[float]]]
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
# Shared layer-classification helpers
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


def _block_weights_fn(key: str,
                      early_weights: List[float],
                      mid_weights:   List[float],
                      late_weights:  List[float],
                      early_end:     int,
                      late_start:    int) -> List[float]:
    """Return the weight vector that should govern tensor `key`."""
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


# ---------------------------------------------------------------------------
# Disk-based merge helpers  (used only when use_lora=False — original path)
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


def merge_and_save_weights_blockwise(
    expert_model_paths: List[str],
    early_weights: List[float],
    mid_weights:   List[float],
    late_weights:  List[float],
    save_path: str,
    early_frac: float = EARLY_FRAC,
    late_frac:  float = LATE_FRAC,
):
    """Blockwise disk merge — different weight vectors per layer block."""
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
          f"late {late_start}–{n_layers-1} {late_weights}")

    merged = {
        key: sum(
            _block_weights_fn(key, early_weights, mid_weights, late_weights,
                              early_end, late_start)[k] * state_dicts[k][key].float()
            for k in range(n_experts)
        )
        for key in state_dicts[0]
    }
    models[0].load_state_dict(merged)
    models[0].half()
    models[0].save_pretrained(save_path)
    AutoTokenizer.from_pretrained(expert_model_paths[0]).save_pretrained(save_path)
    print(f"  Saved blockwise-merged model → {save_path}")


# ---------------------------------------------------------------------------
# LoRA-based merge helpers  (used only when use_lora=True — new fast path)
# ---------------------------------------------------------------------------

def load_lora_adapters(base_model: PeftModel,
                       expert_model_paths: List[str]) -> List[dict]:
    """
    Load each expert's LoRA adapter into CPU memory as a plain state dict.
    The base model is never duplicated — only the tiny delta weights are kept.
    Called ONCE before the inference sweep.

    Loads adapter files directly from disk (safetensors or bin) so that
    base_model is never modified: the previous approach used PeftModel.from_pretrained
    + unload() which stripped LoRA modules from base_model in-place, causing
    set_peft_model_state_dict to silently no-op during inference.
    """
    import os
    adapter_state_dicts = []
    for path in expert_model_paths:
        sf_path  = os.path.join(path, 'adapter_model.safetensors')
        bin_path = os.path.join(path, 'adapter_model.bin')
        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            sd = {k: v.clone().cpu() for k, v in load_file(sf_path).items()}
        elif os.path.exists(bin_path):
            sd = {k: v.clone().cpu() for k, v in torch.load(bin_path, map_location='cpu').items()}
        else:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.bin found in {path}")
        adapter_state_dicts.append(sd)
        print(f"  Cached LoRA adapter: {path}")
    return adapter_state_dicts


def apply_merged_lora(
    peft_model: PeftModel,
    adapter_state_dicts: List[dict],
    weights: List[float],
) -> None:
    """
    Flat interpolation: merge adapter dicts with `weights` and hot-swap them
    onto `peft_model` in-place.  No disk I/O, no model reload.
    """
    assert abs(sum(weights) - 1.0) < 1e-6, f"weights must sum to 1, got {sum(weights)}"
    merged = {
        key: sum(weights[k] * adapter_state_dicts[k][key].float()
                 for k in range(len(weights)))
        for key in adapter_state_dicts[0]
    }
    set_peft_model_state_dict(peft_model,
                              {k: v.to(peft_model.device) for k, v in merged.items()})
    del merged
    gc.collect()


def apply_merged_lora_blockwise(
    peft_model: PeftModel,
    adapter_state_dicts: List[dict],
    early_weights: List[float],
    mid_weights:   List[float],
    late_weights:  List[float],
    n_layers: int,
    early_frac: float = EARLY_FRAC,
    late_frac:  float = LATE_FRAC,
) -> None:
    """
    Blockwise interpolation: different weight vectors for early / mid / late
    LoRA adapter keys.  Hot-swapped in-place, no disk I/O.
    """
    early_end  = int(n_layers * early_frac)
    late_start = n_layers - int(n_layers * late_frac)

    merged = {
        key: sum(
            _block_weights_fn(key, early_weights, mid_weights, late_weights,
                              early_end, late_start)[k]
            * adapter_state_dicts[k][key].float()
            for k in range(len(adapter_state_dicts))
        )
        for key in adapter_state_dicts[0]
    }
    set_peft_model_state_dict(peft_model,
                              {k: v.to(peft_model.device) for k, v in merged.items()})
    del merged
    gc.collect()


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
    split:              str       = 'train'
    block_mode:         str       = 'uniform'
    simplex_step:       float     = 0.2
    use_lora:           bool      = True    # True → in-memory LoRA swap (recommended)
                                            # False → original disk merge


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
      f'use_lora={script_args.use_lora} | '
      f'total combinations={len(SAMPLE_WEIGHTS)}')


# ---------------------------------------------------------------------------
# Dataset / dataloader setup — unchanged
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
                merge_and_save_weights(
                    expert_model_paths=script_args.expert_model_paths,
                    weights=sample, save_path=temp_path)
            else:
                early_w, mid_w, late_w = sample
                weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                               '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                               '_L' + '_'.join(f'{w:.2f}' for w in late_w))
                temp_path = os.path.join(output_dir, f'temp_model_w{weights_str}')
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

    full_responses       = []
    full_prompts_decoded = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=weights_str):
            input_ids      = batch['input_ids'].to(f'cuda:{gpu_id}')
            attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            outputs = accelerator.unwrap_model(model).generate(
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

    if not script_args.use_lora:
        # Disk path only — LoRA path reuses base_model across all iterations
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Cleanup temp models — disk path only
# ---------------------------------------------------------------------------

if not script_args.use_lora and process_id == 0:
    for sample in SAMPLE_WEIGHTS:
        if script_args.block_mode == 'uniform':
            weights_str = '_'.join(f'{w:.2f}' for w in sample)
        else:
            early_w, mid_w, late_w = sample
            weights_str = ('E' + '_'.join(f'{w:.2f}' for w in early_w) +
                           '_M' + '_'.join(f'{w:.2f}' for w in mid_w) +
                           '_L' + '_'.join(f'{w:.2f}' for w in late_w))
        shutil.rmtree(os.path.join(output_dir, f'temp_model_w{weights_str}'),
                      ignore_errors=True)


# ---------------------------------------------------------------------------
# Save shards and merge — unchanged
# ---------------------------------------------------------------------------

shard_path = os.path.join(output_dir, f'collected_rewards_rank{process_id}.csv')
pd.DataFrame(all_rows).to_csv(shard_path, index=False, escapechar='\\')
print(f'[Rank {process_id}] Saved {len(all_rows)} rows → {shard_path}')

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