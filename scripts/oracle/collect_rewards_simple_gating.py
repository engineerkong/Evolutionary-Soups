"""Step 1 (SimpleMoEForCausalLM variant): For each prompt, run inference with every
fixed gating coefficient on the simplex using SimpleMoEForCausalLM and record reward
vectors. Output CSV has the same schema as collect_rewards.py so build_dataset.py
can consume it directly.

Unlike collect_rewards.py (which pre-merges LoRA adapters), this script keeps both
expert models in memory and merges their FINAL hidden states at inference time.
The gating coefficient w0 here is the weight on expert[0]'s final hidden state.
"""
import gc
import os
import sys
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir.parent / 'evolutionary'))   # es_architecture / es_utils 

from scripts.baselines.utils.multi_reward_models import RewardModels
from scripts.baselines.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_ppo, build_dataset_summary_ppo,
    build_dataset_eval, build_dataset_summary_eval,
    build_dataset_beaver_ppo, build_dataset_beaver_eval,
    get_clean_data, load_main_tokenizer,
)
from es_architecture import SimpleMoEForCausalLM
from es_utils import REWARD_PATHS
from optimal_utils import get_simplex_samples


# ---------------------------------------------------------------------------
# Fixed gating shim (identical to nsgaii_test.py FixedGating)
# ---------------------------------------------------------------------------

class FixedGating(torch.nn.Module):
    def __init__(self, coeffs: List[float]):
        super().__init__()
        self.register_buffer('_c', torch.tensor(coeffs, dtype=torch.float32))
        self.fixed_alpha = 1.0

    def alpha_floats(self): return [1.0] * 999

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._c.unsqueeze(0).expand(hidden_states.shape[0], -1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:    str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    exp_type:           str       = 'assistant'
    batch_size:         int       = 64
    split:              str       = 'test'
    simplex_step:       float     = 0.1
    do_sample:          bool      = False
    save_directory:     str       = './results/optimal/'
    run_name:           str       = 'optimal_beaver_simple'


parser      = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir  = os.path.join(script_args.save_directory, script_args.run_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
torch.distributed.init_process_group(
    backend='nccl', timeout=datetime.timedelta(minutes=120))
accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id
device      = f'cuda:{gpu_id}'

reward_names       = [x.strip() for x in script_args.reward_names.split(',')]
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

SAMPLE_WEIGHTS = get_simplex_samples(len(reward_names), step=script_args.simplex_step)
print(f'simplex_step={script_args.simplex_step} | total combinations={len(SAMPLE_WEIGHTS)}')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

tokenizer              = load_main_tokenizer(script_args.expert_model_paths[0])
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
            'PKU-Alignment/PKU-SafeRLHF-10K', tokenizer,
            reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_beaver_ppo(
            'PKU-Alignment/PKU-SafeRLHF-10K', tokenizer,
            reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
else:
    if script_args.split == 'test':
        dataset = build_dataset_summary_eval(
            'openai/summarize_from_feedback', tokenizer,
            reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_summary_ppo(
            'openai/summarize_from_feedback', tokenizer,
            reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()

_full_size = len(dataset)
if accelerator.num_processes > 1:
    _div, _mod = divmod(_full_size, accelerator.num_processes)
    shard_start = _div * process_id + min(process_id, _mod)
    dataset = dataset.shard(num_shards=accelerator.num_processes,
                            index=process_id, contiguous=True)
    print(f'[Rank {process_id}] {_full_size} total → shard {len(dataset)} prompts '
          f'(start idx {shard_start})', flush=True)
else:
    shard_start = 0

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

dataloader = DataLoader(dataset, batch_size=script_args.batch_size,
                        drop_last=False,
                        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))

generation_kwargs = {
    'max_new_tokens': 48 if script_args.exp_type == 'summary' else 128,
    'do_sample': script_args.do_sample,
}


# ---------------------------------------------------------------------------
# Load expert models ONCE (kept in memory for all gating sweeps)
# ---------------------------------------------------------------------------

print(f'[Rank {process_id}] Loading {len(script_args.expert_model_paths)} expert models …',
      flush=True)
experts = []
for i, path in enumerate(script_args.expert_model_paths):
    base = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.resize_token_embeddings(len(tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    experts.append(m)
    print(f'  [{i}] {path}', flush=True)

print(f'[Rank {process_id}] Experts ready. Starting sweep …', flush=True)


# ---------------------------------------------------------------------------
# Sweep over simplex weights
# ---------------------------------------------------------------------------

shard_path          = os.path.join(output_dir, f'collected_rewards_rank{process_id}.csv')
_csv_header_written = os.path.exists(shard_path)

# Track completed weight combos so we can resume
done_weights = set()
if _csv_header_written:
    try:
        _df_done = pd.read_csv(shard_path)
        weight_cols_done = [f'w{k}' for k in range(len(reward_names))]
        for _, row in _df_done[weight_cols_done].drop_duplicates().iterrows():
            done_weights.add(tuple(round(row[c], 8) for c in weight_cols_done))
        print(f'[Rank {process_id}] Resuming — {len(done_weights)} combos already done',
              flush=True)
    except Exception:
        pass

for sample in SAMPLE_WEIGHTS:
    key = tuple(round(w, 8) for w in sample)
    if key in done_weights:
        print(f'[Rank {process_id}] skip (cached): {sample}', flush=True)
        continue

    weights_str = '_'.join(f'{w:.2f}' for w in sample)
    print(f'\n[Rank {process_id}] weights={weights_str}', flush=True)

    gating = FixedGating(list(sample)).to(device)
    model  = SimpleMoEForCausalLM(experts, gating)
    model.eval()

    full_responses       = []
    full_prompts_decoded = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=weights_str, disable=(process_id != 0)):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.generate(
                input_ids, attention_mask=attention_mask, **generation_kwargs)
            full_responses.extend(tokenizer.batch_decode(outputs.cpu()))
            full_prompts_decoded.extend(tokenizer.batch_decode(input_ids.cpu()))
            del outputs, input_ids, attention_mask
            torch.cuda.empty_cache()

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

    rows = []
    for idx in range(len(full_prompts_decoded)):
        row = {'prompt_idx':  shard_start + idx,
               'prompt_text': full_prompts_decoded[idx].replace('\r', '').replace('\n', ' ')}
        for k, w in enumerate(sample):
            row[f'w{k}'] = w
        for k, name in enumerate(reward_names):
            row[f'reward_{name}'] = float(rewards_list[k][idx])
        rows.append(row)

    df_chunk = pd.DataFrame(rows)
    df_chunk.to_csv(shard_path, mode='a', index=False,
                    header=not _csv_header_written, escapechar='\\')
    _csv_header_written = True
    print(f'[Rank {process_id}] Appended {len(rows)} rows → {shard_path}', flush=True)

    del gating, model
    gc.collect()
    torch.cuda.empty_cache()


print(f'[Rank {process_id}] Shard complete → {shard_path}', flush=True)
accelerator.wait_for_everyone()

# ---------------------------------------------------------------------------
# Rank 0 merges shards
# ---------------------------------------------------------------------------

if process_id == 0:
    weight_cols = [f'w{k}' for k in range(len(reward_names))]
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
