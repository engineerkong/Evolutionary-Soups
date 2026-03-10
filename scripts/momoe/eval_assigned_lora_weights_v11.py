import os
import sys
import types
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer, HfArgumentParser
from trl import set_seed
from accelerate import Accelerator

from moe_utils_v11 import (
    convert_to_moe_model,
    load_moe_gating_weights,
    resolve_gating_checkpoint_path,
)

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.utils import (
    load_main_tokenizer,
    Instructions,
    Instructions_summary,
    build_dataset_eval_ppo,
    build_dataset_summary_eval_ppo,
    get_clean_data,
)
from scripts.utils.multi_reward_models import RewardModels


hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    base_model_name: str = field(default="./models/sft/model/")
    lora_expert_paths: List[str] = field(default_factory=lambda: [])
    checkpoint_path: str = field(default='')
    assigned_lora_weights: str = field(
        default='1,0;0,1;0.5,0.5',
        metadata={"help": "semicolon separated expert weights, e.g. '1,0;0,1;0.5,0.5'"}
    )
    num_eval_samples: Optional[int] = field(default=0)
    eval_batch_size: int = field(default=8)
    reward_names: Optional[str] = field(default='harmless,helpful')
    exp_type: Optional[str] = field(default='assistant')
    save_directory: str = field(default='./results/momoe/')
    run_name: str = field(default='assigned_lora_weights_eval')
    seed: int = field(default=8888)


def parse_weight_sets(spec: str) -> List[List[float]]:
    sets = []
    for block in spec.split(';'):
        block = block.strip()
        if not block:
            continue
        vals = [float(x.strip()) for x in block.split(',') if x.strip()]
        total = sum(vals)
        if total <= 0:
            raise ValueError(f"Invalid weight set (sum<=0): {block}")
        vals = [v / total for v in vals]
        sets.append(vals)
    if not sets:
        raise ValueError("No valid assigned_lora_weights found.")
    return sets


def set_fixed_lora_weights(model, weights: List[float]):
    core_model = model.module if hasattr(model, 'module') else model
    ref_dtype = next(core_model.parameters()).dtype
    ref_device = next(core_model.parameters()).device
    manual = torch.tensor(weights, dtype=ref_dtype, device=ref_device)

    for layer in core_model.model.layers:
        if not hasattr(layer.mlp, 'gate'):
            continue
        gate = layer.mlp.gate
        if gate.num_lora_experts != manual.numel():
            raise ValueError(
                f"Weight size mismatch: got {manual.numel()}, expected {gate.num_lora_experts}"
            )

        gate._manual_lora_weights = manual

        if getattr(gate, '_forward_overridden_for_manual_weights', False):
            continue

        def manual_forward(self, x, preference=None):
            batch, seq_len, _ = x.shape
            w = self._manual_lora_weights.to(device=x.device, dtype=x.dtype)
            w_batch = w.unsqueeze(0).expand(batch, -1)
            self._last_routing_weights = w_batch.detach()
            return w_batch.unsqueeze(1).expand(-1, seq_len, -1)

        gate.forward = types.MethodType(manual_forward, gate)
        gate._forward_overridden_for_manual_weights = True


def get_scalarized_reward(reward_vector: List[float], scalar_pref: List[float]) -> float:
    return float(sum(scalar_pref[i] * reward_vector[i] for i in range(len(reward_vector))))


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    accelerator = Accelerator()
    process_id = accelerator.local_process_index
    gpu_id = process_id if torch.cuda.is_available() else -1
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"process: {process_id}, model gpu id: {gpu_id}")

    reward_names = [x.strip() for x in args.reward_names.split(',')]
    reward_path_tokenizer_dict = {
        'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
        'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
        'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
        'summary': ['Tristan/gpt2_reward_summarization'],
        'faithful': ['CogComp/bart-faithful-summary-detector'],
        'humor': ['mohameddhiab/humor-no-humor'],
    }

    reward_model_path_list = []
    rm_tokenizer_path_list = []
    for name in reward_names:
        if name not in reward_path_tokenizer_dict:
            raise NotImplementedError(f"Unsupported reward name: {name}")
        reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
        rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

    reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
    _ = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])

    model = convert_to_moe_model(
        base_model_name=args.base_model_name,
        lora_expert_paths=args.lora_expert_paths,
        num_rewards=len(reward_names),
        target_device=device,
    )
    tokenizer = load_main_tokenizer(args.base_model_name)

    if args.checkpoint_path:
        ckpt = resolve_gating_checkpoint_path(args.checkpoint_path)
        print(f"Loading gating weights from: {ckpt}")
        _ = load_moe_gating_weights(model, ckpt)

    if args.exp_type == 'assistant':
        valid_dataset = build_dataset_eval_ppo(
            hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test'
        )
        instructions = Instructions()
        default_max_new_tokens = 128
    else:
        valid_dataset = build_dataset_summary_eval_ppo(
            summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test'
        )
        instructions = Instructions_summary()
        default_max_new_tokens = 48
    print(f"Full dataset size: {len(valid_dataset)}")

    if args.num_eval_samples and args.num_eval_samples > 0 and args.num_eval_samples < len(valid_dataset):
        valid_dataset = valid_dataset.select(range(args.num_eval_samples))
    print(f"Evaluation dataset size: {len(valid_dataset)}")

    global_eval_size = len(valid_dataset)
    shard_start = 0
    if accelerator.num_processes > 1:
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        base = global_eval_size // world_size
        remainder = global_eval_size % world_size
        shard_start = rank * base + min(rank, remainder)
        valid_dataset = valid_dataset.shard(
            num_shards=world_size,
            index=rank,
            contiguous=True
        )
        print(f"Rank {rank}: local eval shard size {len(valid_dataset)} (start={shard_start})")

    for key in ['key', 'text', 'prompt', 'response', 'query']:
        if key in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(key)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        drop_last=True,
        collate_fn=data_collator,
    )

    assigned_weight_sets = parse_weight_sets(args.assigned_lora_weights)
    num_experts = len(args.lora_expert_paths)
    for ws in assigned_weight_sets:
        if len(ws) != num_experts:
            raise ValueError(f"Each assigned weight set must have {num_experts} values, got {len(ws)}: {ws}")

    output_dir = os.path.join(args.save_directory, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    generation_kwargs = {
        "max_new_tokens": 128 if args.exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": False,
    }

    tokenizer.padding_side = 'left'
    model.eval()

    base_columns = [
        'input_idx', 'weight_idx', 'assigned_lora_weights', 'prompt', 'response', 'scalarized_reward'
    ] + [f"reward_{name}" for name in reward_names]
    all_rows = []
    pbar = tqdm(
        total=len(valid_dataset),
        desc=f'Assigned-Weight Eval (rank {accelerator.process_index})',
        disable=not accelerator.is_local_main_process
    )
    global_input_offset = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            batch_size = input_ids.shape[0]

            for weight_idx, assigned_weights in enumerate(assigned_weight_sets):
                set_fixed_lora_weights(model, assigned_weights)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

                full_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                full_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                full_prompts, full_responses = get_clean_data(full_responses, full_prompts, remove_bad=False)

                queries_responses = [
                    (instructions.get_input(text), instructions.get_response(text))
                    for text in full_responses
                ]

                if hasattr(instructions, 'get_post'):
                    rewards_list = reward_models.get_reward_model_scores(
                        queries_responses, instructions.get_post, normalize_rewards=False
                    )
                else:
                    rewards_list = reward_models.get_reward_model_scores(
                        queries_responses, normalize_rewards=False
                    )

                scalar_pref = assigned_weights[:len(reward_names)]
                pref_total = sum(scalar_pref)
                if pref_total <= 0:
                    scalar_pref = [1.0 / len(reward_names)] * len(reward_names)
                else:
                    scalar_pref = [v / pref_total for v in scalar_pref]

                for i in range(batch_size):
                    reward_vector = [float(rewards_list[k][i]) for k in range(len(reward_names))]
                    scalarized_reward = get_scalarized_reward(reward_vector, scalar_pref)
                    print(f"assigned_lora_weights: {assigned_weights}, scalarized_reward: {scalarized_reward}, reward_vector: {reward_vector}")

                    response = full_responses[i].replace(full_prompts[i], '').strip()
                    row = {
                        'input_idx': shard_start + global_input_offset + i,
                        'weight_idx': weight_idx,
                        'assigned_lora_weights': str(assigned_weights),
                        'prompt': full_prompts[i],
                        'response': response,
                        'scalarized_reward': scalarized_reward,
                    }
                    for k, name in enumerate(reward_names):
                        row[f'reward_{name}'] = reward_vector[k]
                    all_rows.append(row)

            global_input_offset += batch_size
            pbar.update(batch_size)

    pbar.close()
    rank_csv = os.path.join(output_dir, f'assigned_lora_weights_eval_rank{accelerator.process_index}.csv')
    if all_rows:
        pd.DataFrame(all_rows).to_csv(rank_csv, index=False)
    else:
        pd.DataFrame(columns=base_columns).to_csv(rank_csv, index=False)

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        print(f"Rank {accelerator.process_index}: shard complete.")
        return

    all_rank_paths = [
        os.path.join(output_dir, f'assigned_lora_weights_eval_rank{r}.csv')
        for r in range(accelerator.num_processes)
    ]
    rank_dfs = []
    for path in all_rank_paths:
        if os.path.exists(path):
            rank_dfs.append(pd.read_csv(path))
    if rank_dfs:
        df = pd.concat(rank_dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=base_columns)

    result_csv = os.path.join(output_dir, 'assigned_lora_weights_eval.csv')
    df.to_csv(result_csv, index=False)

    group_cols = ['weight_idx', 'assigned_lora_weights']
    agg = {
        'scalarized_reward': ['mean', 'std'],
    }
    for n in reward_names:
        agg[f'reward_{n}'] = ['mean', 'std']

    if len(df) > 0:
        summary_df = df.groupby(group_cols, as_index=False).agg(agg)
        summary_df.columns = ['_'.join(c).strip('_') for c in summary_df.columns.to_flat_index()]
    else:
        summary_df = pd.DataFrame(columns=['weight_idx', 'assigned_lora_weights'])
    summary_csv = os.path.join(output_dir, 'assigned_lora_weights_summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    print("=" * 60)
    print(f"Saved: {result_csv}")
    print(f"Saved: {summary_csv}")
    print("=" * 60)


if __name__ == '__main__':
    main()
