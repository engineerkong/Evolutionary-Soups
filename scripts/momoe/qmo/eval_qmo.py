import sys
from pathlib import Path
import os
import gc
from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM, DataCollatorWithPadding
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    get_clean_data, load_main_tokenizer, save_configs,
    sample_preferences_uniform, Instructions, Instructions_summary,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
)
from qmo_utils import merge_and_save_weights, load_moe_qtable, resolve_gating_checkpoint_path
from qmo_architecture import QTableGating

tqdm.pandas()

hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    sft_model_name: Optional[str] = field(default="./models/sft/model/")
    expert_model_paths: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = field(default="")
    manual_expert_weights: Optional[str] = field(
        default="1.0,0.0",
        metadata={"help": "comma-separated weights used when no q-table checkpoint is found"},
    )
    num_pref_samples: int = field(default=10)
    reward_names: Optional[str] = field(default='harmless,helpful')
    exp_type: Optional[str] = field(default='assistant')
    save_directory: Optional[str] = field(default='./results/qmo/')
    wandb_name: Optional[str] = field(default='qmo_assistant_eval')


def parse_manual_weights(spec, num_experts):
    weights = [float(v.strip()) for v in spec.split(",") if v.strip()]
    if len(weights) != num_experts:
        raise ValueError(f"manual_expert_weights has {len(weights)} values, expected {num_experts}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("manual_expert_weights must sum to a positive value")
    return [v / total for v in weights]


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir:', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ========== load reward models ==========
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print('reward names:', reward_names)
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful':  ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta':  ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary':  ['Tristan/gpt2_reward_summarization'],
    'faithful': ['CogComp/bart-faithful-summary-detector'],
    'humor':    ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f"Unknown reward name: {name}")
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

save_configs({
    'sft_model_name': script_args.sft_model_name,
    'expert_model_paths': str(script_args.expert_model_paths),
    **{f'reward_peft_path{i+1}': reward_model_path_list[i] for i in range(len(reward_model_path_list))},
}, output_dir)
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

# ========== load q-table or manual weights ==========
q_table = QTableGating()
resolved_checkpoint = resolve_gating_checkpoint_path(script_args.checkpoint_path)
weights_loaded = bool(script_args.checkpoint_path) and load_moe_qtable(q_table, resolved_checkpoint)
if not weights_loaded:
    manual_weights = parse_manual_weights(script_args.manual_expert_weights, len(script_args.expert_model_paths))
    print(f"No Q-table found. Using manual_expert_weights={manual_weights}")

# ========== prepare evaluation dataset ==========
tokenizer = load_main_tokenizer(script_args.sft_model_name)
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_ppo(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_ppo(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)


# ========== evaluation function — identical pattern to eval_ppo_rs ==========
def evaluate_model(temp_save_path, tokenizer, valid_dataset):
    mini_batch_size = 64
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, mini_batch_size, drop_last=True, collate_fn=data_collator)

    # Load freshly merged model from disk — same as eval_ppo_rs
    model = AutoModelForCausalLM.from_pretrained(
        temp_save_path,
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    model.resize_token_embeddings(len(tokenizer))

    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    generation_kwargs = {
        "max_new_tokens": 128 if script_args.exp_type == 'assistant' else 48,
        "min_length": -1,
        "do_sample": False,
    }
    tokenizer.padding_side = "left"

    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for batch in valid_data_loader:
            response_tensors = accelerator.unwrap_model(model).generate(
                batch["input_ids"], attention_mask=batch['attention_mask'], **generation_kwargs
            )
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)

    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post, normalize_rewards=False)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses, normalize_rewards=False)

    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_full_prompts, all_full_responses


# ========== start evaluation ==========
print("evaluating........")
sampled_preferences = sample_preferences_uniform(reward_models.num_rewards, num_samples=script_args.num_pref_samples)
sampled_preferences = [[0.0,1.0]]
print(f"\nSampled {len(sampled_preferences)} preferences:")
for i, pref in enumerate(sampled_preferences):
    pref_str = ", ".join([f"{reward_names[k]}={pref[k]:.2f}" for k in range(len(reward_names))])
    print(f"  Pref {i+1}: [{pref_str}]")

all_results = []

for k, preference in enumerate(sampled_preferences):
    expert_weights = q_table.best_weights(preference) if weights_loaded else manual_weights
    print(f"\nPref {k+1}: preference={preference}, expert_weights={expert_weights}")

    temp_save_path = os.path.join(output_dir, 'temp_merged_model_pref_{}_{}'.format(
        '_'.join([str(round(p, 4)) for p in preference]), k
    ))

    # Only rank 0 does the merge and saves to disk — same as eval_ppo_rs
    if process_id == 0:
        merge_and_save_weights(script_args.expert_model_paths, expert_weights, temp_save_path)
        print("merged model saved to {}".format(temp_save_path))

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

    all_rewards, all_full_prompts, all_full_responses = evaluate_model(temp_save_path, tokenizer, valid_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
            'preference': [preference] * len(all_full_prompts),
            'expert_weights': [expert_weights] * len(all_full_prompts),
        }
        for i in range(reward_models.num_rewards):
            evaluation_result[f'reward_{reward_names[i]}'] = all_rewards[i]
            print(f'  avg {reward_names[i]} reward: {np.mean(all_rewards[i]):.4f}')

        df = pd.DataFrame(evaluation_result)
        df.to_csv(
            os.path.join(output_dir, f'eval_data_pref{"_".join([str(round(p, 2)) for p in preference])}.csv'),
            escapechar='\\'
        )
        all_results.append({
            'pref_idx': k,
            **{f'pref_{reward_names[j]}': preference[j] for j in range(len(reward_names))},
            **{f'expert_weight_{j}': expert_weights[j] for j in range(len(expert_weights))},
            **{f'mean_reward_{reward_names[i]}': float(np.mean(all_rewards[i])) for i in range(reward_models.num_rewards)},
        })

if process_id == 0:
    pd.DataFrame(all_results).to_csv(os.path.join(output_dir, 'eval_summary.csv'), index=False)
    print("\nEvaluation complete. Results saved to:", output_dir)