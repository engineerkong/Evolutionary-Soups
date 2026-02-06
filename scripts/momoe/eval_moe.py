import sys
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DataCollatorWithPadding, AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
from trl import set_seed
from accelerate import Accelerator
from moe_architecture import MoEGatingTrainer
from moe_utils import compute_hypervolume, convert_to_moe_model, load_moe_gating_weights

script_dir = Path(__file__).resolve().parent  # project/scripts/momoe
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.utils import (
    load_main_tokenizer, 
    Instructions, 
    Instructions_summary,
    build_dataset_eval_ppo,
    build_dataset_summary_eval_ppo,
    get_clean_data,
    load_config,
    sample_preferences_uniform
)
from scripts.utils.multi_reward_models import RewardModels

# ========== define paths for two datasets ==========
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: str = field(default="./models/sft/model/", metadata={"help": "local path to the sft model"})
    lora_expert_paths: List[str] = field(default_factory=lambda: [], metadata={"help": "list of paths to LoRA expert models"})
    checkpoint_path: str = field(default='', metadata={"help": "path to MoE gating weights checkpoint"})
    num_pref_samples: int = field(default=10, metadata={"help": "number of preference samples per input"})
    num_eval_samples: Optional[int] = field(default=0, metadata={"help": "number of dataset samples to evaluate"})
    reward_names: Optional[str] = field(default='harmless,helpful', metadata={"help": "comma-separated reward model names"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "experiment type: 'assistant' or 'summary'"})
    save_directory: str = field(default='./results/momoe/')
    wandb_name: str = field(default="assistant_moemoe", metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config(script_dir / 'config.yaml')
print(f"Script arguments: {script_args}")
print(f"Training config: {cfg}")

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ========== load reward models ==========
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(f"reward names: {reward_names}")
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
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])

# ========== load MoE Model ==========
model = convert_to_moe_model(
    base_model_name=script_args.base_model_name,
    lora_expert_paths=script_args.lora_expert_paths,
    subspace_rank=cfg['subspace_rank'],
    d_model=cfg['d_model'],
    num_rewards=len(reward_names),
    target_device=f"cuda:{gpu_id}"
)
tokenizer = load_main_tokenizer(script_args.base_model_name)

print(f"Loading gating weights from: {script_args.checkpoint_path}")
load_moe_gating_weights(model, script_args.checkpoint_path)

# ========== prepare evaluation dataset and dataloader ==========
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_ppo(
        hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test'
    )
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_ppo(
        summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test'
    )
    instructions = Instructions_summary()

print(f"Full dataset size: {len(valid_dataset)}")

# Limit to num_eval_samples
if script_args.num_eval_samples > 0 and script_args.num_eval_samples < len(valid_dataset):
    valid_dataset = valid_dataset.select(range(script_args.num_eval_samples))
print(f"Evaluation dataset size: {len(valid_dataset)}")

valid_batch_size = 8
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

# Create DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_dataloader = DataLoader(
    valid_dataset, 
    batch_size=valid_batch_size, 
    drop_last=False, 
    collate_fn=data_collator
)

sampled_preferences = sample_preferences_uniform(len(reward_names), script_args.num_pref_samples)
print(f"\nSampled {len(sampled_preferences)} preferences:")
for i, pref in enumerate(sampled_preferences):
    pref_str = ", ".join([f"{reward_names[k]}={pref[k]:.2f}" for k in range(len(reward_names))])
    print(f"  Pref {i+1}: [{pref_str}]")

# ========== define generation kwargs ==========
generation_kwargs = {
    "max_new_tokens": 128 if script_args.exp_type == 'assistant' else 48,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
}
tokenizer.padding_side = "left"
model.eval()

# ========== start evaluation ==========
# Storage for results
all_results = []  # List of dicts for DataFrame
per_input_hv = []  # HV per input (across all preferences)
per_input_rewards = []  # Reward vectors per input

# Create trainer for utility functions
trainer = MoEGatingTrainer(
    moe_model=model,
    reward_models=reward_models,
    instructions=instructions,
    num_rewards=len(reward_names),
    num_pref_samples=script_args.num_pref_samples
)

input_idx = 0
pbar = tqdm(total=len(valid_dataset), desc="Evaluating")

with torch.no_grad():
    for batch_idx, batch in enumerate(eval_dataloader):
        batch_input_ids = batch['input_ids'].to(model.device)
        batch_attention_mask = batch['attention_mask'].to(model.device)
        batch_size = batch_input_ids.shape[0]
        
        # Decode prompts for this batch
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        
        # For each preference, generate for the ENTIRE batch at once
        for pref_idx, pref in enumerate(sampled_preferences):
            trainer.set_model_preference(pref)
            
            # Generate for entire batch at once
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                **generation_kwargs
            )
            
            # Decode all responses at once
            full_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            full_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=False)
            full_prompts, full_responses = get_clean_data(full_responses, full_prompts, remove_bad=False)
            
            # Build query-response pairs for entire batch
            queries_responses = [
                (instructions.get_input(text), instructions.get_response(text))
                for text in full_responses
            ]
            
            # Get rewards for entire batch at once
            if hasattr(instructions, 'get_post'):
                rewards_list = reward_models.get_reward_model_scores(
                    queries_responses, instructions.get_post, normalize_rewards=False
                )
            else:
                rewards_list = reward_models.get_reward_model_scores(
                    queries_responses, normalize_rewards=False
                )
            
            # Process results for each sample in batch
            for local_idx in range(batch_size):
                reward_vector = [rewards_list[k][local_idx] for k in range(len(reward_names))]
                scalarized_reward = sum(pref[k] * reward_vector[k] for k in range(len(reward_names)))
                
                # Extract response (without prompt)
                response = full_responses[local_idx].replace(full_prompts[local_idx], '').strip()
                
                result = {
                    'input_idx': input_idx + local_idx,
                    'pref_idx': pref_idx,
                    'prompt': full_prompts[local_idx],
                    'response': response,
                    'scalarized_reward': scalarized_reward,
                }
                for k, name in enumerate(reward_names):
                    result[f'pref_{name}'] = pref[k]
                    result[f'reward_{name}'] = reward_vector[k]
                
                all_results.append(result)
                
                # Track per-input rewards for hypervolume
                if pref_idx == 0:
                    per_input_rewards.append([])
                per_input_rewards[input_idx + local_idx].append(reward_vector)
        
        # Compute hypervolume for each input in batch (after all preferences)
        for local_idx in range(batch_size):
            input_hv = compute_hypervolume(
                per_input_rewards[input_idx + local_idx]
            )
            per_input_hv.append(input_hv)
        
        input_idx += batch_size
        pbar.update(batch_size)

pbar.close()

# ========== summarize and print evaluation results ==========
print("\n" + "=" * 60)
print("Evaluation Summary")
print("=" * 60)

results_df = pd.DataFrame(all_results)

# Overview
print(f"\nInputs: {len(valid_dataset)} | Preferences/input: {script_args.num_pref_samples} | Total generations: {len(all_results)}")

# Per-objective rewards (overall)
print(f"\nPer-Objective Rewards (Overall):")
for name in reward_names:
    col = f'reward_{name}'
    print(f"  {name}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f} [{results_df[col].min():.4f}, {results_df[col].max():.4f}]")

# Scalarized reward (overall)
print(f"\nScalarized Reward (Overall): {results_df['scalarized_reward'].mean():.4f} ± {results_df['scalarized_reward'].std():.4f}")

# Per-preference sample results
print(f"\n" + "-" * 60)
print("Per-Preference Sample Results:")
print("-" * 60)

summary_rows = []

# Add overall summary row
all_reward_vectors = [[r[f'reward_{name}'] for name in reward_names] for r in all_results]
global_hv = compute_hypervolume(all_reward_vectors)

overall_row = {
    'type': 'overall',
    'pref_idx': -1,
    **{f'pref_{name}': None for name in reward_names},
    'mean_scalarized_reward': results_df['scalarized_reward'].mean(),
    'std_scalarized_reward': results_df['scalarized_reward'].std(),
    **{f'mean_reward_{name}': results_df[f'reward_{name}'].mean() for name in reward_names},
    **{f'std_reward_{name}': results_df[f'reward_{name}'].std() for name in reward_names},
    'hypervolume': global_hv,
    'num_inputs': len(valid_dataset),
    'num_preferences': script_args.num_pref_samples,
}
summary_rows.append(overall_row)

# Add per-preference rows
for pref_idx, pref in enumerate(sampled_preferences):
    pref_df = results_df[results_df['pref_idx'] == pref_idx]
    pref_str = ", ".join([f"{reward_names[k]}={pref[k]:.2f}" for k in range(len(reward_names))])
    
    print(f"\nPref {pref_idx + 1}: [{pref_str}]")
    
    pref_row = {
        'type': 'preference',
        'pref_idx': pref_idx,
        **{f'pref_{name}': pref[k] for k, name in enumerate(reward_names)},
        'mean_scalarized_reward': pref_df['scalarized_reward'].mean(),
        'std_scalarized_reward': pref_df['scalarized_reward'].std(),
        'hypervolume': None,
        'num_inputs': None,
        'num_preferences': None,
    }
    
    for name in reward_names:
        col = f'reward_{name}'
        mean_val = pref_df[col].mean()
        std_val = pref_df[col].std()
        print(f"  {name}: {mean_val:.4f} ± {std_val:.4f}")
        pref_row[f'mean_reward_{name}'] = mean_val
        pref_row[f'std_reward_{name}'] = std_val
    
    print(f"  Scalarized: {pref_df['scalarized_reward'].mean():.4f} ± {pref_df['scalarized_reward'].std():.4f}")
    summary_rows.append(pref_row)

print("-" * 60)

# Hypervolume
print(f"\nHypervolume (Global): {global_hv:.4f}")

# Save results
results_df.to_csv(os.path.join(output_dir, 'eval_results.csv'), index=False)
pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, 'eval_summary.csv'), index=False)

print(f"\nResults saved to: {output_dir}")
print("  - eval_results.csv (all generations)")
print("  - eval_summary.csv (overall + per-preference summary)")
print("=" * 60)