import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from trl import set_seed
from accelerate import Accelerator
from moe_architecture import RewardModels, MoEGatingTrainer
from train_moe import convert_to_moe_model, load_moe_gating_weights
from utils import (
    load_main_tokenizer, 
    Instructions, 
    Instructions_summary,
    build_dataset_eval,
    build_dataset_summary_eval
)

# Dataset paths
HHRLHF_DATASET_PATH = 'Anthropic/hh-rlhf'
SUMMARY_DATASET_PATH = 'openai/summarize_from_feedback'


def get_clean_response(response_tensor, input_ids, tokenizer):
    """Extract and clean the generated response from full output tensor.
    
    Args:
        response_tensor: Full generated sequence [seq_len]
        input_ids: Original input (with padding) [prompt_len] 
        tokenizer: Tokenizer with pad_token_id
    """
    full_response = tokenizer.decode(response_tensor, skip_special_tokens=True)
    print(f"Full text: {full_response}")
    # Find actual prompt length (exclude left padding)
    pad_token_id = tokenizer.pad_token_id
    # # Count non-pad tokens in input_ids to get actual prompt length
    # actual_prompt_len = (input_ids != pad_token_id).sum().item()
    
    # Or find first non-pad position if left-padded
    actual_prompt_len = len(input_ids) - (input_ids == pad_token_id).sum().item()
    
    # The generated sequence starts after the original input length
    input_len = len(input_ids)
    # print(f"Input length (with padding): {input_len}, Actual prompt length: {actual_prompt_len}")
    
    # New tokens are everything after input_len in the response_tensor
    new_tokens = response_tensor[input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"Generated response: {response}")
    
    # # Clean up extra dialogue turns
    # response = response.split('\n\nHuman:')[0].strip()
    # response = response.split('\nHuman:')[0].strip()
    # response = response.split('\n\nAssistant:')[0].strip()
    # response = response.split('\nAssistant:')[0].strip()
    # response = response.split('\n\n\n')[0].strip()
    # response = response.split('###')[0].strip()
    
    return response


def compute_hypervolume_2d(points, ref_point):
    """
    Compute hypervolume for 2D points.
    points: list of [r1, r2] reward vectors
    ref_point: reference point [ref1, ref2]
    """
    if len(points) == 0:
        return 0.0
    
    # Filter dominated points and sort by first objective (descending)
    points = np.array(points)
    
    # Only keep points that dominate the reference point
    valid_mask = np.all(points > ref_point, axis=1)
    if not np.any(valid_mask):
        return 0.0
    
    valid_points = points[valid_mask]
    
    # Sort by first objective (descending)
    sorted_indices = np.argsort(-valid_points[:, 0])
    sorted_points = valid_points[sorted_indices]
    
    # Compute hypervolume
    hv = 0.0
    prev_y = ref_point[1]
    
    for point in sorted_points:
        if point[1] > prev_y:
            hv += (point[0] - ref_point[0]) * (point[1] - prev_y)
            prev_y = point[1]
    
    return hv


def sample_preferences_uniform(num_rewards, num_samples):
    """Sample preferences uniformly from the simplex."""
    preferences = []
    for i in range(num_samples):
        # Uniform sampling on simplex
        if num_rewards == 2:
            # For 2D, use evenly spaced points
            w = i / (num_samples - 1) if num_samples > 1 else 0.5
            preferences.append([w, 1 - w])
        else:
            # For higher dimensions, use Dirichlet
            pref = np.random.dirichlet(np.ones(num_rewards))
            preferences.append(pref.tolist())
    return preferences


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MoE model with dataset inputs')
    parser.add_argument('--base_model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--lora_expert_paths', type=str, nargs='+', required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--subspace_rank', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--reward_names', type=str, default='harmless,helpful')
    parser.add_argument('--exp_type', type=str, default='assistant')
    parser.add_argument('--num_pref_samples', type=int, default=1, 
                        help='Number of preference samples per input')
    parser.add_argument('--num_eval_samples', type=int, default=100,
                        help='Number of dataset samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_directory', type=str, default='./moe_eval/')
    parser.add_argument('--seed', type=int, default=8888)
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.save_directory, exist_ok=True)

    reward_names = [x.strip() for x in args.reward_names.split(',')]
    num_rewards = len(reward_names)
    print(f"Number of rewards: {num_rewards}")
    print(f"Reward names: {reward_names}")
    
    # =========================================================================
    # 1. Load MoE Model
    # =========================================================================
    accelerator = Accelerator()
    gpu_id = accelerator.local_process_index
    target_device = f"cuda:{gpu_id}"
    
    print("Loading MoE model...")
    model = convert_to_moe_model(
        base_model_name=args.base_model_name,
        lora_expert_paths=args.lora_expert_paths,
        subspace_rank=args.subspace_rank,
        d_model=args.d_model,
        num_rewards=num_rewards,
        target_device=target_device # or 'cpu' (hard coded)
    )
    
    print(f"Loading gating weights from: {args.checkpoint_path}")
    load_moe_gating_weights(model, args.checkpoint_path)
    model.eval()
    
    tokenizer = load_main_tokenizer(args.base_model_name)
    tokenizer.padding_side = "left"
    
    # =========================================================================
    # 2. Load Reward Models
    # =========================================================================
    print("Loading reward models...")
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
            raise NotImplementedError(f"Reward model '{name}' not found")
        reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
        rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
    
    # Use GPU 0 for reward models (or adjust as needed)
    rm_gpu_id = 0
    reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, rm_gpu_id)
    
    # =========================================================================
    # 3. Load Dataset
    # =========================================================================
    print("Loading dataset...")
    if args.exp_type == 'assistant':
        valid_dataset = build_dataset_eval(
            HHRLHF_DATASET_PATH, tokenizer, reward_model.rm_tokenizers, split='test'
        )
        instructions = Instructions()
    else:
        valid_dataset = build_dataset_summary_eval(
            SUMMARY_DATASET_PATH, tokenizer, reward_model.rm_tokenizers, split='test'
        )
        instructions = Instructions_summary()
    
    print(f"Full dataset size: {len(valid_dataset)}")
    
    # # Limit to num_eval_samples
    # if args.num_eval_samples < len(valid_dataset):
    #     valid_dataset = valid_dataset.select(range(args.num_eval_samples))
    # print(f"Evaluation dataset size: {len(valid_dataset)}")
    
    # Remove unnecessary columns
    for key in ['key', 'text', 'response']:
        if key in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(key)
    
    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        drop_last=False, 
        collate_fn=data_collator
    )
    
    # =========================================================================
    # 4. Sample Preferences
    # =========================================================================
    sampled_preferences = sample_preferences_uniform(num_rewards, args.num_pref_samples)
    print(f"\nSampled {len(sampled_preferences)} preferences:")
    for i, pref in enumerate(sampled_preferences):
        pref_str = ", ".join([f"{reward_names[k]}={pref[k]:.2f}" for k in range(num_rewards)])
        print(f"  Pref {i+1}: [{pref_str}]")
    
    # =========================================================================
    # 5. Generation and Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Starting Evaluation")
    print("=" * 80)
    
    generation_kwargs = {
        "max_new_tokens": 128 if args.exp_type == 'assistant' else 48,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # Storage for results
    all_results = []  # List of dicts for DataFrame
    per_input_hv = []  # HV per input (across all preferences)
    per_input_rewards = []  # Reward vectors per input
    
    # Reference point for HV calculation (adjust based on your reward scale)
    ref_point = [-5.0] * num_rewards  # Assuming rewards can be negative
    
    # Create trainer for utility functions
    trainer = MoEGatingTrainer(
        moe_model=model,
        reward_model=reward_model,
        instructions=instructions,
        num_rewards=num_rewards,
        num_pref_samples=args.num_pref_samples
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
            batch_prompt_lens = [batch_attention_mask[i].sum().item() for i in range(batch_size)]
            
            # For each input in batch, collect rewards across all preferences
            for local_idx in range(batch_size):
                single_input_ids = batch_input_ids[local_idx:local_idx+1]
                single_attention_mask = batch_attention_mask[local_idx:local_idx+1]
                single_prompt = batch_prompts[local_idx]
                single_prompt_len = batch_prompt_lens[local_idx]
                
                input_reward_vectors = []  # Rewards for this input across preferences
                
                # Generate for each preference
                for pref_idx, pref in enumerate(sampled_preferences):
                    # Set preference for gating
                    trainer.set_model_preference(pref)
                    
                    # Generate
                    outputs = model.generate(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_mask,
                        **generation_kwargs
                    )

                    # Extract response
                    response = get_clean_response(
                        outputs[0], single_input_ids[0], tokenizer
                    )

                    # Compute rewards
                    full_text = single_prompt + ' ' + response
                    query_response = (
                        instructions.get_input(full_text),
                        instructions.get_response(full_text)
                    )
                    
                    if hasattr(instructions, 'get_post'):
                        rewards_list = reward_model.get_reward_model_scores(
                            [query_response], instructions.get_post, normalize_rewards=False
                        )
                    else:
                        rewards_list = reward_model.get_reward_model_scores([query_response], normalize_rewards=False)
                    
                    reward_vector = [rewards_list[k][0] for k in range(num_rewards)]
                    scalarized_reward = sum(pref[k] * reward_vector[k] for k in range(num_rewards))
                    
                    input_reward_vectors.append(reward_vector)
                    
                    # Store result
                    result = {
                        'input_idx': input_idx,
                        'pref_idx': pref_idx,
                        'prompt': single_prompt,
                        'response': response,
                        'scalarized_reward': scalarized_reward,
                    }
                    for k, name in enumerate(reward_names):
                        result[f'pref_{name}'] = pref[k]
                        result[f'reward_{name}'] = reward_vector[k]
                    
                    all_results.append(result)
                
                # Compute HV for this input (across all preferences)
                input_hv = compute_hypervolume_2d(input_reward_vectors, ref_point)
                per_input_hv.append(input_hv)
                per_input_rewards.append(input_reward_vectors)
                
                input_idx += 1
                pbar.update(1)
    
    pbar.close()
    
    # =========================================================================
    # 6. Compute Summary Statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    # Overall statistics
    print(f"\nTotal inputs evaluated: {len(per_input_hv)}")
    print(f"Preferences per input: {args.num_pref_samples}")
    print(f"Total generations: {len(all_results)}")
    
    # Per-objective reward statistics
    print(f"\n{'='*40}")
    print("Per-Objective Reward Statistics:")
    print(f"{'='*40}")
    for name in reward_names:
        col = f'reward_{name}'
        print(f"  {name}:")
        print(f"    Mean: {results_df[col].mean():.4f}")
        print(f"    Std:  {results_df[col].std():.4f}")
        print(f"    Min:  {results_df[col].min():.4f}")
        print(f"    Max:  {results_df[col].max():.4f}")
    
    # Scalarized reward statistics
    print(f"\n{'='*40}")
    print("Scalarized Reward Statistics:")
    print(f"{'='*40}")
    print(f"  Mean: {results_df['scalarized_reward'].mean():.4f}")
    print(f"  Std:  {results_df['scalarized_reward'].std():.4f}")
    
    # Per-input Hypervolume statistics
    print(f"\n{'='*40}")
    print("Per-Input Hypervolume Statistics:")
    print(f"{'='*40}")
    hv_array = np.array(per_input_hv)
    print(f"  Mean HV: {hv_array.mean():.4f}")
    print(f"  Std HV:  {hv_array.std():.4f}")
    print(f"  Min HV:  {hv_array.min():.4f}")
    print(f"  Max HV:  {hv_array.max():.4f}")
    
    # Global Hypervolume (all points)
    all_reward_vectors = [[r[f'reward_{name}'] for name in reward_names] for r in all_results]
    global_hv = compute_hypervolume_2d(all_reward_vectors, ref_point)
    print(f"\n{'='*40}")
    print(f"Global Hypervolume (all points): {global_hv:.4f}")
    print(f"{'='*40}")
    
    # Per-preference statistics
    print(f"\n{'='*40}")
    print("Per-Preference Statistics:")
    print(f"{'='*40}")
    for pref_idx, pref in enumerate(sampled_preferences):
        pref_df = results_df[results_df['pref_idx'] == pref_idx]
        pref_str = ", ".join([f"{reward_names[k]}={pref[k]:.2f}" for k in range(num_rewards)])
        print(f"\n  Preference {pref_idx + 1} [{pref_str}]:")
        for name in reward_names:
            col = f'reward_{name}'
            print(f"    {name}: mean={pref_df[col].mean():.4f}, std={pref_df[col].std():.4f}")
        print(f"    Scalarized: mean={pref_df['scalarized_reward'].mean():.4f}")
    
    # =========================================================================
    # 7. Save Results
    # =========================================================================
    # Save detailed results
    results_path = os.path.join(args.save_directory, 'eval_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save per-input HV
    hv_df = pd.DataFrame({
        'input_idx': range(len(per_input_hv)),
        'hypervolume': per_input_hv
    })
    hv_path = os.path.join(args.save_directory, 'per_input_hv.csv')
    hv_df.to_csv(hv_path, index=False)
    print(f"Per-input HV saved to: {hv_path}")
    
    # Save summary statistics
    summary = {
        'num_inputs': len(per_input_hv),
        'num_preferences': args.num_pref_samples,
        'mean_hv': hv_array.mean(),
        'std_hv': hv_array.std(),
        'global_hv': global_hv,
        'mean_scalarized_reward': results_df['scalarized_reward'].mean(),
    }
    for name in reward_names:
        summary[f'mean_reward_{name}'] = results_df[f'reward_{name}'].mean()
        summary[f'std_reward_{name}'] = results_df[f'reward_{name}'].std()
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.save_directory, 'eval_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()