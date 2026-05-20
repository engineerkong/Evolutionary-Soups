#!/bin/bash
# Qwen2-7B fine-tuning on PKU-Alignment/PKU-SafeRLHF-10K (beaver), GPU 7

# # SFT TRAINING
# CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/sft.py \
#     --base_model_name 'Qwen/Qwen2-7B' \
#     --dataset_name 'PKU-Alignment/PKU-SafeRLHF-10K' \
#     --wandb_name 'sft_qwen2_beaver' \
#     2>&1 | tee ./logs/sft/sft_qwen2_beaver.log

# # SFT EVALUATION
# CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/eval_sft.py \
#     --base_model_name 'Qwen/Qwen2-7B' \
#     --sft_model_name './models/sft/sft_qwen2_beaver/model/' \
#     --dataset_name 'PKU-Alignment/PKU-SafeRLHF-10K' \
#     --reward_names 'beaver_reward,beaver_cost' \
#     --wandb_name 'sft_qwen2_beaver_eval' \
#     2>&1 | tee ./logs/sft/sft_qwen2_beaver_eval.log

# # PPO OBJECTIVE-SPECIFIC TRAINING — beaver_reward
# CUDA_VISIBLE_DEVICES=4,7 accelerate launch --main_process_port 29604 ./scripts/fine-tuning/ppo.py \
#     --base_model_name 'Qwen/Qwen2-7B' \
#     --sft_model_name './models/sft/sft_qwen2_beaver/model/' \
#     --dataset_name 'PKU-Alignment/PKU-SafeRLHF-10K' \
#     --reward_name 'beaver_reward' \
#     --wandb_name 'ppo_qwen2_beaver_reward' \
#     2>&1 | tee ./logs/ppo/ppo_qwen2_beaver_reward.log

# # PPO OBJECTIVE-SPECIFIC TRAINING — beaver_cost
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/ppo.py \
#     --base_model_name 'Qwen/Qwen2-7B' \
#     --sft_model_name './models/sft/sft_qwen2_beaver/model/' \
#     --dataset_name 'PKU-Alignment/PKU-SafeRLHF-10K' \
#     --reward_name 'beaver_cost' \
#     --wandb_name 'ppo_qwen2_beaver_cost' \
#     2>&1 | tee ./logs/ppo/ppo_qwen2_beaver_cost.log

# # PPO OBJECTIVE-SPECIFIC EVALUATION — beaver_reward
# CUDA_VISIBLE_DEVICES=4,7 accelerate launch --main_process_port 29604 ./scripts/fine-tuning/eval_ppo_single.py \
#     --base_model_name 'Qwen/Qwen2-7B' \
#     --ppo_model_name './models/ppo/ppo_qwen2_beaver_reward/best_model/' \
#     --dataset_name 'PKU-Alignment/PKU-SafeRLHF-10K' \
#     --reward_names 'beaver_reward,beaver_cost' \
#     --wandb_name 'ppo_qwen2_beaver_reward_eval' \
#     2>&1 | tee ./logs/ppo/ppo_qwen2_beaver_reward_eval.log

# PPO OBJECTIVE-SPECIFIC EVALUATION — beaver_cost
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/eval_ppo_single.py \
    --base_model_name 'Qwen/Qwen2-7B' \
    --ppo_model_name './models/ppo/ppo_qwen2_beaver_cost/best_model/' \
    --dataset_name 'PKU-Alignment/PKU-SafeRLHF-10K' \
    --reward_names 'beaver_reward,beaver_cost' \
    --wandb_name 'ppo_qwen2_beaver_cost_eval' \
    2>&1 | tee ./logs/ppo/ppo_qwen2_beaver_cost_eval.log
