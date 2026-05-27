#!/usr/bin/env bash
# RiC — beaver task (PKU-Alignment/PKU-SafeRLHF-10K, rewards: beaver_reward, beaver_cost)
# Requires safe-rlhf: pip install git+https://github.com/PKU-Alignment/safe-rlhf.git
# Pipeline: 1) prepare scored dataset → 2) train → 3) evaluate
# Run from MOMoE/ directory: bash scripts/baselines/ric/bash/ric_beaver.bash
set -e

export BNB_CUDA_VERSION=128
export CUDA_VISIBLE_DEVICES=7

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_beaver_2004/model/'
reward_names='beaver_reward,beaver_cost'
dataset_path='./results/datasets/ric_beaver_rewardcost.hf'
save_dir='./results/ric/'
run_name='ric_beaver_2205'
eval_name='ric_beaver_eval_2205'

mkdir -p ./logs/ric ./results/datasets

# # ---- Step 1: Prepare training dataset with reward scores ----
# python ./scripts/baselines/ric/prepare_dataset.py \
#     --reward_names "${reward_names}" \
#     --save_directory "${dataset_path}" \
#     --exp_type "beaver" \
#     2>&1 | tee ./logs/ric/${run_name}_prepare.log

# # ---- Step 2: Train RiC ----
# python ./scripts/baselines/ric/main.py \
#     --base_model_name "${base_model_name}" \
#     --sft_model_name "${sft_model_name}" \
#     --reward_names "${reward_names}" \
#     --exp_type "beaver" \
#     --load_in_8bit False \
#     --train_dataset_path "${dataset_path}" \
#     --save_directory "${save_dir}" \
#     --wandb_name "${run_name}" \
#     --training_steps 20000 \
#     --online_training_steps 4000 \
#     --num_online_iterations 1 \
#     2>&1 | tee ./logs/ric/${run_name}.log

# ---- Step 3: Evaluate across preference simplex ----
model_path="${save_dir}${run_name}/model_iter1"

python ./scripts/baselines/ric/evaluation.py \
    --base_model_name "${base_model_name}" \
    --peft_name "${model_path}" \
    --reward_names "${reward_names}" \
    --exp_type "beaver" \
    --save_directory "${save_dir}" \
    --wandb_name "${eval_name}" \
    --num_prefer_points 11 \
    2>&1 | tee ./logs/ric/${eval_name}.log
