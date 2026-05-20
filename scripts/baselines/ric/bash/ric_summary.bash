#!/usr/bin/env bash
# RiC — summarization task (openai/summarize_from_feedback, rewards: summary, faithful)
# Pipeline: 1) prepare scored dataset → 2) train → 3) evaluate
# Run from MOMoE/ directory: bash scripts/baselines/ric/bash/ric_summary.bash
set -e

export BNB_CUDA_VERSION=128

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_summary_3001/model/'
reward_names='summary,faithful,deberta'
dataset_path='./datasets/ric_summary_summaryfaithfuldeberta.hf'
save_dir='./results/ric/'
run_name='ric_summary_1905'
eval_name='ric_summary_eval_1905'

mkdir -p ./logs/ric ./datasets

# # ---- Step 1: Prepare training dataset with reward scores ----
# CUDA_VISIBLE_DEVICES=5,6,7 python ./scripts/baselines/ric/prepare_dataset.py \
#     --reward_names "${reward_names}" \
#     --save_directory "${dataset_path}" \
#     --exp_type "summary" \
#     2>&1 | tee ./logs/ric/${run_name}_prepare.log

# ---- Step 2: Train RiC ----
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --main_process_port 29813 \
    ./scripts/baselines/ric/main.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --reward_names "${reward_names}" \
    --exp_type "summary" \
    --load_in_8bit False \
    --train_dataset_path "${dataset_path}" \
    --save_directory "${save_dir}" \
    --wandb_name "${run_name}" \
    --training_steps 20000 \
    --online_training_steps 4000 \
    --num_online_iterations 1 \
    2>&1 | tee ./logs/ric/${run_name}.log

# ---- Step 3: Evaluate across preference simplex ----
model_path="${save_dir}${run_name}/model_iter1"

CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --main_process_port 29814 \
    ./scripts/baselines/ric/evaluation.py \
    --base_model_name "${base_model_name}" \
    --peft_name "${model_path}" \
    --reward_names "${reward_names}" \
    --exp_type "summary" \
    --save_directory "${save_dir}" \
    --wandb_name "${eval_name}" \
    --num_prefer_points 21 \
    2>&1 | tee ./logs/ric/${eval_name}.log
