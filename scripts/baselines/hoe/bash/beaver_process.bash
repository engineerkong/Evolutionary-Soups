#!/usr/bin/env bash
# HoE (Hierarchy of Experts) — full pipeline for the beaver task
# Usage: bash bash/beaver_process.bash
#
# Prerequisites:
#   - Base LLaMA-2 model at base_model_name (local path or HF id)
#   - SFT LoRA adapter at sft_model_name (tokenizer files must be here)
#   - Per-objective PPO LoRA adapters listed in expert_model_paths

set -e

export BNB_CUDA_VERSION=128   # force bitsandbytes to use CUDA 12.8 binary on CUDA 12.9 system

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_beaver_2004/model/'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model,./models/ppo/ppo_beaver_cost_2204/best_model'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
reward_names='beaver_reward,beaver_cost'
num_pref_samples=11
train_name='hoe_beaver_train_2304'
eval_name='hoe_beaver_eval_2304'
save_dir='./results/hoe/'
checkpoint_path="${save_dir}${train_name}/model"

mkdir -p ./logs

# # ---- Step 1: Train HoE router via preference-conditioned PPO ----
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
#     --num_processes 2 \
#     --main_process_port 29705 \
#     ./scripts/baselines/hoe/train_hoe.py \
#     --base_model_name "${base_model_name}" \
#     --sft_model_name "${sft_model_name}" \
#     --expert_model_paths "${expert_model_paths}" \
#     --dataset_name "${dataset_name}" \
#     --reward_names "${reward_names}" \
#     --save_directory "${save_dir}" \
#     --wandb_name "${train_name}" \
#     2>&1 | tee ./logs/${train_name}.log

# ---- Step 2: Evaluate trained HoE model across preference simplex ----
CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    --num_processes 2 \
    --main_process_port 29706 \
    ./scripts/baselines/hoe/eval_hoe.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths "${expert_model_paths}" \
    --checkpoint_path "${checkpoint_path}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --num_pref_samples "${num_pref_samples}" \
    --save_directory "${save_dir}" \
    --wandb_name "${eval_name}" \
    2>&1 | tee ./logs/${eval_name}.log
