#!/usr/bin/env bash
# HoE (Hierarchy of Experts) — full pipeline for the summarization task
# Usage: bash bash/summary_process.bash
#
# Prerequisites:
#   - Base LLaMA-2 model at base_model_name (local path or HF id)
#   - SFT LoRA adapter at sft_model_name (tokenizer files must be here)
#   - Per-objective PPO LoRA adapters listed in expert_model_paths

set -e

export BNB_CUDA_VERSION=128   # force bitsandbytes to use CUDA 12.8 binary on CUDA 12.9 system

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/summary_sft/model/'
expert_model_paths='./models/ppo/summary_ppo_summary_0302/batch_307,./models/ppo/summary_ppo_faithful_0302/batch_307'
reward_names='summary,faithful'
exp_type='summary'
num_pref_samples=6
train_name='hoe_summary_train'
eval_name='hoe_summary_eval_test'
save_dir='./results/hoe/'
checkpoint_path="${save_dir}${train_name}/epoch_0_final"

# mkdir -p ./logs

# # ---- Step 1: Train HoE router via preference-conditioned PPO ----
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
#     --num_processes 2 \
#     --main_process_port 29703 \
#     ./scripts/hoe/train_hoe.py \
#     --base_model_name "${base_model_name}" \
#     --sft_model_name "${sft_model_name}" \
#     --expert_model_paths "${expert_model_paths}" \
#     --reward_names "${reward_names}" \
#     --exp_type "${exp_type}" \
#     --save_directory "${save_dir}" \
#     --wandb_name "${train_name}" \
#     2>&1 | tee ./logs/${train_name}.log

# ---- Step 2: Evaluate trained HoE model across preference simplex ----
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 \
    --main_process_port 29704 \
    ./scripts/hoe/eval_hoe.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --expert_model_paths "${expert_model_paths}" \
    --checkpoint_path "${checkpoint_path}" \
    --reward_names "${reward_names}" \
    --exp_type "${exp_type}" \
    --num_pref_samples "${num_pref_samples}" \
    --save_directory "${save_dir}" \
    --wandb_name "${eval_name}" \
    2>&1 | tee ./logs/${eval_name}.log
