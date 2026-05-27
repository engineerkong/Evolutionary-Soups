#!/usr/bin/env bash
# HoE full two-stage pipeline for summarization — faithful to the original HoE design.
#
# Stage 1 (train_hoe_router.py): supervised router pre-training via NLLLoss.
#   The router learns to route content to the appropriate expert using
#   one-hot preference vectors, before any PPO.
#
# Stage 2 (train_hoe.py --pretrained_moe_path): PPO fine-tuning with
#   deepcopy(pre-trained MoE) as KL reference — reproducing Mymorlhf_ref.py.
#
# Usage: bash bash/summary_process_full.bash

set -e

export BNB_CUDA_VERSION=128

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_summary_3001/model/'
expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model,./models/ppo/ppo_summary_faithful_2104/best_model,./models/ppo/ppo_summary_deberta_2104/best_model'
reward_names='summary,faithful,deberta'
dataset_name='openai/summarize_from_feedback'
num_pref_samples=21
router_run_name='hoe_summary_router_stage1'
train_name='hoe_summary_train_stage2'
eval_name='hoe_summary_eval_stage2'
router_save_dir='./models/hoe_router/'
ppo_save_dir='./results/hoe/'
router_checkpoint="${router_save_dir}${router_run_name}"
ppo_checkpoint="${ppo_save_dir}${train_name}/model"

mkdir -p ./logs

# ---- Stage 1: Supervised router pre-training ----
echo "=== Stage 1: Supervised router pre-training ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 \
    --main_process_port 29701 \
    ./scripts/hoe/train_hoe_router.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths "${expert_model_paths}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --save_directory "${router_save_dir}" \
    --run_name "${router_run_name}" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_batch 2 \
    --grad_accum_steps 2 \
    2>&1 | tee ./logs/${router_run_name}.log

# ---- Stage 2: PPO with pre-trained router + deepcopy KL reference ----
echo "=== Stage 2: PPO fine-tuning with pre-trained router ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 \
    --main_process_port 29702 \
    ./scripts/hoe/train_hoe.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --expert_model_paths "${expert_model_paths}" \
    --pretrained_moe_path "${router_checkpoint}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --save_directory "${ppo_save_dir}" \
    --wandb_name "${train_name}" \
    2>&1 | tee ./logs/${train_name}.log

# ---- Evaluation ----
echo "=== Evaluation ==="
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --num_processes 2 \
    --main_process_port 29703 \
    ./scripts/hoe/eval_hoe.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths "${expert_model_paths}" \
    --checkpoint_path "${ppo_checkpoint}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --num_pref_samples "${num_pref_samples}" \
    --save_directory "${ppo_save_dir}" \
    --wandb_name "${eval_name}" \
    2>&1 | tee ./logs/${eval_name}.log
