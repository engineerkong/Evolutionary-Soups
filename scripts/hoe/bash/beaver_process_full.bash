#!/usr/bin/env bash
# HoE Stage 1 + router evaluation for BeaverTails (PKU-SafeRLHF-10K).
# Runs supervised router pre-training then evaluates the router across
# the preference simplex (without Stage-2 PPO).
set -e

export BNB_CUDA_VERSION=128

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model,./models/ppo/ppo_beaver_cost_2204/best_model'
reward_names='beaver_reward,beaver_cost'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
num_pref_samples=11
router_run_name='hoe_beaver_router_stage1'
eval_name='hoe_beaver_router_eval'
router_save_dir='./models/hoe_router/'
router_checkpoint="${router_save_dir}${router_run_name}"

mkdir -p ./logs

echo "=== Stage 1: Supervised router pre-training ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
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
    --per_device_batch 16 \
    --grad_accum_steps 2 \
    2>&1 | tee ./logs/${router_run_name}.log

echo "=== Evaluate Stage-1 router across preference simplex ==="
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    --main_process_port 29702 \
    ./scripts/hoe/eval_hoe.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths "${expert_model_paths}" \
    --pretrained_moe_path "${router_checkpoint}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --num_pref_samples "${num_pref_samples}" \
    --save_directory "${router_save_dir}" \
    --wandb_name "${eval_name}" \
    2>&1 | tee ./logs/${eval_name}.log
