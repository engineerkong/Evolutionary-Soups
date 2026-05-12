#!/usr/bin/env bash
# Oracle upper-bound pipeline — assistant task (harmless + helpful)
#
# Step 1: collect_rewards.py  — run all 11 blended models on test set, save per-prompt rewards
# Step 2: build_dataset.py    — find utility-optimal weights per (prompt, preference)
# Step 3: eval_oracle.py      — report oracle vs naive utility comparison
#
# Usage: bash scripts/optimal/bash/run_optimal_assistant.bash

set -e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_assistant_2701/model/'
expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'
reward_names='harmless,helpful,humor'
simplex_step=0.2
save_directory='./results/optimal'
wandb_name='optimal_assistant_1205'

mkdir -p ./logs

# ---------------------------------------------------------------------------
# Step 1: collect per-prompt rewards for all merging coefficients
# ---------------------------------------------------------------------------
echo "========================================"
echo "Step 1: Collecting rewards"
echo "========================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29601 \
    ./scripts/optimal/collect_rewards.py \
    --base_model_name   "${base_model_name}" \
    --sft_model_name    "${sft_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --reward_names      "${reward_names}" \
    --simplex_step      "${simplex_step}" \
    --split             test \
    --save_directory    "${save_directory}" \
    --wandb_name        "${wandb_name}" \
    2>&1 | tee ./logs/${wandb_name}_collect.log

# ---------------------------------------------------------------------------
# Step 2: build gating dataset (optimal weights per prompt per preference)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 2: Building gating dataset"
echo "========================================"

python ./scripts/optimal/build_dataset.py \
    --rewards_csv    "${save_directory}/${wandb_name}/collected_rewards.csv" \
    --reward_names   "${reward_names}" \
    --simplex_step   "${simplex_step}" \
    --save_directory "${save_directory}" \
    --wandb_name     "${wandb_name}" \
    2>&1 | tee ./logs/${wandb_name}_build.log

# ---------------------------------------------------------------------------
# Step 3: oracle vs naive evaluation
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 3: Oracle evaluation"
echo "========================================"

python ./scripts/optimal/eval_oracle.py \
    --rewards_csv  "${save_directory}/${wandb_name}/collected_rewards.csv" \
    --reward_names "${reward_names}" \
    --simplex_step "${simplex_step}" \
    2>&1 | tee ./logs/${wandb_name}_oracle.log
