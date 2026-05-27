#!/usr/bin/env bash
# Oracle upper-bound pipeline — beaver task (beaver_reward + beaver_cost)
#
# Step 1: collect_rewards_simple_gating.py — sweep all simplex gating coefficients with
#         SimpleMoEForCausalLM (experts stay in memory; merge final hidden states)
# Step 2: build_dataset.py    — find utility-optimal weights per (prompt, preference)
# Step 3: eval_oracle.py      — report oracle vs naive utility comparison
#
# Usage: bash scripts/optimal/bash/run_optimal_beaver.bash

set -e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_beaver_2004/model/'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
reward_names='beaver_reward,beaver_cost'
simplex_step=0.1
save_directory='./results/optimal'
wandb_name='optimal_beaver_simple_1305'

mkdir -p ./logs

# ---------------------------------------------------------------------------
# Step 1: collect per-prompt rewards for all SimpleMoEForCausalLM gating coefficients
# ---------------------------------------------------------------------------
echo "========================================"
echo "Step 1: Collecting rewards (SimpleMoEForCausalLM)"
echo "========================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29602 \
    ./scripts/optimal/collect_rewards_simple_gating.py \
    --base_model_name    "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --reward_names       "${reward_names}" \
    --exp_type           beaver \
    --simplex_step       "${simplex_step}" \
    --split              test \
    --save_directory     "${save_directory}" \
    --run_name           "${wandb_name}" \
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
