#!/usr/bin/env bash
# MOMoE NSGA-II evaluation — summary task
# Usage: bash bash/nsgaii_summary_test.bash

set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'
gating_paths=''
dataset_name='Anthropic/hh-rlhf'
reward_names='harmless,helpful,humor'
eval_prompts=0
pref_step=0.2
run_name='nsgaiii_assistant_test_rs_2604'

mkdir -p ./logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29604 \
    ./scripts/momoe/nsgaii_test.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_paths "${gating_paths}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --pref_step "${pref_step}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
