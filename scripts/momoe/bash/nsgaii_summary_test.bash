#!/usr/bin/env bash
# MOMoE NSGA-II evaluation — summary task
# Usage: bash bash/nsgaii_summary_test.bash

set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model/ ./models/ppo/ppo_summary_faithful_2104/best_model/ ./models/ppo/ppo_summary_deberta_2104/best_model/'
gating_paths='./models/ES/es_summary_1405/gen_0025'
dataset_name='openai/summarize_from_feedback'
reward_names='summary,faithful,deberta'
eval_prompts=0
pref_step=0.2
run_name='es_summary_test_gen25_1405'

mkdir -p ./logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 29607 \
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
