#!/usr/bin/env bash
# MOMoE NSGA-II evaluation — summary task
# Usage: bash bash/nsgaii_summary_test.bash

set -e

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/summary_sft/model/'
expert_model_paths='./models/ppo/summary_ppo_summary_0302/batch_307/ ./models/ppo/summary_ppo_faithful_0302/batch_307/'
gating_paths='./models/nsgaii/nsgaii_summary_1704/gen_0040/'
reward_names='summary,faithful'
dataset_name='openai/summarize_from_feedback'
pref_step=0.2
run_name='nsgaii_summary_test_2004'

mkdir -p ./logs

CUDA_VISIBLE_DEVICES=4,5 accelerate launch \
    ./scripts/momoe/nsgaii_test.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_paths "${gating_paths}" \
    --reward_names "${reward_names}" \
    --dataset_name "${dataset_name}" \
    --run_name "${run_name}" \
    --pref_step "${pref_step}" \
    2>&1 | tee ./logs/${run_name}.log
