#!/usr/bin/env bash

set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model'
gating_paths='./models/ES/es_assistant_obj2_nsgaii_1105_2/gen_0015'
dataset_name='Anthropic/hh-rlhf'
use_train_split=false
reward_names='harmless,helpful'
eval_prompts=0
pref_step=0.1
run_name='nsgaii_assistant_obj2_nsgaii_testprompts_1105_2'

mkdir -p ./logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29606 \
    ./scripts/momoe/nsgaii_test.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_paths "${gating_paths}" \
    --dataset_name "${dataset_name}" \
    --use_train_split "${use_train_split}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --pref_step "${pref_step}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
