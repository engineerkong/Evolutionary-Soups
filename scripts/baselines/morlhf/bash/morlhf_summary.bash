#!/usr/bin/env bash
# MORLHF — summarization task (openai/summarize_from_feedback, 3 rewards: summary, faithful, deberta)
# Trains one model per preference point on the 3-objective simplex (step 0.2).
# 21 points total. Run from MOMoE/ directory:
#   bash scripts/baselines/morlhf/bash/morlhf_summary.bash
set -e

export BNB_CUDA_VERSION=128

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_summary_3001/model/'
reward_names='summary,faithful,deberta'
save_dir='./results/morlhf/'

mkdir -p ./logs/morlhf

# All simplex corners/edges/interior with step 0.2 (21 points)
preferences=(
    "0.0,0.0,1.0"
    "0.0,0.2,0.8"
    "0.0,0.4,0.6"
    "0.0,0.6,0.4"
    "0.0,0.8,0.2"
    "0.0,1.0,0.0"
    "0.2,0.0,0.8"
    "0.2,0.2,0.6"
    "0.2,0.4,0.4"
    "0.2,0.6,0.2"
    "0.2,0.8,0.0"
    "0.4,0.0,0.6"
    "0.4,0.2,0.4"
    "0.4,0.4,0.2"
    "0.4,0.6,0.0"
    "0.6,0.0,0.4"
    "0.6,0.2,0.2"
    "0.6,0.4,0.0"
    "0.8,0.0,0.2"
    "0.8,0.2,0.0"
    "1.0,0.0,0.0"
)

for pref in "${preferences[@]}"; do
    pref_tag="${pref//,/_}"
    run_name="morlhf_summary_0105_pref${pref_tag}"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 \
        --main_process_port 29801 \
        ./scripts/baselines/morlhf/morlhf.py \
        --base_model_name "${base_model_name}" \
        --sft_model_name "${sft_model_name}" \
        --reward_names "${reward_names}" \
        --exp_type "summary" \
        --preference "${pref}" \
        --epochs 3 \
        --learning_rate 5e-6 \
        --mini_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --target 6 \
        --init_kl_coef 0.05 \
        --save_directory "${save_dir}" \
        --wandb_name "morlhf_summary_0105" \
        2>&1 | tee ./logs/morlhf/${run_name}.log
done
