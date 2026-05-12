#!/usr/bin/env bash
# Evaluate MORLHF summary checkpoints (openai/summarize_from_feedback, 3 rewards: summary, faithful, deberta)
# Run from MOMoE/ directory:
#   bash scripts/baselines/morlhf/bash/eval_morlhf_summary.bash
set -e

base_model_name='meta-llama/Llama-2-7b-hf'
reward_names='summary,faithful,deberta'
batch_size=64
train_run='morlhf_summary_0105'
save_dir="./results/morlhf/eval_${train_run}"
log_file="./logs/morlhf/eval_${train_run}.log"

mkdir -p ./logs/morlhf

# 3-objective simplex, step 0.2 (21 points)
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
    checkpoint_path="./results/morlhf/${train_run}_pref${pref_tag}/best_model"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
        --num_processes 2 \
        --main_process_port 29807 \
        ./scripts/baselines/morlhf/eval_morlhf.py \
        --base_model_name "${base_model_name}" \
        --checkpoint_path "${checkpoint_path}" \
        --reward_names "${reward_names}" \
        --batch_size "${batch_size}" \
        --exp_type "summary" \
        --save_directory "${save_dir}" \
        --wandb_name "pref${pref_tag}" \
        2>&1 | tee -a "${log_file}"
done
