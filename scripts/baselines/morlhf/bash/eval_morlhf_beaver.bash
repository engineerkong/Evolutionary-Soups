#!/usr/bin/env bash
# Evaluate MORLHF beaver checkpoints (PKU-Alignment/PKU-SafeRLHF-10K, 2 rewards: beaver_reward, beaver_cost)
# Run from MOMoE/ directory:
#   bash scripts/baselines/morlhf/bash/eval_morlhf_beaver.bash
set -e

base_model_name='meta-llama/Llama-2-7b-hf'
reward_names='beaver_reward,beaver_cost'
batch_size=64
train_run='morlhf_beaver_2704'
save_dir="./results/morlhf/eval_${train_run}"
log_file="./logs/morlhf/eval_${train_run}.log"

mkdir -p ./logs/morlhf

preferences=(
    "0.0,1.0"
    "0.1,0.9"
    "0.2,0.8"
    "0.3,0.7"
    "0.4,0.6"
    "0.5,0.5"
    "0.6,0.4"
    "0.7,0.3"
    "0.8,0.2"
    "0.9,0.1"
    "1.0,0.0"
)

for pref in "${preferences[@]}"; do
    pref_tag="${pref//,/_}"
    checkpoint_path="./results/morlhf/${train_run}_pref${pref_tag}/best_model"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 \
        --main_process_port 29806 \
        ./scripts/baselines/morlhf/eval_morlhf.py \
        --base_model_name "${base_model_name}" \
        --checkpoint_path "${checkpoint_path}" \
        --reward_names "${reward_names}" \
        --batch_size "${batch_size}" \
        --exp_type "beaver" \
        --save_directory "${save_dir}" \
        --wandb_name "pref${pref_tag}" \
        2>&1 | tee -a "${log_file}"
done
