#!/usr/bin/env bash
# MORLHF — beaver task (PKU-Alignment/PKU-SafeRLHF-10K, 2 rewards: beaver_reward, beaver_cost)
# Trains one model per preference point, step 0.1 (11 points).
# Requires safe-rlhf: pip install git+https://github.com/PKU-Alignment/safe-rlhf.git
# Run from MOMoE/ directory:
#   bash baselines/morlhf/bash/morlhf_beaver.bash
set -e

export BNB_CUDA_VERSION=128

base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_beaver_2004/model/'
reward_names='beaver_reward,beaver_cost'
save_dir='./results/morlhf/'

mkdir -p ./logs/morlhf

# 2-objective simplex, step 0.1 (11 points, w_reward + w_cost = 1)
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
    run_name="morlhf_beaver_2704_pref${pref_tag}"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --num_processes 2 \
        --main_process_port 29805 \
        ./baselines/morlhf/morlhf.py \
        --base_model_name "${base_model_name}" \
        --sft_model_name "${sft_model_name}" \
        --reward_names "${reward_names}" \
        --exp_type "beaver" \
        --preference "${pref}" \
        --epochs 5 \
        --learning_rate 1e-5 \
        --mini_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --target 6 \
        --init_kl_coef 0.1 \
        --save_directory "${save_dir}" \
        --wandb_name "morlhf_beaver_2704" \
        2>&1 | tee ./logs/morlhf/${run_name}.log
done
