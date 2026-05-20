#!/bin/bash
# MOD evaluation — PKU-Alignment/PKU-SafeRLHF-10K (beaver_reward + beaver_cost)

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29604 \
    scripts/baselines/mod/eval_mod.py \
    --base_model_name   'meta-llama/Llama-2-7b-hf' \
    --expert_model_paths './models/ppo/ppo_beaver_reward_2204/best_model' \
                         './models/ppo/ppo_beaver_cost_2204/best_model' \
    --dataset_name      'PKU-Alignment/PKU-SafeRLHF-10K' \
    --reward_names      'beaver_reward,beaver_cost' \
    --num_pref_samples  11 \
    --wandb_name        'mod_beaver' \
    2>&1 | tee ./logs/mod/eval_mod_beaver.log
