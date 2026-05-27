#!/bin/bash
# MOD evaluation — Anthropic/hh-rlhf (harmless + helpful + humor)

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29605 \
    scripts/baselines/mod/eval_mod.py \
    --base_model_name   'meta-llama/Llama-2-7b-hf' \
    --expert_model_paths './models/ppo/ppo_assistant_harmless_2104/best_model' \
                          './models/ppo/ppo_assistant_helpful_2104/best_model' \
                          './models/ppo/ppo_assistant_humor_2104/best_model' \
    --dataset_name      'Anthropic/hh-rlhf' \
    --reward_names      'harmless,helpful,humor' \
    --num_pref_samples  21 \
    --wandb_name        'mod_assistant_2505' \
    2>&1 | tee ./logs/mod/eval_mod_assistant_2505.log
