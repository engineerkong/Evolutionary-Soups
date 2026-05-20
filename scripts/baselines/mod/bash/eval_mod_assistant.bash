#!/bin/bash
# MOD evaluation — Anthropic/hh-rlhf (harmless + helpful)

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 \
    scripts/baselines/mod/eval_mod.py \
    --base_model_name   'meta-llama/Llama-2-7b-hf' \
    --expert_model_paths './models/ppo/ppo_assistant_harmless/best_model' \
                          './models/ppo/ppo_assistant_helpful/best_model' \
    --dataset_name      'Anthropic/hh-rlhf' \
    --reward_names      'harmless,helpful' \
    --num_pref_samples  10 \
    --wandb_name        'mod_assistant' \
    2>&1 | tee ./logs/mod/eval_mod_assistant.log
