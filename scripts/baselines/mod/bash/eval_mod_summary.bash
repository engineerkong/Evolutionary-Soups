#!/bin/bash
# MOD evaluation — openai/summarize_from_feedback (summary + faithful)

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29603 \
    scripts/baselines/mod/eval_mod.py \
    --base_model_name   'meta-llama/Llama-2-7b-hf' \
    --expert_model_paths './models/ppo/ppo_summary_summary/best_model' \
                          './models/ppo/ppo_summary_faithful/best_model' \
    --dataset_name      'openai/summarize_from_feedback' \
    --reward_names      'summary,faithful' \
    --num_pref_samples  10 \
    --wandb_name        'mod_summary' \
    2>&1 | tee ./logs/mod/eval_mod_summary.log
