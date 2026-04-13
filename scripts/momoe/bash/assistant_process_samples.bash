PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --simplex_step 0.1 --use_lora False --wandb_name 'new_assistant_2903asample' 2>&1 | tee ./logs/new_assistant_2903asample.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/new_assistant_2903asample/collected_rewards.csv' \
    --wandb_name 'new_assistant_2903bsample' 2>&1 | tee ./logs/new_assistant_2903bsample.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/train_new.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --rewards_csv './results/new/new_assistant_2903asample/collected_rewards.csv' --dataset_csv './results/new/new_assistant_2903bsample/gating_dataset.csv' \
    --loss_mode 'reward' --use_lora False --log_with 'wandb' --wandb_name 'new_assistant_2903csample' 2>&1 | tee ./logs/new_assistant_2903csample.log