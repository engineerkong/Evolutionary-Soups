PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --simplex_step 0.1 --use_lora False --wandb_name 'new_assistant_2903a' 2>&1 | tee ./logs/new_assistant_2903a.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/new_assistant_2903a/collected_rewards.csv' \
    --wandb_name 'new_assistant_2903b' 2>&1 | tee ./logs/new_assistant_2903b.log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/train_new.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --rewards_csv './results/new/new_assistant_1403a/collected_rewards.csv' \
    --dataset_csv './results/new/new_assistant_1403b/gating_dataset.csv' --loss_mode 'reward' --use_lora False --log_with 'wandb' --wandb_name 'new_assistant_2903c' 2>&1 | tee ./logs/new_assistant_2903c.log