CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --simplex_step 0.1 --wandb_name 'new_assistant_2703a' 2>&1 | tee ./logs/new_assistant_2703a.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/new_assistant_2703a/collected_rewards.csv' \
    --wandb_name 'new_assistant_2703b' 2>&1 | tee ./logs/new_assistant_2703b.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/train_new.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --dataset_csv './results/new/new_assistant_2703b/gating_dataset.csv' --loss_mode 'reward' --log_with 'wandb' --wandb_name 'new_assistant_2703c' 2>&1 | tee ./logs/new_assistant_2703c.log