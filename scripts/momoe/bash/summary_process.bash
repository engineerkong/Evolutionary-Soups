CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29602 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/summary_sft/model/' \
    --expert_model_paths './models/ppo/summary_ppo_summary_0302/batch_307/' './models/ppo/summary_ppo_faithful_0302/batch_307/' './models/ppo/summary_ppo_deberta_0302/batch_307/' \
    --reward_names 'summary,faithful,deberta' --exp_type 'summary' --use_lora True --wandb_name 'new_summary_2703a' 2>&1 | tee ./logs/new_summary_2703a.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29602 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/new_summary_2703a/collected_rewards.csv' \
    --reward_names 'summary,faithful,deberta' --num_pref_samples 21 --wandb_name 'new_summary_2703b' 2>&1 | tee ./logs/new_summary_2703b.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29602 ./scripts/momoe/train_new.py --sft_model_name './models/sft/summary_sft/model/' \
    --expert_model_paths './models/ppo/summary_ppo_summary_0302/batch_307/' './models/ppo/summary_ppo_faithful_0302/batch_307/' './models/ppo/summary_ppo_deberta_0302/batch_307/' \
    --reward_names 'summary,faithful,deberta' --loss_mode 'reward' --dataset_csv './results/new/new_summary_2703b/gating_dataset.csv' --rewards_csv './results/new/new_summary_2703a/collected_rewards.csv' \
    --log_with 'wandb' --wandb_name 'new_summary_2703c' 2>&1 | tee ./logs/new_summary_2703c.log