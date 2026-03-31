CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --split 'test' --use_lora False --wandb_name 'new_assistant_3103atest' 2>&1 | tee ./logs/new_assistant_3103atest.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/new_assistant_3103atest/collected_rewards.csv' \
    --wandb_name 'new_assistant_3103btest' 2>&1 | tee ./logs/new_assistant_3103btest.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/eval_new.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --checkpoint_path '' --dataset_csv_test './results/new/new_assistant_3103btest/gating_dataset.csv' --num_pref_samples 11 \
    --use_reward_features False --use_lora False --wandb_name 'new_assistant_3103etest' 2>&1 | tee ./logs/new_assistant_3103etest.log