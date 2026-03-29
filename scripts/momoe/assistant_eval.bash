CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --split 'test' --simplex_step 0.1 --wandb_name 'new_assistant_2703atest' 2>&1 | tee ./logs/new_assistant_2703atest.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/new_assistant_2703atest/collected_rewards.csv' \
    --wandb_name 'new_assistant_2703btest' 2>&1 | tee ./logs/new_assistant_2703btest.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/eval_new.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --checkpoint_path './models/new/new_assistant_2703c/epoch_73_step_334486' --dataset_csv_test './results/new/new_assistant_2703btest/gating_dataset.csv' --num_pref_samples 11 \
    --wandb_name 'new_assistant_2703etest' 2>&1 | tee ./logs/new_assistant_2703etest73.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/eval_new.py --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --checkpoint_path './models/new/new_assistant_2703c/epoch_38_step_174116' --dataset_csv_test './results/new/new_assistant_2703b/gating_dataset.csv' --num_pref_samples 11 \
    --wandb_name 'new_assistant_2703etrain' 2>&1 | tee ./logs/new_assistant_2703etrain.log