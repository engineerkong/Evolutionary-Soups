### Supervised Fine-Tuning (SFT)
```
# SFT TRAINING
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/sft.py --base_model_name 'meta-llama/Llama-2-7b-hf' --exp_type 'assistant' --wandb_name 'assistant_sft' 2>&1 | tee ./logs/sft_assistant.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/sft.py --base_model_name 'meta-llama/Llama-2-7b-hf' --exp_type 'summary' --wandb_name 'summary_sft' 2>&1 | tee ./logs/sft_summary.log

# SFT EVALUATION
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_sft.py --sft_model_name './models/sft/assistant_sft/model/' --exp_type 'assistant' --reward_names 'harmless,helpful' --wandb_name 'assistant_sft_eval' 2>&1 | tee ./logs/sft_assistant_eval.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_sft.py --sft_model_name './models/sft/summary_sft/model/' --exp_type 'summary' --reward_names 'summary,faithful,deberta' --wandb_name 'summary_sft_eval' 2>&1 | tee ./logs/sft_summary_eval.log
```

### PPO Fine-Tuning (RLFT)
```
# PPO OBJECTIVE-SPECIFIC TRAINING (--main_process_port 29601)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/assistant_sft/model/' --exp_type 'assistant' --reward_name 'harmless' --wandb_name 'assistant_ppo_harmless' 2>&1 | tee ./logs/ppo_harmless.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/summary_sft/model/' --exp_type 'summary' --reward_name 'summary' --wandb_name 'summary_ppo_summary' 2>&1 | tee ./logs/ppo_summary.log

# PPO OBJECTIVE-SPECIFIC EVALUATION (--main_process_port 29601)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_ppo_single.py --sft_model_name './models/sft/assistant_sft/model/' --ppo_model_name './models/ppo/assistant_ppo_harmless/batch_832/' --exp_type 'assistant' --reward_names 'harmless,helpful' --wandb_name 'assistant_ppo_harmless_eval' 2>&1 | tee ./logs/ppo_harmless_eval.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_ppo_single.py --sft_model_name './models/sft/summary_sft/model/' --ppo_model_name './models/ppo/summary_ppo_faithful/batch_307/' --exp_type 'summary' --reward_names 'summary,faithful,deberta' --wandb_name 'summary_ppo_summary_eval' 2>&1 | tee ./logs/ppo_summary_eval.log

# PPO REWARDED-SOUPS EVALUATION (--main_process_port 29601)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_ppo_rs.py --sft_model_name './models/sft/assistant_sft/model/' --ppo_model_name1 './models/ppo/assistant_ppo_harmless/batch_832/' --ppo_model_name2 './models/ppo/assistant_ppo_helpful/batch_832/' --exp_type 'assistant' --reward_names 'harmless,helpful' --wandb_name 'assistant_ppo_rs' 2>&1 | tee ./logs/ppo_assistant_rs.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_ppo_rs.py --sft_model_name './models/sft/summary_sft/model/' --ppo_model_name1 './models/ppo/summary_ppo_summary/batch_307/' --ppo_model_name2 './models/ppo/summary_ppo_faithful/batch_307/' --ppo_model_name3 './models/ppo/summary_ppo_deberta/batch_307/' --exp_type 'summary' --reward_names 'summary,faithful,deberta' --wandb_name 'summary_ppo_rs' 2>&1 | tee ./logs/ppo_summary_rs.log
```

### Multi-Objective Mixture-of-Experts (MOMoE) (--main_process_port 29601)
```
# MOE ROUTER TRAINING
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/train_moe.py --base_model_name './models/sft/assistant_sft/model/' --lora_expert_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --reward_names 'harmless,helpful' --num_pref_samples 5 --exp_type 'assistant' --wandb_name 'assistant_momoe_0203a' 2>&1 | tee ./logs/momoe_assistant_0203a.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/momoe/train_moe.py --base_model_name './models/sft/summary_sft/model/' --lora_expert_paths './models/ppo/summary_ppo_summary_0302/batch_307/' './models/ppo/summary_ppo_faithful_0302/batch_307/' './models/ppo/summary_ppo_deberta_0302/batch_307/' --reward_names 'summary,faithful,deberta' --num_pref_samples 5 --exp_type 'summary' --wandb_name 'summary_momoe' 2>&1 | tee ./logs/momoe_summary.log

# MOMOE EVALUATION
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/eval_moe_v1.py --base_model_name './models/sft/assistant_sft/model/' --lora_expert_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --checkpoint_path './models/momoe/assistant_momoe_0503a/epoch_2_step_1668_final/' --reward_names 'harmless,helpful' --num_pref_samples 5 --num_eval_samples 0 --exp_type 'assistant' --wandb_name 'assistant_momoe_0503a' 2>&1 | tee ./logs/momoe_assistant_eval_0503a.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/momoe/eval_moe.py --base_model_name './models/sft/summary_sft/model/' --lora_expert_paths './models/ppo/summary_ppo_summary_0302/batch_307/' './models/ppo/summary_ppo_faithful_0302/batch_307/' './models/ppo/summary_ppo_deberta_0302/batch_307/' --checkpoint_path './models/momoe/summary_moe_0602x/batch_16000/' --reward_names 'summary,faithful,deberta' --num_pref_samples 5 --num_eval_samples 0 --exp_type 'summary' --wandb_name 'summary_moe_eval_0602' 2>&1 | tee ./logs/momoe_summary_eval_0602x.log
```

```
# QMO
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/train_qmo.py --sft_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --num_pref_samples 11 --log_with 'wandb' --wandb_name 'qmo_assistant_1303b' 2>&1 | tee ./logs/qmo_assistant_1303b.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/eval_qmo.py --base_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --manual_expert_weights '0.0,1.0' --num_pref_samples 1 --wandb_name 'qmo_assistant_eval_1303c' 2>&1 | tee ./logs/qmo_assistant_eval_1303c.log
```

```
# NEW
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --wandb_name 'new_assistant_1403a' 2>&1 | tee ./logs/new_assistant_1403a.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/collect_rewards_mg.py --sft_model_name './models/sft/summary_sft/model/' --expert_model_paths './models/ppo/summary_ppo_summary_0302/batch_307/' './models/ppo/summary_ppo_faithful_0302/batch_307/' './models/ppo/summary_ppo_deberta_0302/batch_307/' --reward_names 'summary,faithful,deberta' --exp_type 'summary' --wandb_name 'new_summary_1803a' 2>&1 | tee ./logs/new_summary_1803a.log
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29601 ./scripts/momoe/collect_rewards.py --sft_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --batch_size 64 --block_mode 'custom' --simplex_step 0.5 --wandb_name 'new_assistant_block_2003a' 2>&1 | tee ./logs/new_assistant_block_2003a.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/build_dataset.py --rewards_csv './results/new/data/new_assistant_1403/collected_rewards.csv' --wandb_name 'new_assistant_1403b' 2>&1 | tee ./logs/new_assistant_1403b.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/train_new.py --sft_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --dataset_csv './results/new/data/new_assistant_1403b/gating_dataset.csv' --rewards_csv './results/new/data/new_assistant_1403a/collected_rewards.csv' --log_with 'wandb' --wandb_name 'new_assistant_1403c' 2>&1 | tee ./logs/new_assistant_1403c.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/test_new.py --sft_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --checkpoint_path './models/new/new_assistant_1403c/epoch_10_step_22930' --dataset_csv './results/new/data/new_assistant_1403b/gating_dataset.csv' --rewards_csv './results/new/data/new_assistant_1403a/collected_rewards.csv' --wandb_name 'new_assistant_1403d' 2>&1 | tee ./logs/new_assistant_1403d.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 ./scripts/momoe/eval_new.py --sft_model_name './models/sft/assistant_sft/model/' --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' './models/ppo/assistant_ppo_helpful_2701/batch_832/' --checkpoint_path './models/new/new_assistant_1403c/epoch_10_step_22930' --rewards_csv_test './results/new/data/new_assistant_1403a_test/collected_rewards.csv' --gating_dataset_test './results/new/new_assistant_1403b_test/gating_dataset.csv' --wandb_name 'new_assistant_1403_test5' 2>&1 | tee ./logs/new_assistant_1403_test5.log
```