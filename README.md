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

PPO REWARDED-SOUPS EVALUATION (--main_process_port 29601)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_ppo_rs.py --ppo_model_name1 './models/ppo/assistant_ppo_harmless/batch_832/' --ppo_model_name2 './models/ppo/assistant_ppo_helpful/batch_832/' --exp_type 'assistant' --reward_names 'harmless,helpful' --wandb_name 'assistant_ppo_rs' 2>&1 | tee ./logs/ppo_assistant_rs.log
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/eval_ppo_rs.py --ppo_model_name1 './models/ppo/summary_ppo_summary/batch_307/' --ppo_model_name2 './models/ppo/summary_ppo_faithful/batch_307/' --ppo_model_name3 './models/ppo/summary_ppo_deberta/batch_307/' --exp_type 'summary' --reward_names 'summary,faithful,deberta' --wandb_name 'summary_ppo_rs' 2>&1 | tee ./logs/ppo_summary_rs.log
```