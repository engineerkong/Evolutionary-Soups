### Supervised Fine-Tuning (SFT)
```
# SFT TRAINING
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/sft/sft.py --base_model_name 'meta-llama/Llama-2-7b-hf' 2>&1 | tee ./logs/sft.log

# SFT EVALUATION
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/sft/eval_multiobjective.py --sft_model_name './models/sft/assistant_sft/model/' --reward_names 'harmless,helpful' 2>&1 | tee ./logs/sft_eval.log
```

### PPO Fine-Tuning (RLFT)
```
# PPO HARMLESS TRAINING
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/ppo/ppo.py --sft_model_name './models/sft/assistant_sft/model/' --reward_name 'harmless' --wandb_name 'assistant_ppo_harmless' 2>&1 | tee ./logs/ppo_harmless.log

# PPO HELPFUL TRAINING
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo/ppo.py --sft_model_name './models/sft/assistant_sft/model/' --reward_name 'helpful' --wandb_name 'assistant_ppo_helpful' 2>&1 | tee ./logs/ppo_helpful.log

# PPO HARMLESS EVALUATION
CUDA_VISIBLE_DEVICES=0,1 accelerate launch ./scripts/fine-tuning/ppo/eval_ppo_single_model.py --ppo_model_name './models/ppo/assistant_ppo_harmless/batch_400/' --sft_model_name './models/sft/assistant_sft/model/' --reward_name 'harmless,helpful' --wandb_name 'assistant_ppo_harmless_eval' 2>&1 | tee ./logs/ppo_harmless_eval.log

# PPO HELPFUL EVALUATION
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo/eval_ppo_single_model.py --ppo_model_name './models/ppo/assistant_ppo_helpful/batch_400/' --sft_model_name './models/sft/assistant_sft/model/' --reward_name 'harmless,helpful' --wandb_name 'assistant_ppo_helpful_eval' 2>&1 | tee ./logs/ppo_helpful_eval.log
```