CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' --reward_name 'steer_helpfulness' --wandb_name 'ppo_steer_helpfulness_2204' 2>&1 | tee ./logs/ppo/ppo_steer_helpfulness_2204.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' --reward_name 'steer_correctness' --wandb_name 'ppo_steer_correctness_2204' 2>&1 | tee ./logs/ppo/ppo_steer_correctness_2204.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' --reward_name 'steer_coherence' --wandb_name 'ppo_steer_coherence_2204' 2>&1 | tee ./logs/ppo/ppo_steer_coherence_2204.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' --reward_name 'steer_complexity' --wandb_name 'ppo_steer_complexity_2204' 2>&1 | tee ./logs/ppo/ppo_steer_complexity_2204.log

CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29601 ./scripts/fine-tuning/ppo.py --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' --reward_name 'steer_verbosity' --wandb_name 'ppo_steer_verbosity_2204' 2>&1 | tee ./logs/ppo/ppo_steer_verbosity_2204.log
