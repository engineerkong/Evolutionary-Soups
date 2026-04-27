CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/ppo.py \
    --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' \
    --reward_name 'steer_helpfulness' --wandb_name 'ppo_steer2_helpfulness_2504' \
    2>&1 | tee ./logs/ppo/ppo_steer2_helpfulness_2504.log

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/ppo.py \
    --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' \
    --reward_name 'steer_correctness' --wandb_name 'ppo_steer2_correctness_2504' \
    2>&1 | tee ./logs/ppo/ppo_steer2_correctness_2504.log

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/ppo.py \
    --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' \
    --reward_name 'steer_coherence' --wandb_name 'ppo_steer2_coherence_2504' \
    2>&1 | tee ./logs/ppo/ppo_steer2_coherence_2504.log

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/ppo.py \
    --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' \
    --reward_name 'steer_complexity' --wandb_name 'ppo_steer2_complexity_2504' \
    2>&1 | tee ./logs/ppo/ppo_steer2_complexity_2504.log

CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29607 ./scripts/fine-tuning/ppo.py \
    --sft_model_name './models/sft/sft_steer2_2004/model/' --dataset_name 'nvidia/HelpSteer2' \
    --reward_name 'steer_verbosity' --wandb_name 'ppo_steer2_verbosity_2504' \
    2>&1 | tee ./logs/ppo/ppo_steer2_verbosity_2504.log
