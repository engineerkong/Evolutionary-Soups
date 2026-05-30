set -e

# PPO objective-specific fine-tuning — Anthropic/hh-rlhf (assistant), Llama base.
base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_assistant/model/'
dataset_name='Anthropic/hh-rlhf'

# --- harmless ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29602 \
    ./baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'harmless' \
    --wandb_name 'ppo_assistant_harmless_2104' \
    2>&1 | tee ./logs/ppo/ppo_assistant_harmless_2104.log

# --- helpful ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29602 \
    ./baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'helpful' \
    --wandb_name 'ppo_assistant_helpful_2104' \
    2>&1 | tee ./logs/ppo/ppo_assistant_helpful_2104.log

# --- humor ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29602 \
    ./baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'humor' \
    --wandb_name 'ppo_assistant_humor_2104' \
    2>&1 | tee ./logs/ppo/ppo_assistant_humor_2104.log

# --- evaluation (optional) ---
# for run in ppo_assistant_harmless_2104 ppo_assistant_helpful_2104 ppo_assistant_humor_2104; do
#   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29602 \
#       ./baselines/fine-tuning/eval_ppo_single.py \
#       --base_model_name "${base_model_name}" \
#       --ppo_model_name "./models/ppo/${run}/best_model/" \
#       --dataset_name "${dataset_name}" \
#       --reward_names 'harmless,helpful,humor' \
#       --wandb_name "${run}_eval" \
#       2>&1 | tee ./logs/ppo/${run}_eval.log
# done
