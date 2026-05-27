set -e

# PPO objective-specific fine-tuning — PKU-SafeRLHF-10K (beaver), Llama base.
base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_beaver/model/'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'

# --- beaver_reward ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29601 \
    ./scripts/baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'beaver_reward' \
    --wandb_name 'ppo_beaver_reward_2204' \
    2>&1 | tee ./logs/ppo/ppo_beaver_reward_2204.log

# --- beaver_cost ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29601 \
    ./scripts/baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'beaver_cost' \
    --wandb_name 'ppo_beaver_cost_2204' \
    2>&1 | tee ./logs/ppo/ppo_beaver_cost_2204.log

# --- evaluation (optional) ---
# for run in ppo_beaver_reward_2204 ppo_beaver_cost_2204; do
#   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29601 \
#       ./scripts/baselines/fine-tuning/eval_ppo_single.py \
#       --base_model_name "${base_model_name}" \
#       --ppo_model_name "./models/ppo/${run}/best_model/" \
#       --dataset_name "${dataset_name}" \
#       --reward_names 'beaver_reward,beaver_cost' \
#       --wandb_name "${run}_eval" \
#       2>&1 | tee ./logs/ppo/${run}_eval.log
# done
