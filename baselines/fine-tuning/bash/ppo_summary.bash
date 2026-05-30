set -e

# PPO objective-specific fine-tuning — openai/summarize_from_feedback (summary), Llama base.
base_model_name='meta-llama/Llama-2-7b-hf'
sft_model_name='./models/sft/sft_summary/model/'
dataset_name='openai/summarize_from_feedback'

# --- summary ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29603 \
    ./baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'summary' \
    --wandb_name 'ppo_summary_summary_2104' \
    2>&1 | tee ./logs/ppo/ppo_summary_summary_2104.log

# --- faithful ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29603 \
    ./baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'faithful' \
    --wandb_name 'ppo_summary_faithful_2104' \
    2>&1 | tee ./logs/ppo/ppo_summary_faithful_2104.log

# --- deberta ---
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29603 \
    ./baselines/fine-tuning/ppo.py \
    --base_model_name "${base_model_name}" \
    --sft_model_name "${sft_model_name}" \
    --dataset_name "${dataset_name}" \
    --reward_name 'deberta' \
    --wandb_name 'ppo_summary_deberta_2104' \
    2>&1 | tee ./logs/ppo/ppo_summary_deberta_2104.log

# --- evaluation (optional) ---
# for run in ppo_summary_summary_2104 ppo_summary_faithful_2104 ppo_summary_deberta_2104; do
#   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29603 \
#       ./baselines/fine-tuning/eval_ppo_single.py \
#       --base_model_name "${base_model_name}" \
#       --ppo_model_name "./models/ppo/${run}/best_model/" \
#       --dataset_name "${dataset_name}" \
#       --reward_names 'summary,faithful,deberta' \
#       --wandb_name "${run}_eval" \
#       2>&1 | tee ./logs/ppo/${run}_eval.log
# done
