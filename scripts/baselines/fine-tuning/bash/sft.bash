set -e

# === Setting (active: Llama / Beaver; alternatives commented per line) ======
base_model_name='meta-llama/Llama-2-7b-hf'

dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
# dataset_name='Anthropic/hh-rlhf'                  # assistant
# dataset_name='openai/summarize_from_feedback'     # summary

reward_names='beaver_reward,beaver_cost'            # eval only
# reward_names='harmless,helpful,humor'             # assistant
# reward_names='summary,faithful,deberta'           # summary

run_name='sft_beaver'
# run_name='sft_assistant'                          # assistant
# run_name='sft_summary'                            # summary
# ===========================================================================

# --- SFT training ---
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 \
    ./scripts/baselines/fine-tuning/sft.py \
    --base_model_name "${base_model_name}" \
    --dataset_name "${dataset_name}" \
    --save_directory './models/sft/' \
    --wandb_name "${run_name}" \
    2>&1 | tee ./logs/sft/${run_name}.log

# --- SFT evaluation (optional) ---
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 \
#     ./scripts/baselines/fine-tuning/eval_sft.py \
#     --base_model_name "${base_model_name}" \
#     --sft_model_name "./models/sft/${run_name}/model/" \
#     --dataset_name "${dataset_name}" \
#     --reward_names "${reward_names}" \
#     --wandb_name "${run_name}_eval" \
#     2>&1 | tee ./logs/sft/${run_name}_eval.log
