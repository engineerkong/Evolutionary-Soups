set -e

# === Setting (active: Llama / Beaver; alternatives commented per line) ======
base_model_name='meta-llama/Llama-2-7b-hf'
# base_model_name='Qwen/Qwen2-7B'                                                                                                                                          # qwen2

expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
# expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'                                                       # qwen2 beaver
# expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'  # assistant
# expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'      # summary

gating_dataset_csv='./results/optimal/optimal_beaver_1205/gating_dataset.csv'
# gating_dataset_csv='./results/optimal/optimal_assistant/gating_dataset.csv'                                                                                              # assistant
# gating_dataset_csv='./results/optimal/optimal_summary/gating_dataset.csv'                                                                                                # summary
# ===========================================================================

# GatingNetwork (per-layer) checkpoints: e.g. a dummy random-init population
gating_paths_per_layer='./models/ES/dummy_beaver_per_layer/final'
# SimpleGatingNetwork checkpoints: e.g. a dummy random-init population
gating_paths_simple='./models/ES/dummy_beaver_simple/final'
# --------------------
max_prompts=200
batch_size=16
max_prompt_len=256
output_json='./results/analyze_hidden_variance_beaver.json'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port 29608 \
    ./evolutionary/_analyze_variance.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_dataset_csv "${gating_dataset_csv}" \
    --gating_paths_per_layer ${gating_paths_per_layer} \
    --gating_paths_simple ${gating_paths_simple} \
    --max_prompts "${max_prompts}" \
    --batch_size "${batch_size}" \
    --max_prompt_len "${max_prompt_len}" \
    --output_json "${output_json}" \
    2>&1 | tee ./logs/analyze_variance_beaver.log
