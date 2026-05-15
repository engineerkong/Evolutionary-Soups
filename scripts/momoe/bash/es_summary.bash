set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'
dataset_name='openai/summarize_from_feedback'
reward_names='summary,faithful,deberta'
eval_prompts=1024
use_dual_front=true
algorithm='nsga4'
population_size=80
num_generations=100
warm_start_path=''
# --------------------
parent_stability_bonus=0.005
parent_stability_cap=0.10
fixed_alpha=1.2
gating_type='per_layer'   # 'per_layer' = GatingNetwork | 'simple' = SimpleGatingNetwork
normalize_fitness=true
# --------------------
run_name='es_summary_1505'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5,6 accelerate launch --main_process_port 29602 \
    ./scripts/momoe/evolutionary_soups.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --algorithm "${algorithm}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --warm_start_path "${warm_start_path}" \
    --parent_stability_bonus "${parent_stability_bonus}" \
    --parent_stability_cap "${parent_stability_cap}" \
    --fixed_alpha "${fixed_alpha}" \
    --gating_type "${gating_type}" \
    --normalize_fitness "${normalize_fitness}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log