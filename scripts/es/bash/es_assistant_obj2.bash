base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model'
dataset_name='Anthropic/hh-rlhf'
reward_names='harmless,helpful'
eval_prompts=128
use_dual_front=false
algorithm='nsgaii'
population_size=20
num_generations=100
# --------------------
parent_stability_bonus=0.0
parent_stability_cap=0.0
fixed_alpha=1.0
normalize_fitness=false
# --------------------
run_name='es_assistant_obj2_nsgaii_1105_2'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29606 \
    ./scripts/momoe/evolutionary_soups.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --use_dual_front "${use_dual_front}" \
    --algorithm "${algorithm}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --parent_stability_bonus "${parent_stability_bonus}" \
    --parent_stability_cap "${parent_stability_cap}" \
    --fixed_alpha "${fixed_alpha}" \
    --normalize_fitness "${normalize_fitness}"\
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log