base_model_name='Qwen/Qwen2-7B'
expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
reward_names='beaver_reward,beaver_cost'
eval_prompts=1024
use_dual_front=true
use_greedy_hvc=true
algorithm='nsgaii'
population_size=20
num_generations=30
warm_start_path=''    # set to a run directory to resume evolution
# --------------------
parent_stability_bonus=0.01
parent_stability_cap=0.10
fixed_alpha=1.2
gating_type='per_layer'   # 'per_layer' = GatingNetwork | 'simple' = SimpleGatingNetwork
normalize_fitness=true
# --------------------
run_name='es_qwen2_beaver'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 \
    ./scripts/momoe/evolutionary_soups.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --use_dual_front "${use_dual_front}" \
    --use_greedy_hvc "${use_greedy_hvc}" \
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
