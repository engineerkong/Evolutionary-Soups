set -e

# === Setting (active: Llama / Beaver; alternatives commented per line) ======
base_model_name='meta-llama/Llama-2-7b-hf'
# base_model_name='Qwen/Qwen2-7B'                                                                                                                                          # qwen2

expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
# expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'                                                       # qwen2 beaver
# expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'  # assistant
# expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'      # summary

dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
# dataset_name='Anthropic/hh-rlhf'                                                                                                                                         # assistant
# dataset_name='openai/summarize_from_feedback'                                                                                                                           # summary

reward_names='beaver_reward,beaver_cost'
# reward_names='harmless,helpful,humor'                                                                                                                                    # assistant
# reward_names='summary,faithful,deberta'                                                                                                                                  # summary

population_size=20
# population_size=80                                                                                                                                                       # assistant / summary
# ===========================================================================

eval_prompts=1024
algorithm='greedy_hvc'         # 'nsgaii' | 'nsgaiii' | 'greedy_hvc'
use_greedy_hvc=true
num_generations=30
warm_start_path=''             # set to a run dir (e.g. ./models/ES/es_beaver_per_layer) to resume
# --------------------
mutation_sigma=0.05
mutation_rate=0.5
sigma_decay=0.99
sigma_min=0.03
# --------------------
fixed_alpha=1.2
gating_type='per_layer'        # 'per_layer' = GatingNetwork | 'simple' = SimpleGatingNetwork
normalize_fitness=true
normalize_rewards=false
# --------------------
run_name='es_beaver_per_layer'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 \
    ./scripts/es/es_train.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --algorithm "${algorithm}" \
    --use_greedy_hvc "${use_greedy_hvc}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --warm_start_path "${warm_start_path}" \
    --mutation_sigma "${mutation_sigma}" \
    --mutation_rate "${mutation_rate}" \
    --sigma_decay "${sigma_decay}" \
    --sigma_min "${sigma_min}" \
    --fixed_alpha "${fixed_alpha}" \
    --gating_type "${gating_type}" \
    --normalize_fitness "${normalize_fitness}" \
    --normalize_rewards "${normalize_rewards}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
