set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'
dataset_name='openai/summarize_from_feedback'
reward_names='summary,faithful,deberta'
eval_prompts=4096
algorithm='nsgaiii'
n_reference_divisions=5
population_size=40
num_generations=100
use_reward_map=False
fitness_ema_alpha=1.0
run_name='nsgaiii_summary_2804'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29603 \
    ./scripts/momoe/nsgaii.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --algorithm "${algorithm}" \
    --n_reference_divisions "${n_reference_divisions}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --use_reward_map "${use_reward_map}" \
    --fitness_ema_alpha "${fitness_ema_alpha}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log