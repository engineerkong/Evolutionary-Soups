set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'
dataset_name='Anthropic/hh-rlhf'
reward_names='harmless,helpful,humor'
eval_prompts=4096
population_size=40
num_generations=100
use_reward_map=False
fitness_ema_alpha=1.0
run_name='nsgaii_assistant_2604'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    ./scripts/momoe/nsgaii.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --use_reward_map "${use_reward_map}" \
    --fitness_ema_alpha "${fitness_ema_alpha}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log