set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
reward_names='beaver_reward,beaver_cost'
eval_prompts=1024
resume_from='./models/nsgaii/nsgaii_beaver_0105/gen_0035'
population_size=20
num_generations=100
sigma_decay=0.97
crowding_threshold=0.02
run_name='nsgaii_beaver_0305_continue'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29606 \
    ./scripts/momoe/ciea.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --resume_from "${resume_from}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --sigma_decay "${sigma_decay}" \
    --crowding_threshold "${crowding_threshold}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log