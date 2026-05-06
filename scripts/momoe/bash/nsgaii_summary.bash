set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'
dataset_name='openai/summarize_from_feedback'
reward_names='summary,faithful,deberta'
eval_prompts=1024
algorithm='nsgaiii'
n_reference_divisions=5
population_size=80
num_generations=100
# --------------------
ema_alpha_start=1.0
ema_alpha_decay=0.97
fitness_ema_alpha=0.5
archive_size=40
child_penalty=0.97
# α-entmax initial value per layer (1.0=softmax, 2.0=sparsemax, >2=sparser)
alpha_init=1.5
# --------------------
run_name='nsgaiii_summary_entmax_0505'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --main_process_port 29604 \
    ./scripts/momoe/ema.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --algorithm "${algorithm}" \
    --n_reference_divisions "${n_reference_divisions}" \
    --population_size "${population_size}" \
    --num_generations "${num_generations}" \
    --fitness_ema_alpha "${fitness_ema_alpha}" \
    --ema_alpha_start "${ema_alpha_start}" \
    --ema_alpha_decay "${ema_alpha_decay}" \
    --archive_size "${archive_size}" \
    --child_penalty "${child_penalty}" \
    --alpha_init "${alpha_init}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log