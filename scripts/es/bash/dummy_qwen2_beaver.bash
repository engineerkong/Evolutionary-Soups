base_model_name='Qwen/Qwen2-7B'
expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'
reward_names='beaver_reward,beaver_cost'
population_size=20
fixed_alpha=1.2
seed=8888

# --------------------
# per_layer (GatingNetwork)
run_name='dummy_qwen2_beaver_per_layer'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 29603 \
    ./scripts/momoe/dummy.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --reward_names "${reward_names}" \
    --population_size "${population_size}" \
    --fixed_alpha "${fixed_alpha}" \
    --gating_type per_layer \
    --seed "${seed}" \
    --save_directory "./models/ES/" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log

# # --------------------
# # simple (SimpleGatingNetwork)
# run_name_simple='dummy_qwen2_beaver_simple'

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 29603 \
#     ./scripts/momoe/dummy.py \
#     --base_model_name "${base_model_name}" \
#     --expert_model_paths ${expert_model_paths} \
#     --reward_names "${reward_names}" \
#     --population_size "${population_size}" \
#     --fixed_alpha "${fixed_alpha}" \
#     --gating_type simple \
#     --seed "${seed}" \
#     --save_directory "./models/ES/" \
#     --run_name "${run_name_simple}" \
#     2>&1 | tee ./logs/${run_name_simple}.log
