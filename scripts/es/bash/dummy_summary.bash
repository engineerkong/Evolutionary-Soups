base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'
reward_names='summary,faithful,deberta'
population_size=80
fixed_alpha=1.2
seed=8888
run_name='dummy_summary_per_layer'

# # --------------------
# # Step 1: initialise random population

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 29603 \
#     ./scripts/momoe/dummy.py \
#     --base_model_name "${base_model_name}" \
#     --expert_model_paths ${expert_model_paths} \
#     --reward_names "${reward_names}" \
#     --population_size "${population_size}" \
#     --fixed_alpha "${fixed_alpha}" \
#     --gating_type per_layer \
#     --seed "${seed}" \
#     --save_directory "./models/ES/" \
#     --run_name "${run_name}" \
#     2>&1 | tee ./logs/${run_name}.log

# --------------------
# Step 2: evaluate population

test_run_name='dummy_summary_test'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 29603 \
    ./scripts/momoe/nsgaii_test.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_paths "./models/ES/${run_name}/final" \
    --dataset_name 'openai/summarize_from_feedback' \
    --reward_names "${reward_names}" \
    --eval_prompts 0 \
    --pref_step 0.2 \
    --norm_rewards './results/optimal/optimal_summary_1205/reward_norm.json' \
    --run_name "${test_run_name}" \
    2>&1 | tee ./logs/${test_run_name}.log
