set -e

port=29901
norm_rewards='./results/optimal/optimal_beaver_1205/reward_norm.json'
eval_prompts=0
pref_step=0.1

# ---------------------------------------------------------------------------
# Llama-2 beaver ablations
# ---------------------------------------------------------------------------
base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
reward_names='beaver_reward,beaver_cost'


for variant in es_beaver_abl_no_dualfront; do
    run_name="${variant}_test_csv"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port ${port} \
        ./scripts/momoe/nsgaii_test.py \
        --base_model_name "${base_model_name}" \
        --expert_model_paths ${expert_model_paths} \
        --gating_paths "./models/ES/${variant}/gen_0030" \
        --dataset_name "${dataset_name}" \
        --reward_names "${reward_names}" \
        --norm_rewards "${norm_rewards}" \
        --eval_prompts "${eval_prompts}" \
        --pref_step "${pref_step}" \
        --run_name "${run_name}" \
        2>&1 | tee ./logs/${run_name}.log
    port=$((port + 1))
done

# # ---------------------------------------------------------------------------
# # Qwen2 beaver ablations
# # ---------------------------------------------------------------------------
# base_model_name='Qwen/Qwen2-7B'
# expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'
# dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
# reward_names='beaver_reward,beaver_cost'

# for variant in dummy_qwen2_beaver_per_layer; do
#     run_name="${variant}_test"
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port ${port} \
#         ./scripts/momoe/nsgaii_test.py \
#         --base_model_name "${base_model_name}" \
#         --expert_model_paths ${expert_model_paths} \
#         --gating_paths "./models/ES/${variant}/final" \
#         --dataset_name "${dataset_name}" \
#         --reward_names "${reward_names}" \
#         --norm_rewards "${norm_rewards}" \
#         --eval_prompts "${eval_prompts}" \
#         --pref_step "${pref_step}" \
#         --run_name "${run_name}" \
#         2>&1 | tee ./logs/${run_name}.log
#     port=$((port + 1))
# done
