set -e

# === Setting (active: Llama / Beaver; alternatives commented per line) ======
base_model_name='meta-llama/Llama-2-7b-hf'
# base_model_name='Qwen/Qwen2-7B'                                                                                                                                          # qwen2

expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
# expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'                                                       # qwen2 beaver
# expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'  # assistant
# expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'      # summary

reward_names='beaver_reward,beaver_cost'
# reward_names='harmless,helpful,humor'                                                                                                                                    # assistant
# reward_names='summary,faithful,deberta'                                                                                                                                  # summary

population_size=20
# population_size=40                                                                                                                                                       # assistant / summary
# ===========================================================================

fixed_alpha=1.2
gating_type='per_layer'        # 'per_layer' = GatingNetwork | 'simple' = SimpleGatingNetwork
seed=8888
# --------------------
run_name='dummy_beaver'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29605 \
    ./scripts/es/_dummy.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --reward_names "${reward_names}" \
    --population_size "${population_size}" \
    --fixed_alpha "${fixed_alpha}" \
    --gating_type "${gating_type}" \
    --seed "${seed}" \
    --save_directory "./models/ES/" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
