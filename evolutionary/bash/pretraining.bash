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

# Per-prompt optimal-weight CSV produced upstream (columns: prompt_idx, prompt_text, pref_*, optimal_w*)
gating_dataset_csv='./results/optimal/optimal_beaver_simple_1305/gating_dataset.csv'
# gating_dataset_csv='./results/optimal/optimal_assistant_simple/gating_dataset.csv'                                                                                       # assistant
# gating_dataset_csv='./results/optimal/optimal_summary_simple/gating_dataset.csv'                                                                                         # summary
# ===========================================================================

gating_type='simple'           # 'simple' = SimpleGatingNetwork | 'per_layer' = GatingNetwork
gating_hidden_size=256
fixed_alpha=1.0                 # 1.0 = softmax (recommended for supervised pretraining)
num_epochs=100
batch_size=64
learning_rate=1e-3
weight_decay=1e-4
max_prompt_len=256
seed=8888
# --------------------
run_name='pretrain_gating_beaver'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29607 \
    ./evolutionary/_pretraining.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --reward_names "${reward_names}" \
    --gating_dataset_csv "${gating_dataset_csv}" \
    --gating_type "${gating_type}" \
    --gating_hidden_size "${gating_hidden_size}" \
    --fixed_alpha "${fixed_alpha}" \
    --num_epochs "${num_epochs}" \
    --batch_size "${batch_size}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --max_prompt_len "${max_prompt_len}" \
    --seed "${seed}" \
    --output_dir "./models/gating_opt/" \
    --run_name "${run_name}" \
    --log_steps 20 \
    2>&1 | tee ./logs/${run_name}.log
