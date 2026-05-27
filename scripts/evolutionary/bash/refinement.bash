set -e

# === Setting (active: Llama / Beaver; alternatives commented per line) ======
# NOTE: _refinement.py supports the Beaver task only (M=2 objectives).
base_model_name='meta-llama/Llama-2-7b-hf'
# base_model_name='Qwen/Qwen2-7B'                                                                                                                                          # qwen2

expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
# expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'                                                       # qwen2 beaver

dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
reward_names='beaver_reward,beaver_cost'
# ===========================================================================

# Population to refine (point at a gen_XXXX / final dir containing ind_XXX/ sub-dirs)
population_dir='./models/ES/es_beaver/final'
individual_indices=''          # comma-separated, empty = refine every individual
# --- PPO hyper-parameters ---
num_iterations=500
ppo_epochs=2
mini_batch_size=8
eval_batch_size=16
learning_rate=1e-5
clip_range=0.2
init_kl_coef=0.1
value_coef=0.0
entropy_coef=0.0
max_grad_norm=0.5
target_kl=6.0
# --- Generation ---
max_new_tokens=128
# --- Gating template (must match the ES run that produced the population) ---
fixed_alpha=1.2
gating_type='per_layer'
# --------------------
run_name='es_refined_beaver'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29606 \
    ./scripts/evolutionary/_refinement.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --population_dir "${population_dir}" \
    --individual_indices "${individual_indices}" \
    --num_iterations "${num_iterations}" \
    --ppo_epochs "${ppo_epochs}" \
    --mini_batch_size "${mini_batch_size}" \
    --eval_batch_size "${eval_batch_size}" \
    --learning_rate "${learning_rate}" \
    --clip_range "${clip_range}" \
    --init_kl_coef "${init_kl_coef}" \
    --value_coef "${value_coef}" \
    --entropy_coef "${entropy_coef}" \
    --max_grad_norm "${max_grad_norm}" \
    --target_kl "${target_kl}" \
    --max_new_tokens "${max_new_tokens}" \
    --fixed_alpha "${fixed_alpha}" \
    --gating_type "${gating_type}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
