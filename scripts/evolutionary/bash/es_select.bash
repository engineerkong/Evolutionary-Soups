set -e

# === Setting (active: Llama / Beaver; alternatives commented per line) ======
base_model_name='meta-llama/Llama-2-7b-hf'
# base_model_name='Qwen/Qwen2-7B'                                                                                                                                          # qwen2

expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
# expert_model_paths='./models/ppo/ppo_qwen2_beaver_reward/best_model ./models/ppo/ppo_qwen2_beaver_cost/best_model'                                                       # qwen2 beaver
# expert_model_paths='./models/ppo/ppo_assistant_harmless_2104/best_model ./models/ppo/ppo_assistant_helpful_2104/best_model ./models/ppo/ppo_assistant_humor_2104/best_model'  # assistant
# expert_model_paths='./models/ppo/ppo_summary_summary_2104/best_model ./models/ppo/ppo_summary_faithful_2104/best_model ./models/ppo/ppo_summary_deberta_2104/best_model'      # summary

dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
# dataset_name='Anthropic/hh-rlhf'                                                                                                                                         # assistant
# dataset_name='openai/summarize_from_feedback'                                                                                                                           # summary

reward_names='beaver_reward,beaver_cost'
# reward_names='harmless,helpful,humor'                                                                                                                                    # assistant
# reward_names='summary,faithful,deberta'                                                                                                                                  # summary
# ===========================================================================

# Population dir containing ind_XXX/ sub-dirs (each with fitness.json + gating_network.pt)
es_gating_dir='./models/ES/es_beaver/final'
es_meta_path=''                # auto-discovered (walks up from es_gating_dir) if blank
utility='tchebyshev'           # 'linear' | 'tchebyshev'
eval_prompts=0                 # 0 = all test prompts
batch_size=128
num_continuations=1
pref_step=0.1
# --------------------
run_name='es_beaver_select'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29604 \
    ./scripts/evolutionary/es_select.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --es_gating_dir "${es_gating_dir}" \
    --es_meta_path "${es_meta_path}" \
    --utility "${utility}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --batch_size "${batch_size}" \
    --num_continuations "${num_continuations}" \
    --pref_step "${pref_step}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
