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

norm_rewards='./results/optimal/optimal_beaver_1205/reward_norm.json'    # '' to disable normalisation for the λ-selection table
# norm_rewards=''                                                                                                                                                          # assistant / summary
# ===========================================================================

# Checkpoint(s) to evaluate: a dir with gating_network.pt, or a dir of ind_*/ sub-dirs (e.g. gen_0030/)
gating_paths='./models/ES/es_beaver/final'
eval_prompts=0                 # 0 = all test prompts
batch_size=128
do_sample=false
num_continuations=1
pref_step=0.1
# --------------------
run_name='es_beaver_test'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29603 \
    ./evolutionary/es_test.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_paths ${gating_paths} \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --norm_rewards "${norm_rewards}" \
    --eval_prompts "${eval_prompts}" \
    --batch_size "${batch_size}" \
    --do_sample "${do_sample}" \
    --num_continuations "${num_continuations}" \
    --pref_step "${pref_step}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log
