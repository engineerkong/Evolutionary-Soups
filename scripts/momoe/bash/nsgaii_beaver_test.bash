set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
gating_paths='./models/ES/es_beaver_1105/gen_0055/'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
reward_names='beaver_reward,beaver_cost'
eval_prompts=0
pref_step=0.1
run_name='nsgaii_beaver_test_1105'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29601 \
    ./scripts/momoe/nsgaii_test.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_paths "${gating_paths}" \
    --dataset_name "${dataset_name}" \
    --reward_names "${reward_names}" \
    --eval_prompts "${eval_prompts}" \
    --pref_step "${pref_step}" \
    --run_name "${run_name}" \
    2>&1 | tee ./logs/${run_name}.log