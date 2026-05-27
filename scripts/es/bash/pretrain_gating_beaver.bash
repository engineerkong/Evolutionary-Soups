set -e

base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
reward_names='beaver_reward,beaver_cost'
gating_dataset_csv='./results/optimal/optimal_beaver_simple_1305/gating_dataset.csv'
pretrain_run_name='pretrain_gating_beaver_1305'
test_run_name='nsgaii_beaver_pretrain_simple_test_1305'

# ---------------------------------------------------------------------------
# Step 1: Pretrain SimpleGatingNetwork for each preference point
# ---------------------------------------------------------------------------
echo "========================================"
echo "Step 1: Pretraining gating networks"
echo "========================================"

rm -rf "./models/gating_opt/${pretrain_run_name}"
rm -rf "./results/nsgaii/${test_run_name}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29602 \
    ./scripts/momoe/pretraining_opt.py \
    --base_model_name      "${base_model_name}" \
    --expert_model_paths   ${expert_model_paths} \
    --reward_names         "${reward_names}" \
    --gating_dataset_csv   "${gating_dataset_csv}" \
    --gating_type          simple \
    --gating_hidden_size   256 \
    --fixed_alpha          1.0 \
    --num_epochs           100 \
    --batch_size           64 \
    --learning_rate        1e-3 \
    --max_prompt_len       256 \
    --seed                 8888 \
    --output_dir           "./models/gating_opt/" \
    --run_name             "${pretrain_run_name}" \
    --log_steps            20 \
    2>&1 | tee ./logs/${pretrain_run_name}.log

# ---------------------------------------------------------------------------
# Step 2: Evaluate pretrained gating networks
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 2: Evaluating pretrained gating networks"
echo "========================================"

gating_paths=$(echo ./models/gating_opt/${pretrain_run_name}/pref_*)
norm_rewards='./results/optimal/optimal_beaver_simple_1305/reward_norm.json'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29602 \
    ./scripts/momoe/nsgaii_test.py \
    --base_model_name      "${base_model_name}" \
    --expert_model_paths   ${expert_model_paths} \
    --gating_paths         ${gating_paths} \
    --dataset_name         'PKU-Alignment/PKU-SafeRLHF-10K' \
    --reward_names         "${reward_names}" \
    --norm_rewards         "${norm_rewards}" \
    --eval_prompts         0 \
    --pref_step            0.1 \
    --run_name             "${test_run_name}" \
    2>&1 | tee ./logs/${test_run_name}.log
