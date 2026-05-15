base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'

# GatingNetwork (per-layer) checkpoints: dummy random-init population
gating_paths_per_layer='./models/ES/dummy_beaver/final'

# SimpleGatingNetwork checkpoints: random-init dummy population
gating_paths_simple='./models/ES/dummy_beaver_simple/final'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port 29604 \
    ./scripts/momoe/analyze_hidden_state_variance.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --gating_dataset_csv './results/optimal/optimal_beaver_1205/gating_dataset.csv' \
    --gating_paths_per_layer ${gating_paths_per_layer} \
    --gating_paths_simple    ${gating_paths_simple} \
    --max_prompts 200 \
    --batch_size 16 \
    --max_prompt_len 256 \
    --output_json './results/analyze_hidden_variance_beaver.json' \
    2>&1 | tee ./logs/analyze_hidden_variance_beaver.log
