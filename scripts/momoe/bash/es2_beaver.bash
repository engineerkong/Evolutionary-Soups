base_model_name='meta-llama/Llama-2-7b-hf'
expert_model_paths='./models/ppo/ppo_beaver_reward_2204/best_model ./models/ppo/ppo_beaver_cost_2204/best_model'
dataset_name='PKU-Alignment/PKU-SafeRLHF-10K'
gating_hidden_size=256
# --------------------
run_name='es2_beaver_gating_sft'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29602 \
    ./scripts/momoe/evolutionary_soups2.py \
    --base_model_name "${base_model_name}" \
    --expert_model_paths ${expert_model_paths} \
    --dataset_name "${dataset_name}" \
    --gating_hidden_size "${gating_hidden_size}" \
    --wandb_name "${run_name}" \
    --output_dir "./models/gating_sft/${run_name}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --save_strategy epoch \
    --logging_steps 10 \
    --use_dual_front false \
    --algorithm nsgaii \
    --warm_start_path "./models/gating_sft/${run_name}/gating_network" \
    2>&1 | tee ./logs/${run_name}.log
