CUDA_VISIBLE_DEVICES=6,7 accelerate launch --main_process_port 29606 \
    ./scripts/momoe/preference_eval.py \
    --expert_model_paths './models/ppo/ppo_beaver_reward_2204/best_model' \
                         './models/ppo/ppo_beaver_cost_2204/best_model' \
    --nsgaii_gating_dir  './models/nsgaii/nsgaii_beaver_0105/gen_0035/meta' \
    --hoe_path           '' \
    --morlhf_dir         '' \
    --morlhf_prefix      'morlhf_beaver_2704_pref' \
    --reward_names       'beaver_reward,beaver_cost' \
    --dataset_name       'PKU-Alignment/PKU-SafeRLHF-10K' \
    --run_name           'preference_eval_0305_ciea' \
    2>&1 | tee ./logs/preference_eval_0305_ciea.log
