# MORLHF Baseline

Multi-Objective RLHF via preference-weighted PPO. A single model is trained with a
fixed linear combination of reward signals. Running multiple preference vectors traces
the Pareto front.

## Original source

| File | Source |
|------|--------|
| `morlhf.py` | `RiC/ppo/morlhf.py` |
| `eval_morlhf.py` | `RiC/ppo/eval_ppo_single_model.py` |

---

## Changes vs original

### `morlhf.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **Imports**: `sys.path` → `Evolutionary-Soups/baselines/utils`; uses `baselines.utils.{utils, multi_reward_models}` | Centralised utils with beaver support |
| 2 | **Dataset builders**: `build_dataset` / `build_dataset_summary` → `build_dataset_ppo` / `build_dataset_summary_ppo` | ES equivalents with optional rm_tokenizer |
| 3 | **Beaver support**: `exp_type='beaver'` added; uses `build_dataset_beaver_ppo` and `Instructions()` (same Human/Assistant format) | New task |
| 4 | **Reward dict**: extended with `beaver_reward` / `beaver_cost` | New task |
| 5 | **`--preference`**: changed from single `float` to comma-separated `str` (e.g. `'0.2,0.4,0.4'`) covering all N reward dimensions; the original equal-weight override for N==3 removed | Multi-objective simplex sampling from bash |
| 6 | **Model loading**: split into `base_model_name` (bare LLaMA) + `sft_model_name` (SFT LoRA adapter). Loads base, applies SFT LoRA via `PeftModel.from_pretrained(..., is_trainable=True)`. The original expected a **merged** model and added a fresh LoRA on top — that fails since `sft.py` saves an adapter only | Correct loading of project SFT models |
| 7 | **PPO hyper-parameters**: `mini_batch_size` and `gradient_accumulation_steps` default changed from **1 → 8**. The original defaults caused 64× more optimizer steps per rollout (~3–10× wall-clock overhead vs `ppo.py`) | Performance parity with `fine-tuning/ppo.py` |
| 8 | **Best-model tracking**: rolling window of 10 batches; saves `best_model/` whenever a new high is reached — identical to `fine-tuning/ppo.py` | Reproducible best checkpoint |
| 9 | **Checkpoint naming**: `step_N` (every 100 steps), `epoch_N_final` (end of epoch), `final` (end of training). Original used `batch_N` which was ambiguous | Consistent with `ppo.py` |
| 10 | **Multi-GPU**: single `Accelerator()` instance created once; `process_id = accelerator.local_process_index` (no redundant `Accelerator()` calls); dead `current_device` variable removed | Cleaner code |

### `eval_morlhf.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **Imports**: same sys.path + ES utils | Centralised utils |
| 2 | **Beaver support**: `exp_type='beaver'` uses `build_dataset_beaver_eval` and `Instructions()` | New task |
| 3 | **Reward dict**: extended with beaver entries | New task |
| 4 | **Model loading**: `base_model_name` (bare LLaMA) + `checkpoint_path` (trained LoRA adapter from `ppo_trainer.save_pretrained`). Loads base, applies LoRA via `PeftModel.from_pretrained`, then `merge_and_unload()` | Correct loading of LoRA checkpoints |
| 5 | **Multi-GPU**: single `Accelerator()` instance at top; no redundant instantiations | Cleaner code |

---

## Usage

All commands run from `Evolutionary-Soups/` directory.

### Train

```bash
# Assistant (harmless, helpful, humor — 21 simplex points, step 0.2)
bash baselines/morlhf/bash/morlhf_assistant.bash

# Summary (summary, faithful, deberta — 21 simplex points, step 0.2)
bash baselines/morlhf/bash/morlhf_summary.bash

# Beaver (beaver_reward, beaver_cost — 11 simplex points, step 0.1)
bash baselines/morlhf/bash/morlhf_beaver.bash
```

Each bash script loops over all preference points; each preference trains one model.
`best_model/` is saved automatically during training.

### Evaluate a single checkpoint

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    ./baselines/morlhf/eval_morlhf.py \
    --base_model_name 'meta-llama/Llama-2-7b-hf' \
    --checkpoint_path './results/morlhf/morlhf_assistant_2704_pref0.33_0.33_0.33/best_model' \
    --reward_names 'harmless,helpful,humor' \
    --exp_type 'assistant' \
    --wandb_name 'eval_morlhf_assistant_pref0.33_0.33_0.33'
```

Results saved to `results/morlhf/<wandb_name>/eval_data.csv`.

### PPO hyper-parameters per task

| Task | epochs | mini_batch_size | grad_accum | target | init_kl_coef |
|------|--------|-----------------|------------|--------|--------------|
| assistant | 1 | 8 | 8 | 3 | 0.2 |
| summary | 3 | 8 | 8 | 6 | 0.05 |
| beaver | 5 | 8 | 4 | 6 | 0.1 |

### Beaver dependency

```bash
pip install git+https://github.com/PKU-Alignment/safe-rlhf.git
```
