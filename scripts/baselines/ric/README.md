# RiC Baseline

Reward-Conditioned SFT (RiC) trains a model to generate text conditioned on target
reward scores embedded in the prompt. It alternates between offline SFT on a
reward-annotated dataset and online self-improvement iterations.

## Original source

All files are adapted from `RiC/ric/`:

| File | Source |
|------|--------|
| `utils.py` | `RiC/ric/utils.py` |
| `multi_reward_models.py` | `RiC/ric/multi_reward_models.py` |
| `prepare_dataset.py` | `RiC/ric/prepare_dataset_with_rewards.py` |
| `main.py` | `RiC/ric/main.py` |
| `training.py` | `RiC/ric/training.py` |
| `generation.py` | `RiC/ric/generation.py` |
| `evaluation.py` | `RiC/ric/evaluation.py` |

---

## Changes vs original

### `utils.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **`load_reward_model`**: beaver branch loads `PKU-Alignment/beaver-7b-v1.0-{reward,cost}` via `safe_rlhf.models.AutoModelForScore` instead of `AutoModelForSequenceClassification` | Different model class required |
| 2 | **`get_rewards`**: `sub_position=-100` branch reads `.end_scores[0]` (beaver models return `end_scores`, not `logits`) | Correct score extraction |
| 3 | **`build_beaver_dataset_with_preference_n`** (new): score-conditioned eval dataset for `PKU-SafeRLHF-10K`; uses `response_{better_response_id}` columns and the same `\n\nHuman:/\n\nAssistant:` template | Beaver evaluation |
| 4 | **`reset_score_in_dataset`, `dataset_from_csv_n`, `balancing_rewards`**: exp_type checks extended from `=='assistant'` to `in ('assistant','beaver')` | Beaver uses same prompt template |

### `multi_reward_models.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **`_encode_beaver`** (new): formats inputs as `'BEGINNING OF CONVERSATION: USER: <q> ASSISTANT: <r>'` | Required by beaver reward models |
| 2 | **`get_reward_model_scores` — beaver branch**: uses `_encode_beaver` for tokenization, calls `get_rewards` with `sub_position=-100` | Correct reward scoring |
| 3 | **`get_reward_model_scores` — cost negation**: `beaver_cost` rewards are negated (`cost → reward = −cost`) | Cost model is inverted |

### `prepare_dataset.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **`build_dataset_beaver`** (new): builds RiC scored training dataset from `PKU-SafeRLHF-10K`; uses `response_{better_response_id}` / `response_{1-id}` as chosen/rejected, formatted as `\n\nHuman:/\n\nAssistant:` | New task |
| 2 | **`add_score_beaver`**: alias for `add_score_assistant` (same template) | Same format |
| 3 | **exp_type='beaver'** handling in main block | New task |
| 4 | **reward_path_tokenizer_dict**: extended with beaver entries | New task |

### `main.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **`exp_type='beaver'`**: accepted as valid choice | New task |
| 2 | **`sft_model_name`** arg (new, optional): SFT LoRA adapter path passed as `peft_name` for the first offline training call so RiC starts from the same SFT init as MORLHF / HoE / NSGA-II | Fair comparison |
| 3 | **reward_path_tokenizer_dict**: extended with beaver entries | New task |

### `training.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **exp_type check**: `=='assistant'` → `in ('assistant','beaver')` for `DataCollatorForCompletionOnlyLM` template selection | Beaver uses same `\n\nAssistant:` template |
| 2 | **Multi-GPU**: `process_id = accelerator.local_process_index` (no second `Accelerator()`) | Cleaner code |

### `generation.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **exp_type check**: `=='assistant'` → `in ('assistant','beaver')` for `max_new_tokens` (128) and `Instructions_n` selection | Same generation settings |
| 2 | **Multi-GPU**: single `Accelerator()` created at function entry; redundant second call removed | Cleaner code |

### `evaluation.py`

| # | Change | Reason |
|---|--------|--------|
| 1 | **`exp_type='beaver'`**: uses `build_beaver_dataset_with_preference_n` and `Instructions_n` | New task |
| 2 | **reward_path_tokenizer_dict**: extended with beaver entries | New task |
| 3 | **score column removal**: changed from hard-coded `['score1','score2']` to dynamic `[c for c in columns if c.startswith('score')]` | Works for 2- and 3-objective tasks |
| 4 | **Multi-GPU**: module-level `Accelerator()` created once; redundant second assignment inside `evaluate_model` removed | Cleaner code |

---

## Pipeline

RiC requires a **pre-scored dataset** before training. Run the three steps in order.

### Full pipeline scripts

Run from `MOMoE/` directory:

| Task | Script |
|------|--------|
| Assistant (harmless, helpful, humor) | `bash scripts/baselines/ric/bash/ric_assistant.bash` |
| Summary (summary, faithful, deberta) | `bash scripts/baselines/ric/bash/ric_summary.bash` |
| Beaver (beaver_reward, beaver_cost) | `bash scripts/baselines/ric/bash/ric_beaver.bash` |

Each script runs all three steps automatically.

### Step 1 — Prepare scored dataset

```bash
CUDA_VISIBLE_DEVICES=0 python ./scripts/baselines/ric/prepare_dataset.py \
    --reward_names 'harmless,helpful,humor' \
    --save_directory './datasets/ric_assistant.hf' \
    --exp_type 'assistant'
```

Runs the full training split through all reward models; saves a HuggingFace dataset
with `score1`, `score2`, … columns and score-conditioned `prompt_with_score_ids`.

### Step 2 — Train

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    ./scripts/baselines/ric/main.py \
    --base_model_name 'meta-llama/Llama-2-7b-hf' \
    --sft_model_name  './models/sft/sft_assistant_2701/model/' \
    --reward_names 'harmless,helpful,humor' \
    --exp_type 'assistant' \
    --train_dataset_path './datasets/ric_assistant.hf' \
    --save_directory './results/ric/' \
    --wandb_name 'ric_assistant_2704'
```

### Step 3 — Evaluate

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    ./scripts/baselines/ric/evaluation.py \
    --base_model_name 'meta-llama/Llama-2-7b-hf' \
    --peft_name './results/ric/ric_assistant_2704/model_iter1' \
    --reward_names 'harmless,helpful,humor' \
    --exp_type 'assistant' \
    --save_directory './results/ric/' \
    --wandb_name 'ric_assistant_eval_2704'
```

Produces one CSV per preference point:
`results/ric/<eval_name>/eval_data_pref<w1>_<w2>[_<w3>].csv`

### Beaver dependency

```bash
pip install git+https://github.com/PKU-Alignment/safe-rlhf.git
```
