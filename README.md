# Evolutionary Soups: Evolving Mixture-of-Experts for Multi-Objective LLM Alignment

Evolutionary Soups evolves MoE gating networks via an evolutionary algorithm, enabling a lightweight inference-time lookup that satisfies any user preference for controllable multi-objective LLM alignment without retraining.

## Repository layout

```
scripts/
├── evolutionary/   # Main method: ES over GatingNetwork parameters (+ pretrain/refine/retention)
├── oracle/         # Per-prompt oracle upper bound + supervised gating dataset
└── baselines/      # SFT/PPO, RS, HoE, MOD, MORLHF, RiC
```

All commands below are run from the repository root (`ES/`). Trained experts are expected under `./models/`, evaluation artifacts go to `./results/`, and logs to `./logs/`.

## Tasks and rewards

| Task | Dataset | Reward dimensions |
|------|---------|-------------------|
| assistant | `Anthropic/hh-rlhf` | `harmless, helpful, humor` |
| summary | `openai/summarize_from_feedback` | `summary, faithful, deberta` |
| beaver | `PKU-Alignment/PKU-SafeRLHF-10K` | `beaver_reward, beaver_cost` |

Beaver requires the PKU safe-rlhf package:

```bash
pip install git+https://github.com/PKU-Alignment/safe-rlhf.git
```

## Usage

### Installation

```bash
# Install PyTorch matching your CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch

# Install the remaining Python dependencies
pip install -r requirements.txt
```

### Running

All commands are run from the repository root. Per-objective experts must be trained first; ES consumes their checkpoints.

```bash
# 1. Per-objective experts: one SFT init + one PPO LoRA per reward dimension
bash ./scripts/baselines/fine-tuning/bash/sft.bash
bash ./scripts/baselines/fine-tuning/bash/ppo_xxx.bash      

# 2. Evolutionary Soups: evolve, select per preference, evaluate
bash ./scripts/evolutionary/bash/es_xxx.bash                 
```

---

## 1. Per-objective experts ([scripts/baselines/fine-tuning](scripts/baselines/fine-tuning))

Trains the SFT initialization and one PPO LoRA expert per reward dimension. All downstream methods (ES, RS, HoE, MOD) consume these checkpoints.

| Script | Purpose |
|--------|---------|
| [sft.py](scripts/baselines/fine-tuning/sft.py) | SFT on the task dataset; produces the LoRA adapter used as the init for every expert |
| [ppo.py](scripts/baselines/fine-tuning/ppo.py) | Per-objective PPO from the SFT init; one run per reward name |
| [eval_sft.py](scripts/baselines/fine-tuning/eval_sft.py) | Evaluate the SFT model on the task |
| [eval_ppo_single.py](scripts/baselines/fine-tuning/eval_ppo_single.py) | Evaluate a single PPO expert |
| [eval_ppo_rs.py](scripts/baselines/fine-tuning/eval_ppo_rs.py) | Rewarded soups: linearly merge per-objective LoRAs at the preference vector |
---

## 2. Evolutionary Soups ([scripts/evolutionary](scripts/evolutionary))

Main contribution. A population of GatingNetwork parameter vectors is evolved with a multi-objective ES; at inference a preference λ selects the best individual by `argmax_i λ · fitness_i`.

### Architecture

[es_architecture.py](scripts/evolutionary/es_architecture.py) defines:
- `GatingNetwork` / `MoEForCausalLM` — per-layer gating over hidden states.
- `SimpleGatingNetwork` / `SimpleMoEForCausalLM` — single coefficient vector over experts' final hidden states.

### Main training: [es_train.py](scripts/evolutionary/es_train.py)

ES with Gaussian mutation; parents are re-evaluated on the same chunk as children for a fair comparison. Selection over the merged 2P pool:

- `algorithm=nsgaii` — non-dominated sort + crowding distance
- `algorithm=nsgaiii` — ND sort + Das–Dennis reference points
- `algorithm=greedy_hvc` (default) — ND sort + sequential greedy hypervolume contribution
- `use_greedy_hvc=true` forces greedy HVC regardless of `algorithm`

Parallel evaluation uses a file-based work queue (no NCCL collectives), so any number of GPUs can join via `accelerate launch`. Entry point: [es_train.bash](scripts/evolutionary/bash/es_train.bash).

### Variants and ablations

| Script | Bash | Role |
|--------|------|------|
| [_pretraining.py](scripts/evolutionary/_pretraining.py) | [pretraining.bash](scripts/evolutionary/bash/pretraining.bash) | Supervised pretraining of one GatingNetwork per λ using the oracle `gating_dataset.csv` (MSE between gating output and optimal w). Output is loadable via `--warm_start_path` |
| [_retention.py](scripts/evolutionary/_retention.py) | [retention.bash](scripts/evolutionary/bash/retention.bash) | ES with dual-front retention: combines current-generation Pareto front with a baseline-stability-boosted front to favour parents whose averaged fitness persists across re-evaluations |
| [_refinement.py](scripts/evolutionary/_refinement.py) | [refinement.bash](scripts/evolutionary/bash/refinement.bash) | Beaver-only PPO refinement of each evolved individual using its optimal linear-utility preference μ* (gating params updated, experts frozen) |
| [_dummy.py](scripts/evolutionary/_dummy.py) | [dummy.bash](scripts/evolutionary/bash/dummy.bash) | Save a fixed-seed random initial population without evolving — reproducible random-init baseline |
| [_analyze_variance.py](scripts/evolutionary/_analyze_variance.py) | [analyze_variance.bash](scripts/evolutionary/bash/analyze_variance.bash) | Diagnostic: hidden-state and gating-coefficient anisotropy across prompts |

### Selection and evaluation

| Script | Bash | Role |
|--------|------|------|
| [es_select.py](scripts/evolutionary/es_select.py) | [es_select.bash](scripts/evolutionary/bash/es_select.bash) | For each preference λ, pick the best individual by linear or Tchebyshev utility (z\* from `es_meta.json`); preferences that select the same individual share one test evaluation |
| [es_test.py](scripts/evolutionary/es_test.py) | [es_test.bash](scripts/evolutionary/bash/es_test.bash) | Evaluate every gating checkpoint on the test (or train) loader and print a λ-selection table |

[es_utils.py] (scripts/evolutionary/es_utils.py) holds the shared evaluation, fitness normalization, and checkpoint I/O.

---
## 3. Oracle pipeline ([scripts/oracle](scripts/oracle))

Computes the prompt-adaptive oracle upper bound on the simplex and produces the supervised gating dataset consumed by ES pretraining.

| Step | Script | Output |
|------|--------|--------|
| 1a | [collect_rewards.py](scripts/oracle/collect_rewards.py) | Sweep every simplex gating coefficient by **pre-merging** LoRA adapters; record reward vectors per (prompt, w) |
| 1b | [collect_rewards_simple_gating.py](scripts/oracle/collect_rewards_simple_gating.py) | Same sweep but merges **final hidden states** at inference time (matches `SimpleMoEForCausalLM`) |
| 2 | [build_dataset.py](scripts/oracle/build_dataset.py) | For each (prompt, preference λ) pick argmax of λ·r over the recorded w; emit `gating_dataset.csv` |
| 3 | [eval_oracle.py](scripts/oracle/eval_oracle.py) | Report oracle vs. naive-soup (w = λ) utility gap |

---

## 4. Baselines ([scripts/baselines](scripts/baselines))

| Baseline | Directory | Idea |
|----------|-----------|------|
| RS | [fine-tuning/eval_ppo_rs.py](scripts/baselines/fine-tuning/eval_ppo_rs.py) | Linear LoRA merging of the PPO experts at the preference vector using Rewarded Soups |
| HoE | [hoe/](scripts/baselines/hoe) | Hierarchical Mixture-of-Experts (HoE) with a preference-conditioned router trained by PPO under a blended preference reward (classes import from https://github.com/lizhuolz/HoE) |
| MOD | [mod/](scripts/baselines/mod) | Multi-Objective Decoding (MOD), a decoding-time algorithm that outputs the next token from a linear combination of predictions of all base models |
| MORLHF | [morlhf/](scripts/baselines/morlhf) | Preference-weighted PPO — one model per simplex point. [README](scripts/baselines/morlhf/README.md) |
| RiC | [ric/](scripts/baselines/ric) | Reward-Conditioned SFT alternating offline SFT and online self-improvement. [README](scripts/baselines/ric/README.md) |

Shared baseline helpers (multi-reward scoring, dataset builders, beaver `Instructions`) live in [scripts/baselines/utils](scripts/baselines/utils).
---