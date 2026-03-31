# MOMoE: Multi-Objective Mixture of Experts

MOMoE is a reward-guided expert merging system that learns to dynamically interpolate multiple fine-tuned language models based on user-specified preference vectors. Given a preference over objectives (e.g., harmlessness vs. helpfulness), a learned **GatingNetwork** predicts optimal merging weights that maximize the corresponding linear utility, enabling inference-time control over model behavior without retraining.

## Overview

The core idea is to treat multiple LoRA-adapted expert LLMs as a continuous mixture: rather than selecting a single expert, we interpolate their weights on the simplex and learn a gating function that maps (prompt, preference) → optimal interpolation weights.

The pipeline has four stages:

1. **Collect Rewards** — Sample weight combinations uniformly on the simplex. For each (prompt, weight) pair, generate responses and score them with multiple reward models. Rewards are averaged over K stochastic continuations for lower-variance labels.

2. **Build Gating Dataset** — For each (prompt, preference) pair, find the weight combination with the highest linear utility (`utility = preference · reward`). This becomes the supervised training target.

3. **Train GatingNetwork** — A lightweight FiLM-conditioned neural network maps (prompt embedding, preference) → merging weights. Training supports three loss modes: weight-space MSE, reward-space MSE, and Chebyshev scalarization.

4. **Evaluate** — Compare GatingNetwork predictions against two baselines: naive (preference used directly as weights) and oracle (true optimal weights from the dataset).

## Architecture

### GatingNetwork

The gating network uses **FiLM (Feature-wise Linear Modulation)** to condition prompt representations on the preference vector:

- `prompt_proj`: 3-layer MLP with LayerNorm encodes the prompt hidden state
- `pref_expand`: projects the log-preference vector to the same dimension
- `film_gen`: generates per-channel scale (γ) and shift (β) from the preference
- Modulation: `h' = (1 + γ) ⊙ h + β`
- Output: softmax over expert weights with a learnable temperature

This allows the preference to selectively amplify or suppress which prompt features are used for routing, rather than simply being concatenated to the representation.

### Merging Modes

- **Uniform**: a single weight vector applied to all layers
- **Blockwise**: independent weight vectors for early, mid, and late transformer layers, enabling layer-specific expert specialization

### Prompt Encoding

Prompt hidden states are extracted from the expert LLMs using attention-mask-weighted mean pooling, then averaged across experts. Optionally, reward model hidden states can be used instead.

## Expert Merging

Two modes are supported:

- **LoRA path** (`--use_lora True`): the base model is loaded once and adapter weights are interpolated in-memory per combination. No disk I/O, no barrier timeout issues in distributed settings.
- **Disk path** (`--use_lora False`): merged models are written to disk by rank 0, then loaded by all ranks. Preserved for compatibility.

## Training

Key training features:

- Prompt-level train/val split for generalization tracking
- Early stopping with best-val checkpoint saving
- Cosine annealing LR schedule with warm restart
- Gradient clipping
- WandB logging for `train/loss` and `val/loss`

## Objectives

Reward models used for evaluation and data collection:

| Name | Model | Task |
|------|-------|------|
| `harmless` | `Ray2333/gpt2-large-harmless-reward_model` | Harmlessness |
| `helpful` | `Ray2333/gpt2-large-helpful-reward_model` | Helpfulness |
| `deberta` | `OpenAssistant/reward-model-deberta-v3-large-v2` | General quality |
| `summary` | `Tristan/gpt2_reward_summarization` | Summarization quality |
| `faithful` | `CogComp/bart-faithful-summary-detector` | Summary faithfulness |
| `humor` | `mohameddhiab/humor-no-humor` | Humor detection |

## File Structure

```
scripts/
  momoe/
    collect_rewards.py    # Step 1: collect reward vectors across weight combinations
    build_dataset.py      # Step 2: find optimal weights per (prompt, preference)
    train_new.py          # Step 3: train GatingNetwork
    eval_new.py           # Step 4: evaluate vs. naive and oracle baselines
    new_architecture.py   # GatingNetwork, GatingDataset, prompt encoding
    new_utils.py          # simplex sampling, LoRA merging, reward maps, model I/O
  utils/
    multi_reward_models.py  # multi-objective reward model inference
    utils.py                # dataset loading, tokenization, preference sampling
```
