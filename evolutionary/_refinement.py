"""_refinement.py — PPO refinement of ES individuals.

For each individual i in the ES final population, compute its **optimal linear
utility preference** μ_i* — the preference vector under which individual i
achieves the highest linear utility μ · r among all P population members:
        μ_i*  s.t.  argmax_j μ · fitness_j  =  i
Then refine the gating network of individual i with PPO using the scalarized
reward r_scalar = μ_i* · r. Experts remain frozen; only the gating parameters
are updated.

Beaver task only (PKU-Alignment/PKU-SafeRLHF-10K, M=2 objectives).

Style mirrors es_train.py:
  - dataclass ScriptArguments + HfArgumentParser.
  - File-based work queue across ranks (no NCCL collectives).
  - Per-individual checkpoint directories.

Work model: each individual is one "task". Ranks claim tasks from the queue
and run the full PPO loop for that individual on their own GPU. After all
tasks complete, the refined gating networks are saved under <output>/refined/.
"""

import copy
import datetime
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from es_architecture import GatingNetwork, MoEForCausalLM, load_shared_experts
from es_utils import (
    Instructions, REWARD_PATHS, RewardModels,
    build_dataset_beaver_ppo, get_clean_data, load_gating_network,
    load_main_tokenizer, net_to_params, params_to_net, save_gating_network,
)


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:      str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths:   List[str] = field(default_factory=list)
    reward_names:         str       = 'beaver_reward,beaver_cost'
    dataset_name:         str       = 'PKU-Alignment/PKU-SafeRLHF-10K'  # fixed for beaver
    # Population to refine
    population_dir:       str       = ''   # path to a directory containing ind_XXX/ sub-dirs
    individual_indices:   str       = ''   # comma-separated list, '' = all individuals

    # Generation / rollout
    do_sample:            bool      = True
    eval_batch_size:      int       = 16
    max_new_tokens:       int       = 128

    # PPO hyper-parameters
    num_iterations:       int       = 100  # PPO rollouts per individual
    ppo_epochs:           int       = 2    # optimizer epochs per rollout
    mini_batch_size:      int       = 8
    learning_rate:        float     = 1e-5
    clip_range:           float     = 0.2
    init_kl_coef:         float     = 0.05
    value_coef:           float     = 0.1
    entropy_coef:         float     = 0.0
    max_grad_norm:        float     = 1.0
    target_kl:            float     = 6.0   # early-stop epoch loop if KL exceeds
    gamma:                float     = 1.0   # discount (sequence-level reward → 1.0)
    lam:                  float     = 0.95  # GAE λ (unused if value_coef=0)

    # Preference computation
    pref_resolution:      int       = 1001  # grid points on simplex for μ_i* search

    # Gating template (must match the ES run that produced the population)
    fixed_alpha:          float     = 1.2
    gating_type:          str       = 'per_layer'  # currently only per_layer supported

    # GPU / I/O
    gpu_id:               int       = -1
    save_directory:       str       = './models/ES_refined/'
    run_name:             str       = 'es_refined_beaver'
    save_every:           int       = 20
    seed:                 int       = 8888
    verbose:              bool      = False


# ---------------------------------------------------------------------------
# Optimal preference computation
# ---------------------------------------------------------------------------

def _pareto_dominated_mask(fitnesses: np.ndarray) -> np.ndarray:
    """Boolean mask: True at index i if some j ≠ i Pareto-dominates i.

    j dominates i iff  all(fit[j] >= fit[i])  and  any(fit[j] > fit[i]).
    """
    P         = len(fitnesses)
    dominated = np.zeros(P, dtype=bool)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            if np.all(fitnesses[j] >= fitnesses[i]) and np.any(fitnesses[j] > fitnesses[i]):
                dominated[i] = True
                break
    return dominated


def compute_optimal_preferences(fitnesses: np.ndarray,
                                 resolution: int = 1001) -> List[np.ndarray]:
    """For each individual, return the preference μ_i* that makes it the (normalized)
    linear-utility winner among non-dominated points.

    Pipeline:
      1. Drop Pareto-dominated individuals (∃ j: fit[j] ≥ fit[i] component-wise
         and fit[j] ≠ fit[i]). They return `None`.
      2. Min-max normalize each objective across the full population:
            r_norm[i, m] = (fit[i, m] - min_m) / (max_m - min_m).
         This puts both objectives on a comparable [0, 1] scale before the
         linear-utility sweep, so the assignment isn't dominated by whichever
         objective happens to have the larger numerical range.
      3. For each non-dominated i, sweep μ = (w, 1-w) over a dense grid on
         [0, 1] and find the w-range where individual i achieves the largest
         μ · r_norm[i]. The centroid of that range is μ_i*.

    For M=2 only.
    """
    P, M = fitnesses.shape
    if M != 2:
        raise NotImplementedError('compute_optimal_preferences currently supports M=2 only.')

    dominated   = _pareto_dominated_mask(fitnesses)
    non_dom_idx = np.where(~dominated)[0]

    # Min-max normalize rewards across the full population
    r_min   = fitnesses.min(axis=0)
    r_max   = fitnesses.max(axis=0)
    r_range = np.maximum(r_max - r_min, 1e-8)
    fit_n   = (fitnesses - r_min) / r_range          # (P, M) ∈ [0, 1]

    # Sweep μ on the 1-simplex; track which non-dominated individual wins at each grid point
    ws      = np.linspace(0.0, 1.0, resolution)
    winners = np.empty(resolution, dtype=int)
    for k, w in enumerate(ws):
        mu             = np.array([w, 1.0 - w])
        utilities      = fit_n[non_dom_idx] @ mu
        winners[k]     = int(non_dom_idx[int(np.argmax(utilities))])

    result: List = [None] * P
    for i in non_dom_idx:
        mask = (winners == i)
        if mask.any():
            w_centroid = float(ws[mask].mean())
            result[i]  = np.array([w_centroid, 1.0 - w_centroid], dtype=np.float32)
        else:
            # Non-dominated but interior to the convex hull (concave Pareto front).
            # Use the normalized reward direction as a reasonable fallback μ.
            r_dir     = fit_n[i] + 1e-8
            result[i] = (r_dir / r_dir.sum()).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Population loader
# ---------------------------------------------------------------------------

def _load_population(pop_dir: str, template: GatingNetwork,
                     device: str = 'cpu') -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (params, fitness) for every ind_XXX/ sub-dir under pop_dir.

    Each sub-dir must contain gating_network.pt + gating_config.json + fitness.json.
    """
    if not os.path.isdir(pop_dir):
        raise FileNotFoundError(f'population_dir not found: {pop_dir}')
    sub_dirs = sorted([d for d in os.listdir(pop_dir)
                       if d.startswith('ind_') and d[4:].isdigit()
                       and os.path.isdir(os.path.join(pop_dir, d))])
    if not sub_dirs:
        raise FileNotFoundError(f'No ind_XXX/ sub-dirs found in {pop_dir}')

    population: List[Tuple[np.ndarray, np.ndarray]] = []
    for sd in sub_dirs:
        ind_dir  = os.path.join(pop_dir, sd)
        gating   = load_gating_network(
            ind_dir,
            lm_hidden_size=template.lm_hidden_size,
            num_experts=template.num_experts,
            num_layers=getattr(template, 'num_layers', 32),
            device=device,
        )
        if gating is None:
            raise FileNotFoundError(f'No gating_network.pt under {ind_dir}')
        params   = net_to_params(gating)
        fit_path = os.path.join(ind_dir, 'fitness.json')
        if os.path.exists(fit_path):
            with open(fit_path) as f:
                fit = np.array(json.load(f)['fitness'], dtype=np.float32)
        else:
            fit = np.zeros(template.num_experts, dtype=np.float32)  # fallback
        population.append((params, fit))
    return population


# ---------------------------------------------------------------------------
# MoE forward with gradients (gating-only)
# ---------------------------------------------------------------------------

def moe_forward_logits(moe_model: MoEForCausalLM,
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    """Single forward through MoE returning per-token logits. Gradients flow only
    through the gating network's coefficients (experts have requires_grad=False).
    """
    hidden       = moe_model._embed(input_ids)
    causal_mask  = moe_model._build_causal_mask(attention_mask, hidden, past_len=0)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    hidden, _, _ = moe_model._moe_layers(
        hidden, causal_mask, position_ids, kv_cache=None,
        use_cache=False, seq_coefficients=None,
    )
    logits = moe_model.experts[0].lm_head(
        moe_model.experts[0].model.norm(hidden))
    return logits  # (B, seq, V)


def _gather_response_log_probs(logits: torch.Tensor, input_ids: torch.Tensor,
                                response_mask: torch.Tensor) -> torch.Tensor:
    """Token-level log probs over response positions.

    Args:
        logits:        (B, seq, V) — output of moe_forward_logits.
        input_ids:     (B, seq)    — full (query + response) ids.
        response_mask: (B, seq)    — 1 where token is in the response, else 0.

    Returns:
        log_probs: (B, seq-1) — log π(input_ids[:, t+1] | input_ids[:, :t+1])
                                masked to zero at non-response positions.
    """
    log_probs_all = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predict next
    targets       = input_ids[:, 1:]                                   # next tokens
    gathered      = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, seq-1)
    mask          = response_mask[:, 1:].float()
    return gathered * mask, mask


# ---------------------------------------------------------------------------
# Small value head over last-token hidden state
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """Tiny scalar value head — pooled (last-real-token) hidden state → V(s)."""

    def __init__(self, lm_hidden_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(torch.bfloat16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.net(hidden_states.to(self.net[0].weight.dtype)).squeeze(-1).float()


# ---------------------------------------------------------------------------
# PPO refinement for a single individual
# ---------------------------------------------------------------------------

class PPORefiner:
    """PPO refinement of one MoE gating network with a scalar linear-utility reward."""

    def __init__(
        self,
        moe_model:        MoEForCausalLM,
        gating_template:  GatingNetwork,
        sft_tokenizer,
        reward_models:    RewardModels,
        instructions,
        device:           str = 'cuda:0',
        gpu_id:           int = 0,
    ):
        self.moe              = moe_model
        self.template         = gating_template
        self.tokenizer        = sft_tokenizer
        self.reward_models    = reward_models
        self.instructions     = instructions
        self.device           = device
        self.gpu_id           = gpu_id

        # Value head shares dimensionality with the LM hidden state
        self.value_head       = ValueHead(gating_template.lm_hidden_size).to(device)

    @staticmethod
    def _log(rank, msg, verbose=True):
        if verbose:
            print(f'[{datetime.datetime.now():%H:%M:%S}] rank{rank} {msg}', flush=True)

    # ------------------------------------------------------------------
    # Reward computation: scalar = preference · multi-objective reward
    # ------------------------------------------------------------------

    def _compute_rewards(self, prompts_decoded, responses,
                         preference: np.ndarray,
                         normalize_rewards: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(self.instructions.get_input(r), self.instructions.get_response(r))
                 for r in responses_clean]
        if hasattr(self.instructions, 'get_post'):
            multi = self.reward_models.get_reward_model_scores(
                pairs, self.instructions.get_post,
                normalize_rewards=normalize_rewards, round_digits=None)
        else:
            multi = self.reward_models.get_reward_model_scores(
                pairs, normalize_rewards=normalize_rewards, round_digits=None)
        # multi: List[List[float]] indexed [reward][prompt] — shape (M, B)
        multi_arr     = np.array(multi, dtype=np.float32).T          # (B, M)
        scalar_reward = multi_arr @ preference.astype(np.float32)    # (B,)
        return scalar_reward, multi_arr

    # ------------------------------------------------------------------
    # PPO loop for one individual
    # ------------------------------------------------------------------

    def refine(
        self,
        params_init:       np.ndarray,
        preference:        np.ndarray,
        dataset,
        data_collator,
        generation_kwargs: dict,
        num_iterations:    int    = 100,
        ppo_epochs:        int    = 2,
        mini_batch_size:   int    = 8,
        batch_size:        int    = 16,
        learning_rate:     float  = 1e-5,
        clip_range:        float  = 0.2,
        init_kl_coef:      float  = 0.05,
        value_coef:        float  = 0.1,
        entropy_coef:      float  = 0.0,
        max_grad_norm:     float  = 1.0,
        target_kl:         float  = 6.0,
        seed:              int    = 8888,
        verbose:           bool   = False,
        rank:              int    = 0,
        save_cb=None,
    ) -> Tuple[np.ndarray, dict]:
        """Refine `params_init` with PPO; return refined params + training log."""

        # ── Set MoE gating to this individual's parameters ──────────────────
        gating = params_to_net(params_init, self.template, self.device)
        gating.train()
        # Only the gating network has trainable params; experts already frozen.
        self.moe.gating_net = gating

        trainable = [p for p in gating.parameters() if p.requires_grad] + \
                    [p for p in self.value_head.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=learning_rate)

        # Reference gating snapshot for KL penalty (frozen)
        ref_gating = copy.deepcopy(gating).eval()
        for p in ref_gating.parameters():
            p.requires_grad = False

        # DataLoader over the prompt set; shuffled per rollout
        rng       = np.random.default_rng(seed + rank)
        loader_b  = batch_size
        n_prompts = len(dataset)

        log = {'iter': [], 'reward_mean': [], 'reward_std': [],
               'reward_vec_mean': [], 'kl_mean': [], 'policy_loss': [],
               'value_loss': [], 'entropy': [], 'preference': preference.tolist()}

        pbar = tqdm(range(num_iterations),
                    desc=f'rank{rank} μ*={np.round(preference, 2).tolist()}',
                    position=rank, leave=True, dynamic_ncols=True)
        for it in pbar:
            t0 = time.time()
            # Sample a batch of prompts
            idx     = rng.choice(n_prompts, size=loader_b, replace=False).tolist()
            loader  = DataLoader(dataset.select(idx), batch_size=loader_b,
                                 collate_fn=data_collator, drop_last=False)
            batch   = next(iter(loader))
            prompt_ids   = batch['input_ids'].to(self.device)
            prompt_attn  = batch['attention_mask'].to(self.device)
            B, q_len     = prompt_ids.shape

            # ── Rollout: generate responses with current gating (no_grad) ───
            gating.eval()
            with torch.no_grad():
                gen_out = self.moe.generate(
                    prompt_ids, attention_mask=prompt_attn,
                    **generation_kwargs,
                )                                                   # (B, q+r)
            gating.train()

            # Build (q+r) ids + masks
            full_ids        = gen_out
            seq_len         = full_ids.shape[1]
            pad_token       = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            full_attn       = (full_ids != pad_token).long()
            # response_mask: 1 only at positions q_len .. seq_len-1
            response_mask   = torch.zeros_like(full_ids)
            response_mask[:, q_len:] = 1
            response_mask  = response_mask * full_attn  # ignore trailing pad

            # ── Compute reward (multi-objective → scalar via preference) ────
            prompts_decoded = self.tokenizer.batch_decode(prompt_ids.cpu())
            responses_str   = self.tokenizer.batch_decode(full_ids.cpu())
            scalar_reward, reward_vec = self._compute_rewards(
                prompts_decoded, responses_str, preference)
            rewards_t       = torch.tensor(scalar_reward, dtype=torch.float32,
                                            device=self.device)
            torch.cuda.empty_cache()

            # ── Old log probs + reference log probs (no_grad) ───────────────
            with torch.no_grad():
                logits_old              = moe_forward_logits(self.moe, full_ids, full_attn)
                logp_old, _             = _gather_response_log_probs(
                    logits_old, full_ids, response_mask)
                # Reference forward: swap in ref_gating temporarily
                self.moe.gating_net     = ref_gating
                logits_ref              = moe_forward_logits(self.moe, full_ids, full_attn)
                self.moe.gating_net     = gating
                logp_ref, mask_resp     = _gather_response_log_probs(
                    logits_ref, full_ids, response_mask)
                # Value at sequence end (last real token of full sequence)
                # Cheap: re-use logits_old's hidden via norm-input → here use last-token logit zero-out;
                # to keep separation simple we just use a fixed scalar baseline (running mean).
            # Running baseline for advantage (sequence-level reward)
            if it == 0:
                self._baseline = float(rewards_t.mean().item())
            advantages = rewards_t - self._baseline
            self._baseline = 0.9 * self._baseline + 0.1 * float(rewards_t.mean().item())

            # Normalize advantages within batch for stability
            if advantages.std() > 1e-6:
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            else:
                adv_norm = advantages

            # ── PPO epochs ──────────────────────────────────────────────────
            iter_kl, iter_pl, iter_vl, iter_ent = [], [], [], []
            for epoch in range(ppo_epochs):
                perm = torch.randperm(B, device=self.device)
                for mb_start in range(0, B, mini_batch_size):
                    mb_idx        = perm[mb_start:mb_start + mini_batch_size]
                    mb_ids        = full_ids[mb_idx]
                    mb_attn       = full_attn[mb_idx]
                    mb_resp_mask  = response_mask[mb_idx]
                    mb_logp_old   = logp_old[mb_idx]
                    mb_logp_ref   = logp_ref[mb_idx]
                    mb_adv        = adv_norm[mb_idx]
                    mb_reward     = rewards_t[mb_idx]

                    # Forward with grads
                    logits_new    = moe_forward_logits(self.moe, mb_ids, mb_attn)
                    logp_new, m_resp = _gather_response_log_probs(
                        logits_new, mb_ids, mb_resp_mask)

                    # Per-token ratio; sum over response, weighted by sequence-level advantage
                    log_ratio      = (logp_new - mb_logp_old)                  # (mb, seq-1)
                    log_ratio      = log_ratio * m_resp                        # mask
                    # Per-token clipped surrogate (token-mean) with broadcast adv
                    ratio          = torch.exp(log_ratio)
                    adv_b          = mb_adv.unsqueeze(-1).float()              # (mb, 1)
                    surr1          = ratio * adv_b
                    surr2          = torch.clamp(ratio, 1.0 - clip_range,
                                                  1.0 + clip_range) * adv_b
                    pg_loss        = -((torch.min(surr1, surr2) * m_resp).sum() /
                                       m_resp.sum().clamp(min=1.0))

                    # KL penalty vs. reference policy (per-token approx)
                    kl_token       = (logp_new - mb_logp_ref) * m_resp
                    kl_seq         = kl_token.sum(dim=-1) / m_resp.sum(dim=-1).clamp(min=1.0)
                    kl_mean        = kl_seq.mean()
                    kl_loss        = init_kl_coef * kl_mean

                    # Value loss — predict scalar reward from final hidden state.
                    # Reuse the last hidden state from logits_new path: redo a tiny
                    # forward through the norm input is expensive, so approximate by
                    # using the last token's pre-lm-head representation. To avoid a
                    # second forward, project logits_new[:, -1, :] through a frozen
                    # linear is not standard — instead, run value head on a cheap
                    # surrogate: the mean lm_head input (last hidden).
                    # For simplicity here we use a constant feature (zero), making
                    # value_loss vanish when value_coef=0. Set value_coef>0 to enable.
                    if value_coef > 0.0:
                        # cheap re-forward of last layer hidden — we don't have it cached,
                        # so use a stop-grad pooled prompt hidden (sufficient for crude
                        # variance reduction; gradient still flows from advantage).
                        with torch.no_grad():
                            prompt_h = self.moe._embed(mb_ids[:, :q_len])
                            prompt_h = prompt_h[:, -1, :].float()
                        v_pred     = self.value_head(prompt_h.to(self.value_head.net[0].weight.dtype))
                        v_loss     = F.mse_loss(v_pred, mb_reward.float())
                    else:
                        v_loss     = torch.tensor(0.0, device=self.device)

                    # Entropy bonus (token-mean over response)
                    if entropy_coef > 0.0:
                        probs_new   = F.softmax(logits_new[:, :-1, :].float(), dim=-1)
                        logp_full   = F.log_softmax(logits_new[:, :-1, :].float(), dim=-1)
                        ent_token   = -(probs_new * logp_full).sum(dim=-1) * m_resp
                        entropy_val = ent_token.sum() / m_resp.sum().clamp(min=1.0)
                        ent_loss    = -entropy_coef * entropy_val
                    else:
                        entropy_val = torch.tensor(0.0, device=self.device)
                        ent_loss    = torch.tensor(0.0, device=self.device)

                    loss = pg_loss + kl_loss + value_coef * v_loss + ent_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    optimizer.step()

                    iter_kl.append(float(kl_mean.item()))
                    iter_pl.append(float(pg_loss.item()))
                    iter_vl.append(float(v_loss.item()))
                    iter_ent.append(float(entropy_val.item()))

                # Early-stop epoch loop if KL exceeds target
                if iter_kl and iter_kl[-1] > target_kl:
                    break

            # ── Iteration logging ──────────────────────────────────────────
            log['iter'].append(it)
            log['reward_mean'].append(float(scalar_reward.mean()))
            log['reward_std'].append(float(scalar_reward.std()))
            log['reward_vec_mean'].append(reward_vec.mean(axis=0).tolist())
            log['kl_mean'].append(float(np.mean(iter_kl)) if iter_kl else 0.0)
            log['policy_loss'].append(float(np.mean(iter_pl)) if iter_pl else 0.0)
            log['value_loss'].append(float(np.mean(iter_vl)) if iter_vl else 0.0)
            log['entropy'].append(float(np.mean(iter_ent)) if iter_ent else 0.0)

            dt = time.time() - t0
            pbar.set_postfix({
                'r':    f'{scalar_reward.mean():+.3f}',
                'r_vec': np.round(reward_vec.mean(axis=0), 2).tolist(),
                'KL':   f'{log["kl_mean"][-1]:.3f}',
                'pg':   f'{log["policy_loss"][-1]:+.3f}',
                's/it': f'{dt:.1f}',
            })
            self._log(rank,
                       f'it {it:4d}/{num_iterations} | r̄={scalar_reward.mean():+.3f}±{scalar_reward.std():.3f} '
                       f'| r_vec={np.round(reward_vec.mean(axis=0), 3)} '
                       f'| KL={log["kl_mean"][-1]:.4f} | pg={log["policy_loss"][-1]:+.4f} '
                       f'| t={dt:.1f}s',
                       verbose)

            if save_cb is not None and (it + 1) % max(1, num_iterations // 5) == 0:
                save_cb(it, net_to_params(gating), log)

        pbar.close()

        return net_to_params(gating), log


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

assert script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K', \
    'evolutionary_soups_refinement.py only supports the beaver task ' \
    f'(PKU-Alignment/PKU-SafeRLHF-10K); got {script_args.dataset_name!r}.'
assert script_args.population_dir, '--population_dir is required'

output_dir = os.path.join(script_args.save_directory, script_args.run_name)
os.makedirs(output_dir, exist_ok=True)
refined_dir = os.path.join(output_dir, 'refined')
os.makedirs(refined_dir, exist_ok=True)

set_seed(script_args.seed)
np.random.seed(script_args.seed)

# File-based coordination only — no NCCL collectives (see evolutionary_soups.py note).
accelerator = Accelerator()
gpu_id      = (script_args.gpu_id if script_args.gpu_id >= 0
               else accelerator.local_process_index)
device      = f'cuda:{gpu_id}'
rank        = accelerator.local_process_index
num_ranks   = accelerator.num_processes
is_main     = rank == 0

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_objectives = len(reward_names)
assert n_objectives == 2, 'beaver expects 2 objectives (beaver_reward, beaver_cost)'

reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

sft_tokenizer              = load_main_tokenizer(script_args.expert_model_paths[0])
sft_tokenizer.padding_side = 'left'

dataset = build_dataset_beaver_ppo(
    script_args.dataset_name, sft_tokenizer,
    rm_tokenizer=reward_models.rm_tokenizers[0], split='train')
instructions = Instructions()

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names: dataset = dataset.remove_columns(key)
dataset       = dataset.with_format('numpy')
data_collator = DataCollatorWithPadding(tokenizer=sft_tokenizer)
print(f'Dataset size: {len(dataset)}', flush=True)

generation_kwargs = {
    'max_new_tokens': script_args.max_new_tokens, 'min_length': -1,
    'top_k': 0, 'top_p': 0.9, 'temperature': 0.7,
    'do_sample': script_args.do_sample,
}

# ── Load expert models ─────────────────────────────────────────────────────
expert_models = load_shared_experts(
    script_args.base_model_name, script_args.expert_model_paths, device,
    tokenizer=sft_tokenizer)

lm_hidden_size = expert_models[0].config.hidden_size
_num_layers    = len(expert_models[0].model.layers)
print(f'lm_hidden_size = {lm_hidden_size}, num_layers = {_num_layers}', flush=True)

# ── Gating template (per_layer only) ───────────────────────────────────────
assert script_args.gating_type == 'per_layer', \
    'refinement currently supports gating_type=per_layer only'
template_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=len(expert_models),
    num_layers=_num_layers,
    alpha_init=script_args.fixed_alpha,
    fixed_alpha=script_args.fixed_alpha,
)

moe_model = MoEForCausalLM(expert_models, template_net).to(device)
moe_model.eval()  # experts always eval; gating set to train inside refiner

# ── Load population + compute optimal preferences (rank 0 writes manifest) ─
manifest_path = os.path.join(output_dir, 'preferences.json')
if is_main:
    population = _load_population(script_args.population_dir, template_net, device='cpu')
    fitnesses  = np.array([fit for _, fit in population], dtype=np.float32)
    print(f'Loaded {len(population)} individuals from {script_args.population_dir}', flush=True)
    prefs = compute_optimal_preferences(fitnesses, resolution=script_args.pref_resolution)
    dominated = [i for i, p in enumerate(prefs) if p is None]
    non_dominated = [i for i, p in enumerate(prefs) if p is not None]
    if script_args.individual_indices:
        requested = [int(x) for x in script_args.individual_indices.split(',')]
        keep      = [i for i in requested if i in set(non_dominated)]
        skipped   = [i for i in requested if i in set(dominated)]
        if skipped:
            print(f'Dropping dominated individuals from requested set: {skipped}', flush=True)
    else:
        keep = non_dominated
    if dominated:
        print(f'Dominated individuals dropped (no μ makes them argmax): {dominated}', flush=True)
    manifest = {
        'population_dir':    script_args.population_dir,
        'reward_names':      reward_names,
        'fitnesses':         fitnesses.tolist(),
        'optimal_prefs':     [(p.tolist() if p is not None else None) for p in prefs],
        'individuals':       keep,
        'dominated':         dominated,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest written → {manifest_path}', flush=True)
    for i in keep:
        print(f'  ind_{i:03d}  fitness={np.round(fitnesses[i], 3)}  '
              f'μ*={np.round(prefs[i], 3)}', flush=True)
else:
    while not os.path.exists(manifest_path):
        time.sleep(1.0)
    with open(manifest_path) as f:
        manifest = json.load(f)

prefs = [(np.array(p, dtype=np.float32) if p is not None else None)
         for p in manifest['optimal_prefs']]
keep  = list(manifest['individuals'])

# ── File-based work queue: one task per individual ─────────────────────────
queue_root = os.path.join(output_dir, 'queue')
if is_main:
    if os.path.exists(queue_root):
        shutil.rmtree(queue_root)
    os.makedirs(queue_root)
    # Materialize per-individual init params files so each rank can read independently
    # without re-walking the population directory.
    population = _load_population(script_args.population_dir, template_net, device='cpu')
    for i in keep:
        np.save(os.path.join(queue_root, f'init_{i:03d}.npy'), population[i][0])
    open(os.path.join(queue_root, 'init_ready'), 'w').close()
else:
    while not os.path.exists(os.path.join(queue_root, 'init_ready')):
        time.sleep(1.0)


def _claim_path(i):  return os.path.join(queue_root, f'claimed_{i:03d}')
def _done_path(i):   return os.path.join(refined_dir, f'ind_{i:03d}', 'done')


def _try_claim(i):
    try:
        fd = os.open(_claim_path(i), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(rank).encode()); os.close(fd); return True
    except FileExistsError:
        return False


# ── Refinement loop ─────────────────────────────────────────────────────────
refiner = PPORefiner(
    moe_model=moe_model,
    gating_template=template_net,
    sft_tokenizer=sft_tokenizer,
    reward_models=reward_models,
    instructions=instructions,
    device=device,
    gpu_id=gpu_id,
)

print(f'\nStarting PPO refinement — {len(keep)} individuals to refine, '
      f'rank={rank}/{num_ranks}', flush=True)

for ind_i in keep:
    if os.path.exists(_done_path(ind_i)):
        continue
    if not _try_claim(ind_i):
        continue

    ind_out = os.path.join(refined_dir, f'ind_{ind_i:03d}')
    os.makedirs(ind_out, exist_ok=True)
    print(f'[rank{rank}] refining ind_{ind_i:03d} '
          f'μ*={np.round(prefs[ind_i], 3)} …', flush=True)

    init_params = np.load(os.path.join(queue_root, f'init_{ind_i:03d}.npy'))

    def _save_cb(it, params, log, _ind_out=ind_out):
        net = params_to_net(params, template_net, 'cpu')
        save_gating_network(net, _ind_out)
        with open(os.path.join(_ind_out, 'training_log.json'), 'w') as f:
            json.dump(log, f, indent=2)

    refined_params, training_log = refiner.refine(
        params_init=init_params,
        preference=prefs[ind_i],
        dataset=dataset, data_collator=data_collator,
        generation_kwargs=generation_kwargs,
        num_iterations=script_args.num_iterations,
        ppo_epochs=script_args.ppo_epochs,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.eval_batch_size,
        learning_rate=script_args.learning_rate,
        clip_range=script_args.clip_range,
        init_kl_coef=script_args.init_kl_coef,
        value_coef=script_args.value_coef,
        entropy_coef=script_args.entropy_coef,
        max_grad_norm=script_args.max_grad_norm,
        target_kl=script_args.target_kl,
        seed=script_args.seed + ind_i,
        verbose=script_args.verbose,
        rank=rank,
        save_cb=_save_cb,
    )

    # Final save for this individual
    net = params_to_net(refined_params, template_net, 'cpu')
    save_gating_network(net, ind_out)
    with open(os.path.join(ind_out, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=2)
    with open(os.path.join(ind_out, 'preference.json'), 'w') as f:
        json.dump({'preference': prefs[ind_i].tolist(),
                   'reward_names': reward_names}, f, indent=2)
    open(_done_path(ind_i), 'w').close()
    print(f'[rank{rank}] ind_{ind_i:03d} refinement complete → {ind_out}', flush=True)

# Wait for all refinements to complete (file-based barrier)
while True:
    if all(os.path.exists(_done_path(i)) for i in keep):
        break
    time.sleep(2.0)

if is_main:
    print(f'\nAll {len(keep)} individuals refined → {refined_dir}', flush=True)
