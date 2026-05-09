"""evolutionary_soups.py — Evolve GatingNetwork parameters using standard NSGA-based algorithm.

NSGA-II (Deb et al., 2002).
  - Population of P individuals (flat GatingNetwork parameter vectors).
  - No preference vector during training — purely Pareto-based selection.
  - Each individual is evaluated ONCE to produce an M-dimensional reward vector.
  - Selection: non-dominated sort + crowding distance on raw reward vectors.
  - Parallel evaluation via file-based work queue (same as moead.py).
  - At inference: given preference λ, select best individual by argmax dot(λ, r_i).
"""

import copy
import datetime
import time
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_ppo, build_dataset_summary_ppo, build_dataset_news_summary_ppo,
    build_dataset_beaver_ppo, build_dataset_steer_ppo, build_dataset_ultrafeedback_ppo,
    get_clean_data, load_main_tokenizer,
)
from nsgaii_architecture import GatingNetwork, MoEForCausalLM
from nsgaii_utils import save_gating_network, get_simplex_samples, REWARD_PATHS


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:      str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths:   List[str] = field(default_factory=list)
    reward_names:         str       = 'harmless,helpful'          # auto-selected from dataset_name if empty
    dataset_name:         str       = 'Anthropic/hh-rlhf'         # 'Anthropic/hh-rlhf' | 'openai/summarize_from_feedback' | 'argilla/news-summary'
    do_sample:            bool      = False
    num_continuations:    int       = 1
    eval_prompts:         int       = 8192
    eval_batch_size:      int       = 128
    max_new_tokens:       int       = -1   # -1 = auto-derive from dataset_name (128 hh-rlhf, 48 summary)
    normalize_rewards:    bool      = False
    warm_start_path:      str       = ''
    # Algorithm selection
    algorithm:            str       = 'nsga4'   # 'nsgaii' | 'nsgaiii' | 'nsga4'
    n_reference_divisions: int      = 12         # Das-Dennis divisions for NSGA-III reference points

    # Evolutionary hyper-parameters
    population_size:      int       = 20
    num_generations:      int       = 100
    convergence_window:   int       = 10
    mutation_sigma:       float     = 0.05
    mutation_rate:        float     = 0.5
    sigma_decay:          float     = 0.99
    sigma_min:            float     = 0.03 # sigma floor; sigma_decay=0.97→σ_min@gen50, 0.95→@gen30
# ---------------------------------------------------------------------------
    parent_stability_bonus:  float  = 0.01  # per-gen bonus on parent baseline for front2
                                             # effective = min(gen * bonus, cap)
    parent_stability_cap:    float  = 0.10  # max bonus (10 %)
    fixed_alpha:          float     = 1.2   # entmax α fixed for all layers (1.0=softmax, 2.0=sparsemax)
    normalize_fitness:    bool      = True  # min-max normalize fitness by expert model bounds
# ---------------------------------------------------------------------------
    gpu_id:               int       = -1
    save_directory:       str       = './models/ES/'
    run_name:             str       = 'es_gating'
    save_every:           int       = 5
    seed:                 int       = 8888
    verbose:              bool      = False


# ---------------------------------------------------------------------------
# Genome ↔ GatingNetwork
# ---------------------------------------------------------------------------

def net_to_params(net: GatingNetwork) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().float().numpy().ravel() for p in net.parameters()]
    )


def params_to_net(params: np.ndarray, template: GatingNetwork,
                  device: str = 'cpu') -> GatingNetwork:
    net = copy.deepcopy(template).to(device)
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(params[offset:offset + n].reshape(p.shape),
                                 dtype=p.dtype, device=device))
            offset += n
    return net


def _make_onehot_params(template: GatingNetwork, expert_idx: int) -> np.ndarray:
    """Return params that force gating to output one-hot[expert_idx] for any input.

    Zeroes the last linear weight so only the bias matters, then sets
    bias[expert_idx]=+100 and all others=-100. Entmax → exactly one-hot.
    """
    net = copy.deepcopy(template)
    with torch.no_grad():
        last = net.net[-1]          # nn.Linear(hidden_size, num_experts)
        last.weight.zero_()
        last.bias.fill_(-100.0)
        last.bias[expert_idx] = 100.0
    return net_to_params(net)


# ---------------------------------------------------------------------------
# Reward evaluation
# ---------------------------------------------------------------------------

def generate_and_score(
    moe_model, prompt_input_ids, prompt_attention,
    sft_tokenizer, reward_models, instructions,
    generation_kwargs, gpu_id, num_continuations=1,
) -> np.ndarray:
    accumulated     = None
    prompts_decoded = sft_tokenizer.batch_decode(prompt_input_ids.cpu())

    for _ in range(num_continuations):
        outputs = moe_model.generate(
            prompt_input_ids.to(f'cuda:{gpu_id}'),
            attention_mask=prompt_attention.to(f'cuda:{gpu_id}'),
            **generation_kwargs,
        )
        responses = sft_tokenizer.batch_decode(outputs.cpu())
        del outputs

        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(instructions.get_input(r), instructions.get_response(r))
                 for r in responses_clean]
        if hasattr(instructions, 'get_post'):
            scores = reward_models.get_reward_model_scores(
                pairs, instructions.get_post,
                normalize_rewards=script_args.normalize_rewards, round_digits=None)
        else:
            scores = reward_models.get_reward_model_scores(
                pairs, normalize_rewards=script_args.normalize_rewards, round_digits=None)

        n_prompts, n_rewards = len(prompts_clean), len(scores)
        if accumulated is None:
            accumulated = [[[] for _ in range(n_rewards)] for _ in range(n_prompts)]
        for p in range(n_prompts):
            for k in range(n_rewards):
                accumulated[p][k].append(scores[k][p])

    torch.cuda.empty_cache()
    per_prompt = np.array([[np.mean(accumulated[p][k]) for k in range(n_rewards)]
                           for p in range(n_prompts)])
    return per_prompt.mean(axis=0)   # (M,)


# ---------------------------------------------------------------------------
# NSGA-II selection
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def _non_dominated_sort(rewards: np.ndarray) -> List[List[int]]:
    P = len(rewards)
    dom_count = np.zeros(P, dtype=int)
    dominates = [[] for _ in range(P)]
    for i in range(P):
        for j in range(i + 1, P):
            if _dominates(rewards[i], rewards[j]):
                dominates[i].append(j); dom_count[j] += 1
            elif _dominates(rewards[j], rewards[i]):
                dominates[j].append(i); dom_count[i] += 1
    fronts = [[i for i in range(P) if dom_count[i] == 0]]
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominates[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def _crowding_distance(rewards: np.ndarray, front: List[int]) -> np.ndarray:
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    r    = rewards[front]
    for m in range(r.shape[1]):
        order = np.argsort(r[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = r[order[-1], m] - r[order[0], m]
        if span == 0:
            continue
        for idx in range(1, n - 1):
            dist[order[idx]] += (r[order[idx + 1], m] - r[order[idx - 1], m]) / span
    return dist


def _nsga2_select(rewards: np.ndarray, pop_size: int) -> List[int]:
    fronts   = _non_dominated_sort(rewards)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            remaining = pop_size - len(selected)
            dist      = _crowding_distance(rewards, front)
            ranked    = [x for _, x in sorted(zip(-dist, front))]
            selected.extend(ranked[:remaining])
            break
    return selected


# ---------------------------------------------------------------------------
# NSGA-III helpers
# ---------------------------------------------------------------------------

def _generate_reference_points(n_objectives: int, n_divisions: int) -> np.ndarray:
    """Das-Dennis structured reference points on the unit hyperplane (sum=1, ≥0).

    Returns array of shape (C(n_objectives+n_divisions-1, n_divisions), n_objectives).
    Typical n_divisions: 12 for M=2 (13 pts), 12 for M=3 (91 pts), 7 for M=4 (120 pts).
    """
    def _gen(n_obj: int, n_div: int, cur: list, result: list) -> None:
        if n_obj == 1:
            result.append(cur + [n_div])
        else:
            for i in range(n_div + 1):
                _gen(n_obj - 1, n_div - i, cur + [i], result)
    pts: list = []
    _gen(n_objectives, n_divisions, [], pts)
    return np.array(pts, dtype=np.float32) / n_divisions


def _nsga3_select(rewards: np.ndarray, pop_size: int,
                  reference_points: np.ndarray) -> List[int]:
    """NSGA-III selection: non-dominated sorting + niche-preservation on the last front.

    Core idea (Deb & Jain, 2014):
      1. Fill selection with complete fronts until the last (critical) front.
      2. Normalize the combined fitness onto a unit hyperplane using the ideal
         point and per-objective intercepts.
      3. Associate every individual to its nearest reference *line* (through the
         origin) by minimum perpendicular distance.
      4. Pick n_needed individuals from the last front by iteratively selecting
         the reference point with the smallest niche count, then choosing the
         individual closest to that line (first use) or randomly (subsequent uses).
    """
    fronts   = _non_dominated_sort(rewards)
    selected: List[int] = []
    last_front: List[int] = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            last_front = list(front)
            break

    n_needed = pop_size - len(selected)
    if n_needed == 0 or not last_front:
        return selected[:pop_size]

    # ── Normalize fitness onto unit hyperplane ────────────────────────────────
    all_idx    = selected + last_front
    fit_all    = rewards[all_idx]                          # (N_all, M)
    ideal      = fit_all.min(axis=0)                       # (M,)
    translated = fit_all - ideal

    # Nadir: per-objective max among already-accepted (non-last-front) individuals;
    # fall back to max of all if no individuals were accepted yet.
    if selected:
        nadir = (rewards[selected] - ideal).max(axis=0)
    else:
        nadir = translated.max(axis=0)
    nadir      = np.where(nadir < 1e-10, 1.0, nadir)
    normalized = translated / nadir                        # (N_all, M)

    # ── Associate each individual to its nearest reference line ───────────────
    # Perpendicular distance from point p to line through origin with direction r:
    #   d_perp² = ||p||² - (p·r̂)²
    # We compute this for all (individual, ref_point) pairs vectorised.
    R       = reference_points                             # (R, M)
    r_norms = np.linalg.norm(R, axis=1, keepdims=True)    # (R, 1)
    r_norms = np.where(r_norms < 1e-10, 1.0, r_norms)
    R_hat   = R / r_norms                                  # (R, M) unit reference directions

    dot      = normalized @ R_hat.T                        # (N_all, R)  scalar projections
    proj     = dot[:, :, None] * R_hat[None, :, :]        # (N_all, R, M) projected vectors
    diff     = normalized[:, None, :] - proj               # (N_all, R, M) perpendicular components
    perp_dist = np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))  # (N_all, R)

    assoc_ref  = perp_dist.argmin(axis=1)                  # (N_all,) closest ref index
    assoc_dist = perp_dist.min(axis=1)                     # (N_all,) perp distance

    # ── Niche counts from already-accepted individuals (before last front) ─────
    n_sel       = len(selected)
    niche_count = np.zeros(len(R), dtype=int)
    for j in range(n_sel):
        niche_count[assoc_ref[j]] += 1

    # Associations for individuals in last_front (offset n_sel in all_idx)
    lf_assoc = assoc_ref[n_sel:]                           # (|last_front|,)
    lf_dist  = assoc_dist[n_sel:]

    # ── Niche-based selection from last_front ─────────────────────────────────
    remaining        = list(range(len(last_front)))
    chosen_from_last: List[int] = []

    for _ in range(n_needed):
        # Map each active reference point to candidate local indices
        ref_to_cands: dict = {}
        for loc in remaining:
            ref_to_cands.setdefault(lf_assoc[loc], []).append(loc)

        if not ref_to_cands:
            break

        # Reference point with minimum niche count (ties broken randomly)
        min_nc    = min(niche_count[r] for r in ref_to_cands)
        min_refs  = [r for r in ref_to_cands if niche_count[r] == min_nc]
        chosen_r  = min_refs[np.random.randint(len(min_refs))]

        cands = ref_to_cands[chosen_r]
        if niche_count[chosen_r] == 0:
            # First occupant of this niche: pick the closest individual to the line
            loc = cands[int(np.argmin(lf_dist[cands]))]
        else:
            # Niche already occupied: pick randomly among candidates
            loc = cands[np.random.randint(len(cands))]

        chosen_from_last.append(last_front[loc])
        niche_count[chosen_r] += 1
        remaining.remove(loc)

    return selected + chosen_from_last


from pymoo.indicators.hv import HV as _PymooHV


def _hv(points: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume (maximisation) via pymoo.

    pymoo uses minimisation convention: negate both points and ref.
    """
    ind = _PymooHV(ref_point=-ref)
    return float(ind(-points))


def _hypervolume_contribution(candidates: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Exclusive hypervolume contribution for each candidate point.

    HVC[i] = HV(candidates) - HV(candidates \\ {i}).
    """
    n        = len(candidates)
    total_hv = _hv(candidates, ref)
    hvc      = np.zeros(n)
    for i in range(n):
        hvc[i] = total_hv - _hv(np.delete(candidates, i, axis=0), ref)
    return hvc


def _nsga4_select(rewards: np.ndarray, pop_size: int) -> List[int]:
    """NSGA-IV selection: non-dominated sorting + greedy hypervolume contribution on last front.

    Within the critical last front, individuals are ranked by their exclusive
    hypervolume contribution (HVC) relative to a dynamic nadir reference point
    (per-generation minimum - 0.1).  Higher HVC = larger unique coverage of
    objective space = selected first.  This naturally retains both extreme
    (corner) points and well-spread intermediate points without requiring
    pre-specified reference lines or divisions.
    """
    fronts   = _non_dominated_sort(rewards)
    selected: List[int] = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            n_needed   = pop_size - len(selected)
            candidates = np.array([rewards[i] for i in front])   # (|front|, M)

            # Dynamic nadir: minimum across ALL individuals in the merged pool
            # (selected + last front), shifted down to ensure ref is below all points.
            all_pts = np.vstack([rewards[selected], candidates]) if selected else candidates
            ref     = all_pts.min(axis=0) - 0.1

            hvc     = _hypervolume_contribution(candidates, ref)
            ranked  = [x for _, x in sorted(zip(-hvc, front))]
            selected.extend(ranked[:n_needed])
            break

    return selected



# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------

class NSGAII:
    """Standard NSGA-II for GatingNetwork evolution.

    Population: P flat param vectors (no λ association).
    Each individual evaluated ONCE → M-dim reward vector as fitness.

    Task layout per generation (total 2·P tasks):
      task IDs  0 .. P-1    : child   evals
      task IDs  P .. 2P-1   : parent re-evals
    """

    def __init__(
        self,
        template_net:    GatingNetwork,
        num_objectives:  int,
        population_size: int = 20,
        device:          str = 'cpu',
    ):
        self.template   = template_net.eval()
        self.M          = num_objectives
        self.P          = population_size
        self.device     = device

        base_params     = net_to_params(template_net)
        self.param_dim  = len(base_params)
        self.population = [
            base_params + np.random.randn(self.param_dim) * 0.05
            for _ in range(self.P)
        ]

        # fitness[i]:        raw reward vector from the latest evaluation
        # parent_baseline[i]: cumulative mean of raw fitness across all generations
        # parent_eval_count:  number of generations each parent has been evaluated
        self.fitness          = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.parent_baseline  = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.parent_eval_count = [0] * self.P

        self.z_star          = np.full(self.M, -np.inf, dtype=np.float32)
        self.fitness_history = []
        self.front_history   = []

    def _update_z_star(self, r: np.ndarray):
        improved = r > self.z_star
        self.z_star[improved] = r[improved]

    @staticmethod
    def _log(rank, msg, verbose=True):
        if verbose:
            print(f'[{datetime.datetime.now():%H:%M:%S}] rank{rank} {msg}', flush=True)

    @staticmethod
    def _crossover(p1: np.ndarray, p2: np.ndarray) -> tuple:
        alpha = np.random.uniform(0.3, 0.7, size=p1.shape)
        return alpha * p1 + (1.0 - alpha) * p2, float(alpha.mean())

    @staticmethod
    def _mutate(x: np.ndarray, sigma: float, rate: float) -> np.ndarray:
        mask = np.random.random(x.shape) < rate
        return x + mask.astype(x.dtype) * np.random.normal(0.0, sigma, x.shape).astype(x.dtype)

    def _save_checkpoint(self, output_dir: str, gen: int):
        ckpt_dir = os.path.join(output_dir, f'gen_{gen:04d}')
        os.makedirs(ckpt_dir, exist_ok=True)
        for i in range(self.P):
            subdir = os.path.join(ckpt_dir, f'ind_{i:03d}')
            net    = params_to_net(self.population[i], self.template, 'cpu')
            save_gating_network(net, subdir)
            with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
                json.dump({'fitness': self.fitness[i].tolist()}, f, indent=2)
        meta = {
            'generation': gen,
            'z_star':     self.z_star.tolist(),
            'fitness':    [f.tolist() for f in self.fitness],
        }
        with open(os.path.join(ckpt_dir, 'nsgaii_state.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  Checkpoint saved → {ckpt_dir}', flush=True)

    def run(
        self,
        dataset,
        data_collator,
        eval_prompts:      int,
        eval_batch_size:   int,
        moe_model:         MoEForCausalLM,
        sft_tokenizer,
        reward_models,
        instructions,
        generation_kwargs: dict,
        gpu_id:            int,
        num_generations:   int   = 100,
        mutation_sigma:    float = 0.05,
        mutation_rate:     float = 0.3,
        sigma_decay:       float = 0.999,
        num_continuations: int   = 1,
        save_every:        int   = 10,
        output_dir:        str   = '.',
        poll_interval:     float = 2.0,
        verbose:           bool  = False,
        seed:              int   = 42,
        sigma_min:               float = 0.005,
        convergence_window:      int   = 10,
        parent_stability_bonus:  float = 0.01,
        parent_stability_cap:    float = 0.10,
        algorithm:               str   = 'nsgaii',
        n_reference_divisions:   int   = 12,
        normalize_fitness:       bool  = False,
    ) -> List[np.ndarray]:

        dist_on = torch.distributed.is_initialized()
        rank    = torch.distributed.get_rank() if dist_on else 0
        is_main = rank == 0

        # ── Algorithm-specific setup ──────────────────────────────────────────
        assert algorithm in ('nsgaii', 'nsgaiii', 'nsga4'), \
            f"algorithm must be 'nsgaii', 'nsgaiii', or 'nsga4', got {algorithm!r}"
        if algorithm == 'nsgaiii':
            ref_pts = _generate_reference_points(self.M, n_reference_divisions)
            _select = lambda fit, n: _nsga3_select(fit, n, ref_pts)
            if is_main:
                print(f'NSGA-III: {len(ref_pts)} reference points '
                      f'(M={self.M}, divisions={n_reference_divisions})')
        elif algorithm == 'nsga4':
            _select = _nsga4_select
            if is_main:
                print(f'NSGA-IV: max-Tchebycheff utility selection (M={self.M})')
        else:
            _select = _nsga2_select

        def log(msg): self._log(rank, msg, verbose)

        queue_root = os.path.join(output_dir, 'queue')
        if is_main:
            if os.path.exists(queue_root):
                shutil.rmtree(queue_root)
            os.makedirs(queue_root)

        # ── Queue helpers ─────────────────────────────────────────────────────
        def _gen_dir(g):        return os.path.join(queue_root, f'gen_{g:04d}')
        def _task_path(g, i):   return os.path.join(_gen_dir(g), f'task_{i:03d}.json')
        def _claim_path(g, i):  return os.path.join(_gen_dir(g), f'claimed_{i:03d}')
        def _result_path(g, i): return os.path.join(_gen_dir(g), f'result_{i:03d}.json')
        def _done_path(g):        return os.path.join(_gen_dir(g), 'done')
        def _tasks_ready_path(g): return os.path.join(_gen_dir(g), 'tasks_ready')
        _converged_path = os.path.join(queue_root, 'converged')

        def _try_claim(g, i):
            try:
                fd = os.open(_claim_path(g, i), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(rank).encode()); os.close(fd); return True
            except FileExistsError:
                return False

        def _write_task(g, tid, task):
            tmp = _task_path(g, tid) + '.tmp'
            with open(tmp, 'w') as f: json.dump(task, f)
            os.replace(tmp, _task_path(g, tid))

        def _write_result(g, i, r):
            tmp = _result_path(g, i) + f'.tmp_rank{rank}'
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            with open(tmp, 'w') as f: json.dump({'task_id': i, 'reward_vec': r.tolist()}, f)
            os.replace(tmp, _result_path(g, i))

        # ── Dataset chunks ────────────────────────────────────────────────────
        _chunk_size  = len(dataset) if eval_prompts <= 0 else min(eval_prompts, len(dataset))
        _rng         = np.random.default_rng(seed)
        _shuffled    = _rng.permutation(len(dataset)).tolist()
        _num_chunks  = max(1, (len(dataset) + _chunk_size - 1) // _chunk_size)
        chunk_loaders = [
            DataLoader(dataset.select(_shuffled[c*_chunk_size:(c+1)*_chunk_size]),
                       batch_size=eval_batch_size, collate_fn=data_collator, drop_last=False)
            for c in range(_num_chunks)
        ]
        print(f'Dataset chunks: {_num_chunks} × {_chunk_size} prompts', flush=True)

        # ── Eval helper ───────────────────────────────────────────────────────
        def _eval_individual(params, chunk_idx, label=''):
            log(f'eval [{label}] chunk={chunk_idx % _num_chunks}')
            loader = chunk_loaders[chunk_idx % _num_chunks]
            net    = params_to_net(params, self.template, self.device)
            net.eval()
            moe_model.gating_net = net
            reward_vecs = []
            for batch in loader:
                r = generate_and_score(
                    moe_model, batch['input_ids'], batch['attention_mask'],
                    sft_tokenizer, reward_models, instructions,
                    generation_kwargs, gpu_id, num_continuations)
                reward_vecs.append(r)
            return np.mean(reward_vecs, axis=0)

        # ── Worker loop ───────────────────────────────────────────────────────
        def _worker_loop(gen, task_start, task_end, exit_signal):
            while True:
                if is_main:
                    if all(os.path.exists(_result_path(gen, i)) for i in range(task_start, task_end)):
                        break
                else:
                    if os.path.exists(exit_signal):
                        break
                claimed = False
                for i in range(task_start, task_end):
                    if not os.path.exists(_task_path(gen, i)): continue
                    if os.path.exists(_result_path(gen, i)):   continue
                    if not _try_claim(gen, i):                 continue
                    claimed = True
                    with open(_task_path(gen, i)) as f: task = json.load(f)
                    r = _eval_individual(np.array(task['child_params']),
                                         task['chunk_idx'], label=f'g{gen}/t{i}')
                    _write_result(gen, i, r)
                    break
                if not claimed:
                    time.sleep(poll_interval)

        # ── Collect fitness from one task result ──────────────────────────────
        # bounds is a mutable dict so _collect_fitness can see updates made after
        # it is defined (expert bounds evaluation fills it before gen 0).
        bounds: dict = {'min': None, 'range': None}

        def _collect_fitness(gen, task_id) -> np.ndarray:
            with open(_result_path(gen, task_id)) as f:
                r = np.array(json.load(f)['reward_vec'])
            if bounds['range'] is not None:
                r = (r - bounds['min']) / bounds['range']
            self._update_z_star(r)
            return r

        # ── Expert reward bounds (normalize_fitness=True) ─────────────────────
        # Evaluate each expert individually (one-hot gating) on chunk 0 to get
        # per-objective min/max.  Uses the same task-queue infrastructure as
        # gen 0 so all workers participate in parallel.
        if normalize_fitness:
            _BG = -1   # bounds gen → directory 'gen_-001'
            if is_main:
                print(f'Computing expert reward bounds '
                      f'({self.template.num_experts} experts, chunk 0) …', flush=True)
                os.makedirs(_gen_dir(_BG), exist_ok=True)
                for exp_idx in range(self.template.num_experts):
                    _write_task(_BG, exp_idx, {
                        'task_id':      exp_idx,
                        'chunk_idx':    0,
                        'child_params': _make_onehot_params(self.template, exp_idx).tolist(),
                    })

            _worker_loop(_BG, 0, self.template.num_experts, _done_path(_BG))

            if is_main:
                expert_rewards = []
                for exp_idx in range(self.template.num_experts):
                    with open(_result_path(_BG, exp_idx)) as f:
                        r = np.array(json.load(f)['reward_vec'])
                    expert_rewards.append(r)
                    print(f'  expert {exp_idx}: {np.round(r, 3)}', flush=True)
                r_min   = np.min(expert_rewards, axis=0)
                r_max   = np.max(expert_rewards, axis=0)
                r_range = np.maximum(r_max - r_min, 1e-6)
                bounds['min']   = r_min
                bounds['range'] = r_range
                print(f'  reward_min={np.round(r_min, 3)}', flush=True)
                print(f'  reward_max={np.round(r_max, 3)}', flush=True)
                open(_done_path(_BG), 'w').close()
            else:
                while not os.path.exists(_done_path(_BG)):
                    time.sleep(poll_interval)

        # ── Phase 0: initial population ───────────────────────────────────────
        gen = 0
        if is_main:
            print(f'Evolutionary Soups — initialising population ({self.P} individuals, '
                  f'1 eval each = {self.P} tasks) …')
            os.makedirs(_gen_dir(gen), exist_ok=True)
            for i in range(self.P):
                _write_task(gen, i, {'task_id': i,
                                     'chunk_idx': 0,
                                     'child_params': self.population[i].tolist()})
            log(f'gen 0 all {self.P} init tasks written (chunk 0)')

        _worker_loop(gen, 0, self.P, _done_path(gen))

        if is_main:
            for i in range(self.P):
                self.fitness[i] = _collect_fitness(gen, i)
                self.parent_baseline[i]   = self.fitness[i].copy()
                self.parent_eval_count[i] = 1
            open(_done_path(gen), 'w').close()
            log(f'gen 0 done, z*={np.round(self.z_star, 3)}')
            print(f'Gen {gen:4d}/{num_generations}')

        else:
            while not os.path.exists(_done_path(gen)):
                time.sleep(poll_interval)

        # ── Generational loop ─────────────────────────────────────────────────
        # Single worker loop per generation evaluates 2P individuals on the same chunk:
        #   tasks 0..P-1  : parent re-evals (same chunk as children → fair comparison)
        #   tasks P..2P-1 : children

        sigma             = mutation_sigma
        parents_kept_hist: list = []
        diversity_hist:    list = []
        z_star_hist:       list = []
        for gen in range(1, num_generations + 1):
            log(f'gen {gen}/{num_generations} start')

            if is_main:
                os.makedirs(_gen_dir(gen), exist_ok=True)
                chunk_idx = gen % _num_chunks
                diversity = np.std(np.array(self.population), axis=0).mean()

                parent_params = list(self.population)

                xover_params = parent_params
                xover_size   = len(xover_params)

                # Write parent re-eval tasks (0..P-1)
                for i in range(self.P):
                    _write_task(gen, i, {'task_id': i, 'chunk_idx': chunk_idx,
                                         'child_params': self.population[i].tolist()})

                # Crossover → write child tasks (P..2P-1)
                child_params_list = []
                for i in range(self.P):
                    pi1 = np.random.randint(xover_size)
                    pi2 = np.random.randint(xover_size)
                    child, _ = self._crossover(xover_params[pi1], xover_params[pi2])
                    child = self._mutate(child, sigma, mutation_rate)
                    child_params_list.append(child)
                    _write_task(gen, self.P + i, {'task_id': self.P + i, 'chunk_idx': chunk_idx,
                                                   'child_params': child.tolist()})
                log(f'gen {gen} crossover: parents={self.P}, chunk={chunk_idx}')

                open(_tasks_ready_path(gen), 'w').close()
            else:
                while not os.path.exists(_tasks_ready_path(gen)):
                    if os.path.exists(_converged_path):
                        break
                    time.sleep(poll_interval)
                if os.path.exists(_converged_path):
                    break

            _worker_loop(gen, 0, 2 * self.P, _done_path(gen))

            if is_main:
                # Collect parent results and update cumulative baseline
                for i in range(self.P):
                    raw = _collect_fitness(gen, i)
                    self.fitness[i] = raw
                    n = self.parent_eval_count[i]
                    if n == 0 or np.all(self.parent_baseline[i] == -np.inf):
                        self.parent_baseline[i] = raw.copy()
                    else:
                        # cumulative mean: baseline = (baseline * n + raw) / (n + 1)
                        self.parent_baseline[i] = (self.parent_baseline[i] * n + raw) / (n + 1)
                    self.parent_eval_count[i] = n + 1

                # Collect child results
                child_fitness = [_collect_fitness(gen, self.P + i) for i in range(self.P)]

                merged_params  = parent_params + child_params_list
                merged_fitness = np.vstack([np.array(self.fitness), np.array(child_fitness)])

                # Front 1: raw fitness — equal ground for parents and children.
                front1_set = set(_select(merged_fitness, self.P))

                # Front 2: parent baseline × (1 + per-parent bonus) vs child raw fitness.
                # Bonus = min(eval_count * rate, cap): older parents get a higher threshold,
                # so a child must beat the parent's proven historical average by an increasing
                # margin to displace it.
                parent_front2 = [
                    self.parent_baseline[i] * (1.0 + min(self.parent_eval_count[i] * parent_stability_bonus,
                                                          parent_stability_cap))
                    for i in range(self.P)
                ]
                merged_front2 = np.vstack([np.array(parent_front2), np.array(child_fitness)])
                front2_list   = _select(merged_front2, self.P)

                # Intersection + fill: survivor must pass both fronts.
                intersection = [k for k in front2_list if k in front1_set]
                extras       = [k for k in front2_list if k not in front1_set]
                selected     = intersection + extras[:self.P - len(intersection)]

                n_intersection = len(intersection)
                ind_labels     = [f'{j+1}-parent{k+1}' if k < self.P else f'{j+1}-child'
                                  for j, k in enumerate(selected)]
                n_parents_kept = sum(1 for k in selected if k < self.P)

                # Update population; reset baseline/count for newly selected children
                new_population    = [merged_params[k] for k in selected]
                new_fitness       = [merged_fitness[k] for k in selected]
                new_baseline      = []
                new_eval_count    = []
                for j, k in enumerate(selected):
                    if k < self.P:
                        new_baseline.append(self.parent_baseline[k].copy())
                        new_eval_count.append(self.parent_eval_count[k])
                    else:
                        # Child enters as new parent: baseline starts from its first raw eval
                        new_baseline.append(merged_fitness[k].copy())
                        new_eval_count.append(1)

                self.population       = new_population
                self.fitness          = new_fitness
                self.parent_baseline  = new_baseline
                self.parent_eval_count = new_eval_count

            # ── Post-selection bookkeeping and logging ────────────────────────
            if is_main:
                sigma = max(sigma_min, sigma * sigma_decay)

                fit_arr     = np.array(self.fitness)          # raw current-gen
                base_arr    = np.array(self.parent_baseline)  # cumulative mean
                fronts      = _non_dominated_sort(fit_arr)
                n_front0    = len(fronts[0]) if fronts else 0
                self.fitness_history.append(fit_arr.tolist())
                self.front_history.append(n_front0)

                # Append per-individual record to population log
                log_path = os.path.join(output_dir, 'population_log.json')
                try:
                    with open(log_path) as f:
                        pop_log = json.load(f)
                except FileNotFoundError:
                    pop_log = {}
                alpha_str = f'entmax_α={self.template.fixed_alpha:.3f}(fixed)'
                def _alpha_entry(j): return {}

                pop_log[f'gen_{gen:04d}'] = {
                    label: {
                        'raw':      fit_arr[j].tolist(),
                        'baseline': base_arr[j].tolist(),
                        **_alpha_entry(j),
                    }
                    for j, label in enumerate(ind_labels)
                }
                with open(log_path, 'w') as f:
                    json.dump(pop_log, f, indent=2)

                print(
                    f'Gen {gen:4d}/{num_generations} | '
                    f'front0={n_front0}/{self.P} | '
                    f'parents_kept={n_parents_kept}/{self.P} | '
                    f'intersect={n_intersection}/{self.P} | '
                    f'bonus={np.mean([min(self.parent_eval_count[i]*parent_stability_bonus, parent_stability_cap) for i in range(self.P)]):.3f} | '
                    f'mean_fit={np.round(fit_arr.mean(axis=0), 3)} | '
                    f'best_fit={np.round(fit_arr.max(axis=0), 3)} | '
                    f'z*={np.round(self.z_star, 3)} | '
                    f'σ={sigma:.5f} | '
                    f'{alpha_str} | '
                    f'div={diversity:.5f}',
                    flush=True,
                )

                if gen % save_every == 0:
                    self._save_checkpoint(output_dir, gen)

                # ── Convergence check ─────────────────────────────────────────
                converged = False
                if convergence_window > 0:
                    parents_kept_hist.append(n_parents_kept)
                    diversity_hist.append(diversity)
                    z_star_hist.append(self.z_star.copy())
                    if len(parents_kept_hist) >= convergence_window:
                        avg_kept  = np.mean(parents_kept_hist[-convergence_window:])
                        delta_div = abs(diversity_hist[-1] - diversity_hist[-convergence_window])
                        z_stable  = all(
                            np.array_equal(z_star_hist[-convergence_window], z)
                            for z in z_star_hist[-convergence_window:])
                        if avg_kept >= 16.5 and delta_div < 0.002 and z_stable:
                            print(f'  *** Convergence at gen {gen}: '
                                  f'avg_kept={avg_kept:.1f}/{self.P}, '
                                  f'Δdiv={delta_div:.4f} ***', flush=True)
                            converged = True

                open(_done_path(gen), 'w').close()
                if converged:
                    if gen % save_every != 0:
                        self._save_checkpoint(output_dir, gen)
                    open(_converged_path, 'w').close()
                    break
            else:
                while not os.path.exists(_done_path(gen)):
                    time.sleep(poll_interval)
                if os.path.exists(_converged_path):
                    break

        return self.population

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

output_dir = os.path.join(script_args.save_directory, script_args.run_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(script_args.seed)
np.random.seed(script_args.seed)

if 'RANK' in os.environ:
    torch.distributed.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=600))
accelerator = Accelerator()
gpu_id      = (script_args.gpu_id if script_args.gpu_id >= 0
               else accelerator.local_process_index)
device      = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_objectives = len(reward_names)

reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

sft_tokenizer              = load_main_tokenizer(script_args.expert_model_paths[0])
sft_tokenizer.padding_side = 'left'

if script_args.dataset_name == 'Anthropic/hh-rlhf':
    dataset = build_dataset_ppo(
        script_args.dataset_name, sft_tokenizer,
        reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    dataset = build_dataset_summary_ppo(
        script_args.dataset_name, sft_tokenizer,
        reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    dataset = build_dataset_beaver_ppo(
        script_args.dataset_name, sft_tokenizer,
        rm_tokenizer=reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    dataset = build_dataset_steer_ppo(
        script_args.dataset_name, sft_tokenizer,
        rm_tokenizer=reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.dataset_name == 'argilla/news-summary':
    # argilla/news-summary: train split has only ~1000 samples; test split has ~20k — use test
    dataset = build_dataset_news_summary_ppo(
        script_args.dataset_name, sft_tokenizer,
        reward_models.rm_tokenizers[0], split='test')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'openbmb/UltraFeedback':
    dataset = build_dataset_ultrafeedback_ppo(
        script_args.dataset_name, sft_tokenizer,
        rm_tokenizer=reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}. '
                     f'Choose from: Anthropic/hh-rlhf, openai/summarize_from_feedback, '
                     f'argilla/news-summary, PKU-Alignment/PKU-SafeRLHF-10K, '
                     f'nvidia/HelpSteer, nvidia/HelpSteer2, openbmb/UltraFeedback')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:     dataset     = dataset.remove_columns(key)

dataset = dataset.with_format("numpy")
data_collator = DataCollatorWithPadding(tokenizer=sft_tokenizer)
print(f'Dataset size: {len(dataset)} | eval_prompts per call: {script_args.eval_prompts}')

_max_new_tokens = (script_args.max_new_tokens if script_args.max_new_tokens > 0
                   else (128 if script_args.dataset_name in {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K', 'nvidia/HelpSteer', 'nvidia/HelpSteer2'} else 48))
generation_kwargs = {
    'max_new_tokens': _max_new_tokens, 'min_length': -1,
    'top_k': 0, 'top_p': 0.9, 'temperature': 0.7, 'do_sample': script_args.do_sample,
}

print(f'Loading {len(script_args.expert_model_paths)} expert models …')
expert_models = []
for i, path in enumerate(script_args.expert_model_paths):
    print(f'  Expert {i+1}: {path}')
    base = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.resize_token_embeddings(len(sft_tokenizer))
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    expert_models.append(m)
print(f'  All {len(expert_models)} experts loaded on {device}.')

lm_hidden_size = expert_models[0].config.hidden_size
_num_layers    = len(expert_models[0].model.layers)
print(f'lm_hidden_size = {lm_hidden_size}, num_layers = {_num_layers}')

template_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=len(expert_models),
    num_layers=_num_layers,
    alpha_init=script_args.fixed_alpha,
    fixed_alpha=script_args.fixed_alpha,
)

if script_args.warm_start_path:
    from nsgaii_utils import load_gating_network as _load
    loaded = _load(script_args.warm_start_path, lm_hidden_size=lm_hidden_size,
                   num_experts=len(expert_models), num_layers=_num_layers,
                   device='cpu')
    if loaded is not None:
        template_net = loaded.cpu().bfloat16()
        print(f'Warm-start from {script_args.warm_start_path}')

moe_model = MoEForCausalLM(expert_models, template_net).to(device)
moe_model.eval()
print(f'MoEForCausalLM: {len(expert_models)} experts × {moe_model.num_layers} layers')

nsgaii = NSGAII(
    template_net=template_net,
    num_objectives=n_objectives,
    population_size=script_args.population_size,
    device=device,
)

print(f'\nStarting Evolutionary Soups — {script_args.num_generations} generations, '
      f'P={nsgaii.P}, 1 eval per individual …')
final_population = nsgaii.run(
    dataset=dataset, data_collator=data_collator,
    eval_prompts=script_args.eval_prompts, eval_batch_size=script_args.eval_batch_size,
    moe_model=moe_model, sft_tokenizer=sft_tokenizer,
    reward_models=reward_models, instructions=instructions,
    generation_kwargs=generation_kwargs, gpu_id=gpu_id,
    num_generations=script_args.num_generations,
    mutation_sigma=script_args.mutation_sigma, mutation_rate=script_args.mutation_rate,
    sigma_decay=script_args.sigma_decay, num_continuations=script_args.num_continuations,
    save_every=script_args.save_every, output_dir=output_dir,
    verbose=script_args.verbose, seed=script_args.seed,
    sigma_min=script_args.sigma_min,
    convergence_window=script_args.convergence_window,
    parent_stability_bonus=script_args.parent_stability_bonus,
    parent_stability_cap=script_args.parent_stability_cap,
    algorithm=script_args.algorithm,
    n_reference_divisions=script_args.n_reference_divisions,
    normalize_fitness=script_args.normalize_fitness,
)

is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
if is_main:
    print(f'\nSaving final population to {output_dir}/final …')
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    for i, params in enumerate(final_population):
        subdir = os.path.join(final_dir, f'ind_{i:03d}')
        net    = params_to_net(params, template_net, 'cpu')
        save_gating_network(net, subdir)
        with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
            json.dump({'fitness': nsgaii.fitness[i].tolist()}, f, indent=2)

    fronts = _non_dominated_sort(np.array(nsgaii.fitness))
    with open(os.path.join(output_dir, 'nsgaii_meta.json'), 'w') as f:
        json.dump({
            'reward_names':    reward_names,
            'z_star':          nsgaii.z_star.tolist(),
            'final_fitness':   [f.tolist() for f in nsgaii.fitness],
            'fitness_history': nsgaii.fitness_history,
            'front_history':   nsgaii.front_history,
            'num_generations': script_args.num_generations,
            'population_size': nsgaii.P,
            'lm_hidden_size':  lm_hidden_size,
        }, f, indent=2)

    print(f'\nDone. Non-dominated front: {len(fronts[0])}/{nsgaii.P} individuals')
    print(f'Final ideal point z* = {np.round(nsgaii.z_star, 4)}')