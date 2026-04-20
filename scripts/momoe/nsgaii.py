"""nsgaii.py — Evolve GatingNetwork parameters using standard NSGA-II.

Algorithm: NSGA-II (Deb et al., 2002).
  - Population of P individuals (flat GatingNetwork parameter vectors).
  - No preference vector during training — purely Pareto-based selection.
  - Each individual is evaluated ONCE to produce an M-dimensional reward vector.
  - Selection: non-dominated sort + crowding distance on raw reward vectors.
  - Parallel evaluation via file-based work queue (same as moead.py).
  - At inference: given preference λ, select best individual by argmax dot(λ, r_i).

TO RUN:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./scripts/momoe/nsgaii.py \
    --sft_model_name './models/sft/assistant_sft/model/' \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' \
                         './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --run_name 'nsgaii_gating_0101' 2>&1 | tee ./logs/nsgaii_0101.log
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
from typing import List, Optional

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
    sft_model_name:       str       = './models/sft/model/'
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
    algorithm:            str       = 'nsgaii'   # 'nsgaii' | 'nsgaiii'
    n_reference_divisions: int      = 12         # Das-Dennis divisions for NSGA-III reference points

    # Evolutionary hyper-parameters
    population_size:      int       = 20
    num_generations:      int       = 100
    mutation_sigma:       float     = 0.05
    mutation_rate:        float     = 0.3
    sigma_decay:          float     = 0.999
    use_reward_map:       bool      = False
    use_guided_crossover: bool      = False  # True → p1 from F1, p2 front-decay weighted
                                             # False → both parents uniform-random (recommended)
    crossover_front_decay: float    = 0.5   # only used when use_guided_crossover=True
    fitness_ema_alpha:    float     = 1.0   # EMA smoothing: ema = α·raw + (1-α)·ema_prev
                                            # <1.0 triggers Phase A (parent re-eval every gen)
                                            # 1.0 = no EMA, no Phase A (pure single-phase)

    gpu_id:               int       = -1
    save_directory:       str       = './models/nsgaii/'
    run_name:             str       = 'nsgaii_gating'
    save_every:           int       = 5
    seed:                 int       = 8888
    verbose:              bool      = False


# ---------------------------------------------------------------------------
# Genome ↔ GatingNetwork
# ---------------------------------------------------------------------------

def net_to_params(net: GatingNetwork) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in net.parameters()]
    ).astype(np.float64)


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
    return np.array(pts, dtype=float) / n_divisions


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


# ---------------------------------------------------------------------------
# Reward map — per-chunk Pareto archive
# ---------------------------------------------------------------------------

class RewardMap:
    """Per-chunk Pareto archive: maps chunk_idx → Pareto-optimal (params, fitness) pairs.

    When the evolutionary loop revisits a chunk (every _num_chunks generations),
    stored candidates are injected into the parent pool so that historical best
    solutions can participate in crossover and NSGA selection — without re-evaluation,
    because same chunk_idx means the same set of prompts.
    """

    def __init__(self) -> None:
        self._chunks: dict = {}   # int → List[Tuple[np.ndarray, np.ndarray]]

    def update(self, chunk_idx: int, params: np.ndarray, fitness: np.ndarray) -> bool:
        """Add (params, fitness) to the Pareto front for chunk_idx.

        Prunes dominated entries; returns True if the point was accepted.
        """
        front = self._chunks.get(chunk_idx, [])
        for _, ef in front:
            if _dominates(ef, fitness):
                return False
        front = [(p, f) for p, f in front if not _dominates(fitness, f)]
        front.append((params.copy(), fitness.copy()))
        self._chunks[chunk_idx] = front
        return True

    def get_candidates(self, chunk_idx: int,
                       exclude: Optional[List[np.ndarray]] = None
                       ) -> List[tuple]:
        """Return stored (params, fitness) pairs for chunk_idx.

        Entries whose params exactly match any array in `exclude` are filtered out,
        preventing injection of individuals already present in the current population.
        """
        stored = self._chunks.get(chunk_idx, [])
        if not stored or not exclude:
            return list(stored)
        return [(p, f) for p, f in stored
                if not any(np.array_equal(p, ep) for ep in exclude)]

    def chunk_size(self, chunk_idx: int) -> int:
        return len(self._chunks.get(chunk_idx, []))

    def __len__(self) -> int:
        return sum(len(v) for v in self._chunks.values())


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

        # fitness[i]:     raw reward vector from the latest evaluation
        # ema_fitness[i]: EMA-smoothed fitness used for dominance / selection
        self.fitness     = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.ema_fitness = [np.full(self.M, -np.inf) for _ in range(self.P)]

        self.z_star          = np.full(self.M, -np.inf, dtype=np.float64)
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
    def _crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        alpha = np.random.uniform(0.3, 0.7, size=p1.shape)
        return alpha * p1 + (1.0 - alpha) * p2

    @staticmethod
    def _mutate(x: np.ndarray, sigma: float, rate: float) -> np.ndarray:
        mask = np.random.random(x.shape) < rate
        return x + mask.astype(np.float64) * np.random.normal(0.0, sigma, x.shape)

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
        use_reward_map:    bool  = False,
        use_guided_crossover: bool = False,
        crossover_front_decay: float = 0.5,
        fitness_ema_alpha:    float = 1.0,
        algorithm:         str   = 'nsgaii',
        n_reference_divisions: int = 12,
    ) -> List[np.ndarray]:

        dist_on = torch.distributed.is_initialized()
        rank    = torch.distributed.get_rank() if dist_on else 0
        is_main = rank == 0

        reward_map = RewardMap()  # accumulates Pareto-optimal (params, fitness) across all gens

        # ── Algorithm-specific setup ──────────────────────────────────────────
        assert algorithm in ('nsgaii', 'nsgaiii'), \
            f"algorithm must be 'nsgaii' or 'nsgaiii', got {algorithm!r}"
        if algorithm == 'nsgaiii':
            ref_pts = _generate_reference_points(self.M, n_reference_divisions)
            _select = lambda fit, n: _nsga3_select(fit, n, ref_pts)
            if is_main:
                print(f'NSGA-III: {len(ref_pts)} reference points '
                      f'(M={self.M}, divisions={n_reference_divisions})')
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
        def _done_path(g):       return os.path.join(_gen_dir(g), 'done')
        def _phaseA_done_path(g): return os.path.join(_gen_dir(g), 'phaseA_done')

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
        def _collect_fitness(gen, task_id) -> np.ndarray:
            with open(_result_path(gen, task_id)) as f:
                r = np.array(json.load(f)['reward_vec'])
            self._update_z_star(r)
            return r

        # ── Phase 0: initial population ───────────────────────────────────────
        gen = 0
        if is_main:
            print(f'NSGA-II — initialising population ({self.P} individuals, '
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
                self.fitness[i]     = _collect_fitness(gen, i)
                self.ema_fitness[i] = self.fitness[i].copy()   # gen-0: ema = raw
            if use_reward_map:
                for i in range(self.P):
                    reward_map.update(0, self.population[i], self.fitness[i])
                log(f'gen 0 reward_map[chunk=0]: {reward_map.chunk_size(0)} Pareto entries')
            open(_done_path(gen), 'w').close()
            log(f'gen 0 done, z*={np.round(self.z_star, 3)}')
            print(f'Gen {gen:4d}/{num_generations}')

        else:
            while not os.path.exists(_done_path(gen)):
                time.sleep(poll_interval)

        # ── Generational loop ─────────────────────────────────────────────────
        # Phase A (parent re-eval) is triggered independently by two flags:
        #   fitness_ema_alpha < 1.0  → re-eval to accumulate EMA across gens
        #   use_guided_crossover     → re-eval to get fresh parent fitness for
        #                              front-weighted crossover strategy
        # Phase A task IDs: 0 .. P-1   (parent re-evals)
        # Phase B task IDs: P .. 2P-1  (children)  — only when Phase A is active
        # Single-phase task IDs: 0 .. P-1 (children) — when Phase A is skipped
        #
        # Crossover strategy (Phase B or single-phase):
        #   use_guided_crossover=True  → p1 from F1, p2 weighted by front decay
        #   use_guided_crossover=False → both parents uniform-random from aug pool

        sigma = mutation_sigma
        for gen in range(1, num_generations + 1):
            log(f'gen {gen}/{num_generations} start')

            do_phase_a = use_guided_crossover or (fitness_ema_alpha < 1.0)

            if is_main:
                os.makedirs(_gen_dir(gen), exist_ok=True)
                chunk_idx = gen % _num_chunks
                diversity = np.std(np.array(self.population), axis=0).mean()

            if do_phase_a:
                # ── Phase A: re-evaluate current parents ──────────────────────
                if is_main:
                    for i in range(self.P):
                        _write_task(gen, i, {'task_id': i,
                                             'chunk_idx': chunk_idx,
                                             'child_params': self.population[i].tolist()})
                    log(f'gen {gen} phase-A: {self.P} parent re-eval tasks (chunk={chunk_idx})')

                _worker_loop(gen, 0, self.P, _phaseA_done_path(gen))

                if is_main:
                    for i in range(self.P):
                        raw = _collect_fitness(gen, i)
                        self.fitness[i] = raw
                        if fitness_ema_alpha >= 1.0 or np.all(self.ema_fitness[i] == -np.inf):
                            self.ema_fitness[i] = raw.copy()
                        else:
                            self.ema_fitness[i] = (fitness_ema_alpha * raw
                                                   + (1.0 - fitness_ema_alpha) * self.ema_fitness[i])

                    if use_reward_map:
                        for i in range(self.P):
                            reward_map.update(chunk_idx, self.population[i], self.fitness[i])

            # ── Build augmented parent pool (shared by both crossover strategies) ─
            if is_main:
                stored = (reward_map.get_candidates(chunk_idx, exclude=list(self.population))
                          if use_reward_map else [])
                if stored:
                    log(f'gen {gen} injecting {len(stored)} reward_map candidates '
                        f'(chunk={chunk_idx}, total={len(reward_map)})')

                aug_params  = list(self.population) + [p for p, _ in stored]
                aug_fitness = np.array(list(self.fitness) + [f for _, f in stored])
                aug_size    = len(aug_params)

                # ── Crossover ──────────────────────────────────────────────────
                task_offset = self.P if do_phase_a else 0
                child_params_list = []

                if use_guided_crossover:
                    fronts   = _non_dominated_sort(aug_fitness)
                    front0   = fronts[0]
                    n_fronts = len(fronts)
                    raw_w    = np.array([crossover_front_decay ** k for k in range(n_fronts)])
                    front_w  = raw_w / raw_w.sum()
                    for i in range(self.P):
                        pi1      = front0[np.random.randint(len(front0))]
                        fi_level = int(np.random.choice(n_fronts, p=front_w))
                        pi2      = fronts[fi_level][np.random.randint(len(fronts[fi_level]))]
                        child = self._mutate(self._crossover(aug_params[pi1], aug_params[pi2]),
                                             sigma, mutation_rate)
                        child_params_list.append(child)
                        _write_task(gen, task_offset + i,
                                    {'task_id': task_offset + i, 'chunk_idx': chunk_idx,
                                     'child_params': child.tolist()})
                    log(f'gen {gen} guided crossover: front0={len(front0)}, fronts={n_fronts}')
                else:
                    for i in range(self.P):
                        pi1 = np.random.randint(aug_size)
                        pi2 = np.random.randint(aug_size)
                        child = self._mutate(self._crossover(aug_params[pi1], aug_params[pi2]),
                                             sigma, mutation_rate)
                        child_params_list.append(child)
                        _write_task(gen, task_offset + i,
                                    {'task_id': task_offset + i, 'chunk_idx': chunk_idx,
                                     'child_params': child.tolist()})
                    log(f'gen {gen} uniform crossover: aug={aug_size}, chunk={chunk_idx}')

                # Signal non-main ranks that child tasks are written
                open(_phaseA_done_path(gen), 'w').close()
            else:
                while not os.path.exists(_phaseA_done_path(gen)):
                    time.sleep(poll_interval)

            # task_offset is known to all ranks from do_phase_a (no is_main guard needed)
            task_offset = self.P if do_phase_a else 0
            _worker_loop(gen, task_offset, task_offset + self.P, _done_path(gen))

            if is_main:
                child_fitness = [_collect_fitness(gen, task_offset + i) for i in range(self.P)]

                if use_reward_map:
                    for i in range(self.P):
                        reward_map.update(chunk_idx, child_params_list[i], child_fitness[i])
                    log(f'gen {gen} reward_map[chunk={chunk_idx}]: '
                        f'{reward_map.chunk_size(chunk_idx)} entries, total={len(reward_map)}')

                merged_params  = aug_params + child_params_list
                merged_fitness = np.vstack([aug_fitness, np.array(child_fitness)])

                # EMA for selection: parents accumulated via Phase A (if active),
                # stored candidates use stored fitness, children start with raw.
                aug_ema        = list(self.ema_fitness) + [f.copy() for _, f in stored]
                child_ema_list = [f.copy() for f in child_fitness]
                merged_ema     = np.vstack([np.array(aug_ema), np.array(child_ema_list)])

                selected         = _select(merged_ema, self.P)
                n_parents_kept   = sum(1 for k in selected if k < aug_size)
                self.population  = [merged_params[k]  for k in selected]
                self.fitness     = [merged_fitness[k]  for k in selected]
                self.ema_fitness = [merged_ema[k]      for k in selected]

            # ── Shared post-selection bookkeeping ─────────────────────────────
            if is_main:
                sigma *= sigma_decay

                fit_arr  = np.array(self.fitness)      # raw (for z*, history)
                ema_arr  = np.array(self.ema_fitness)  # smoothed (drives selection/display)
                fronts   = _non_dominated_sort(ema_arr)
                n_front0 = len(fronts[0]) if fronts else 0
                self.fitness_history.append(fit_arr.tolist())
                self.front_history.append(n_front0)

                print(
                    f'Gen {gen:4d}/{num_generations} | '
                    f'front0={n_front0}/{self.P} | '
                    f'parents_kept={n_parents_kept}/{self.P} | '
                    f'mean_fit={np.round(ema_arr.mean(axis=0), 3)} | '
                    f'best_fit={np.round(ema_arr.max(axis=0), 3)} | '
                    f'z*={np.round(self.z_star, 3)} | '
                    f'σ={sigma:.5f} | div={diversity:.5f}',
                    flush=True,
                )

                if gen % save_every == 0:
                    self._save_checkpoint(output_dir, gen)

                open(_done_path(gen), 'w').close()
            else:
                while not os.path.exists(_done_path(gen)):
                    time.sleep(poll_interval)

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

sft_tokenizer              = load_main_tokenizer(script_args.sft_model_name)
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
elif script_args.dataset_name == 'argilla/news-summary':
    # argilla/news-summary: train split has only ~1000 samples; test split has ~20k — use test
    dataset = build_dataset_news_summary_ppo(
        script_args.dataset_name, sft_tokenizer,
        reward_models.rm_tokenizers[0], split='test')
    instructions = Instructions_summary()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}. '
                     f'Choose from: Anthropic/hh-rlhf, openai/summarize_from_feedback, '
                     f'argilla/news-summary')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:     dataset     = dataset.remove_columns(key)

dataset = dataset.with_format("numpy")
data_collator = DataCollatorWithPadding(tokenizer=sft_tokenizer)
print(f'Dataset size: {len(dataset)} | eval_prompts per call: {script_args.eval_prompts}')

_max_new_tokens = (script_args.max_new_tokens if script_args.max_new_tokens > 0
                   else (128 if script_args.dataset_name == 'Anthropic/hh-rlhf' else 48))
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
    base = PeftModel.from_pretrained(base, script_args.sft_model_name).merge_and_unload()
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.resize_token_embeddings(len(sft_tokenizer))
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    expert_models.append(m)
print(f'  All {len(expert_models)} experts loaded on {device}.')

lm_hidden_size = expert_models[0].config.hidden_size
print(f'lm_hidden_size = {lm_hidden_size}')

template_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=len(expert_models),
)

if script_args.warm_start_path:
    from nsgaii_utils import load_gating_network as _load
    loaded = _load(script_args.warm_start_path, lm_hidden_size=lm_hidden_size,
                   num_experts=len(expert_models), device='cpu')
    if loaded is not None:
        template_net = loaded.cpu()
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

print(f'\nStarting NSGA-II — {script_args.num_generations} generations, '
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
    use_reward_map=script_args.use_reward_map,
    use_guided_crossover=script_args.use_guided_crossover,
    crossover_front_decay=script_args.crossover_front_decay,
    fitness_ema_alpha=script_args.fitness_ema_alpha,
    algorithm=script_args.algorithm,
    n_reference_divisions=script_args.n_reference_divisions,
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
