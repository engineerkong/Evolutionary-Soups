"""nsgaii.py — Evolve GatingNetwork parameters using Chunk-Based Incremental NSGA-II.

Algorithm: Chunk-Based Incremental NSGA-II.
  - One pool per generation: pools[g] holds only generation g's P children.
  - Each generation g, ALL active pools are (re-)evaluated on chunk g%N.
  - Cross-gen selection: triggered for EVERY active pool (oldest→newest). Each pool P_k
    is compared against all older active pools AND the meta pool using P_k's full evaluated
    chunk set as mutual reference. Strictly dominated P_k members are eliminated.
    Older pools and meta pool are never eliminated.
  - Intra-gen selection: triggered for EVERY active pool every generation. Keeps only
    non-dominated members (strict Pareto front 1) by that pool's full mean fitness.
  - A pool graduates to Meta after N consecutive chunk evaluations (all N chunks done).
  - Meta pool: after each graduation, _pareto_prune_individuals runs on the full meta pool —
    Pareto filter (remove dominated) then iterative crowding filter (remove most crowded
    member until all have crowding distance ≥ crowding_threshold; 0 = Pareto filter only).
    Crossover donor schedule: gen 0…N-1 use all active pools (no meta yet);
    gen N…2N-2 use meta + active pools with ≥(gen-N+2) chunks evaluated;
    gen ≥ 2N-1 use meta only (active pools too young/noisy to be donors).

Pool lifetime: pool born at gen g is active for gens g … g+N-1, then graduates.
At steady state there are exactly N active pools and N×P tasks per generation.

TO RUN:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./scripts/momoe/nsgaii.py \
    --expert_model_paths './models/ppo/assistant_ppo_harmless_2701/batch_832/' \
                         './models/ppo/assistant_ppo_helpful_2701/batch_832/' \
    --run_name 'nsgaii_gating_0101' 2>&1 | tee ./logs/nsgaii_0101.log
"""

import copy
import datetime
import glob
import time
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    resume_from:          str       = ''   # path to a gen_XXXX checkpoint dir to resume from
    # Algorithm selection
    algorithm:            str       = 'nsgaii'   # 'nsgaii' | 'nsgaiii'
    n_reference_divisions: int      = 12         # Das-Dennis divisions for NSGA-III reference points

    # Evolutionary hyper-parameters
    population_size:      int       = 20
    num_generations:      int       = 100
    mutation_sigma:       float     = 0.05
    mutation_rate:        float     = 0.5
    sigma_decay:          float     = 0.99
    sigma_min:            float     = 0.03
    crowding_threshold:   float     = 0.02 # meta pool crowding filter: iteratively remove most crowded member until all ≥ threshold (0 = disabled)

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
# Individual
# ---------------------------------------------------------------------------

class Individual:
    """Single genome tracked across chunk evaluations."""

    def __init__(self, params: np.ndarray, entry_gen: int):
        self.params        = params.copy()
        self.chunk_fitness: dict = {}   # {chunk_id (int): np.ndarray (M,)}
        self.entry_gen     = entry_gen

    def mean_fitness(self, chunks=None) -> Optional[np.ndarray]:
        keys = (list(self.chunk_fitness) if chunks is None
                else [c for c in chunks if c in self.chunk_fitness])
        if not keys:
            return None
        return np.mean([self.chunk_fitness[c] for c in keys], axis=0)

    @property
    def n_chunks_done(self) -> int:
        return len(self.chunk_fitness)

    def is_complete(self, n_chunks: int) -> bool:
        return len(self.chunk_fitness) >= n_chunks


def _pareto_prune_individuals(individuals: list, crowding_threshold: float = 0.0) -> list:
    """Retain non-dominated Individuals by mean_fitness, then apply crowding filter."""
    non_dom = []
    for ind in individuals:
        f = ind.mean_fitness()
        if f is None:
            continue
        if any(_dominates(o.mean_fitness(), f) for o in non_dom
               if o.mean_fitness() is not None):
            continue
        non_dom = [o for o in non_dom
                   if o.mean_fitness() is None or not _dominates(f, o.mean_fitness())]
        non_dom.append(ind)
    if crowding_threshold > 0:
        while len(non_dom) > 2:
            fits = np.array([ind.mean_fitness() for ind in non_dom])
            dist = _crowding_distance(fits, list(range(len(fits))))
            if dist.min() >= crowding_threshold:
                break
            non_dom.pop(int(np.argmin(dist)))
    return non_dom


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
    """Das-Dennis structured reference points on the unit hyperplane (sum=1, ≥0)."""
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
    """NSGA-III selection: non-dominated sorting + niche-preservation on the last front."""
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

    all_idx    = selected + last_front
    fit_all    = rewards[all_idx]
    ideal      = fit_all.min(axis=0)
    translated = fit_all - ideal

    if selected:
        nadir = (rewards[selected] - ideal).max(axis=0)
    else:
        nadir = translated.max(axis=0)
    nadir      = np.where(nadir < 1e-10, 1.0, nadir)
    normalized = translated / nadir

    R       = reference_points
    r_norms = np.linalg.norm(R, axis=1, keepdims=True)
    r_norms = np.where(r_norms < 1e-10, 1.0, r_norms)
    R_hat   = R / r_norms

    dot      = normalized @ R_hat.T
    proj     = dot[:, :, None] * R_hat[None, :, :]
    diff     = normalized[:, None, :] - proj
    perp_dist = np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))

    assoc_ref  = perp_dist.argmin(axis=1)
    assoc_dist = perp_dist.min(axis=1)

    n_sel       = len(selected)
    niche_count = np.zeros(len(R), dtype=int)
    for j in range(n_sel):
        niche_count[assoc_ref[j]] += 1

    lf_assoc = assoc_ref[n_sel:]
    lf_dist  = assoc_dist[n_sel:]

    remaining        = list(range(len(last_front)))
    chosen_from_last: List[int] = []

    for _ in range(n_needed):
        ref_to_cands: dict = {}
        for loc in remaining:
            ref_to_cands.setdefault(lf_assoc[loc], []).append(loc)
        if not ref_to_cands:
            break
        min_nc    = min(niche_count[r] for r in ref_to_cands)
        min_refs  = [r for r in ref_to_cands if niche_count[r] == min_nc]
        chosen_r  = min_refs[np.random.randint(len(min_refs))]
        cands = ref_to_cands[chosen_r]
        if niche_count[chosen_r] == 0:
            loc = cands[int(np.argmin(lf_dist[cands]))]
        else:
            loc = cands[np.random.randint(len(cands))]
        chosen_from_last.append(last_front[loc])
        niche_count[chosen_r] += 1
        remaining.remove(loc)

    return selected + chosen_from_last


# ---------------------------------------------------------------------------
# Chunk-Based Incremental NSGA-II
# ---------------------------------------------------------------------------

class NSGAII:
    """Chunk-Based Incremental NSGA-II for GatingNetwork evolution.

    Pool layout
    -----------
    pools[g]: List[Individual]  — generation g's children (born at gen g only)
    meta_pool: List[Individual] — graduated individuals, Pareto-filtered

    Per-generation protocol (gen g, chunk = g % N):
      1. Create P new children → pools[g]
      2. Re-evaluate ALL active pools on chunk g%N, update chunk_fitness
      3. For each active pool P_k (oldest→newest):
           Cross-gen: compare P_k against all older active pools + meta pool on mutual
                      chunks (= P_k's full evaluated chunk set); dominated P_k members removed.
           Intra-gen: keep only non-dominated members (front 1) by P_k's full mean fitness.
      4. Graduate pools[g-N+1] → meta pool (Pareto-filtered); remove from active pools.

    At steady state: N active pools, max(N×P) evaluations per generation.
    """

    def __init__(
        self,
        template_net:    GatingNetwork,
        num_objectives:  int,
        population_size: int = 20,
        device:          str = 'cpu',
    ):
        self.template    = template_net.eval()
        self.M           = num_objectives
        self.P           = population_size
        self.device      = device
        self.base_params = net_to_params(template_net)
        self.param_dim   = len(self.base_params)

        self.N           = None   # set in run() once _num_chunks is known
        self.pools: Dict[int, List[Individual]] = {}   # {generation_id: members}
        self.meta_pool:  List[Individual] = []

        self.z_star          = np.full(self.M, -np.inf, dtype=np.float32)
        self.fitness_history = []

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

        for i, ind in enumerate(self.meta_pool):
            subdir = os.path.join(ckpt_dir, 'meta', f'ind_{i:03d}')
            os.makedirs(subdir, exist_ok=True)
            net = params_to_net(ind.params, self.template, 'cpu')
            save_gating_network(net, subdir)
            mf = ind.mean_fitness()
            with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
                json.dump({
                    'mean_fitness':  mf.tolist() if mf is not None else None,
                    'chunk_fitness': {str(k): v.tolist() for k, v in ind.chunk_fitness.items()},
                    'entry_gen':     ind.entry_gen,
                }, f, indent=2)

        for pg in sorted(self.pools):
            for i, ind in enumerate(self.pools[pg]):
                subdir = os.path.join(ckpt_dir, f'pool_g{pg:04d}', f'ind_{i:03d}')
                os.makedirs(subdir, exist_ok=True)
                net = params_to_net(ind.params, self.template, 'cpu')
                save_gating_network(net, subdir)
                mf = ind.mean_fitness()
                with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
                    json.dump({
                        'mean_fitness':  mf.tolist() if mf is not None else None,
                        'chunk_fitness': {str(k): v.tolist() for k, v in ind.chunk_fitness.items()},
                        'entry_gen':     ind.entry_gen,
                    }, f, indent=2)

        state = {
            'generation':  gen,
            'z_star':      self.z_star.tolist(),
            'meta_size':   len(self.meta_pool),
            'active_pools': {str(pg): len(self.pools[pg]) for pg in sorted(self.pools)},
        }
        with open(os.path.join(ckpt_dir, 'nsgaii_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        print(f'  Checkpoint saved → {ckpt_dir}', flush=True)

    def _load_checkpoint(self, ckpt_dir: str, lm_hidden_size: int, num_experts: int) -> int:
        """Restore meta pool, active pools and z_star from a saved checkpoint.
        Returns the checkpoint generation number so run() can start from gen+1."""
        from nsgaii_utils import load_gating_network as _load_net

        with open(os.path.join(ckpt_dir, 'nsgaii_state.json')) as f:
            state = json.load(f)
        self.z_star = np.array(state['z_star'], dtype=np.float32)

        def _load_ind(ind_dir):
            fit_path = os.path.join(ind_dir, 'fitness.json')
            if not os.path.exists(fit_path):
                return None
            with open(fit_path) as f:
                data = json.load(f)
            net = _load_net(ind_dir, lm_hidden_size=lm_hidden_size,
                            num_experts=num_experts, device='cpu')
            if net is None:
                return None
            ind = Individual(net_to_params(net), entry_gen=data.get('entry_gen', 0))
            ind.chunk_fitness = {int(k): np.array(v)
                                 for k, v in data['chunk_fitness'].items()}
            return ind

        self.meta_pool = []
        for ind_dir in sorted(glob.glob(os.path.join(ckpt_dir, 'meta', 'ind_*'))):
            ind = _load_ind(ind_dir)
            if ind is not None:
                self.meta_pool.append(ind)

        self.pools = {}
        for pool_dir in sorted(glob.glob(os.path.join(ckpt_dir, 'pool_g*'))):
            pg = int(os.path.basename(pool_dir).split('_g')[1])
            self.pools[pg] = []
            for ind_dir in sorted(glob.glob(os.path.join(pool_dir, 'ind_*'))):
                ind = _load_ind(ind_dir)
                if ind is not None:
                    self.pools[pg].append(ind)

        pool_str = ' '.join(f'g{pg}:{len(v)}' for pg, v in sorted(self.pools.items()))
        print(f'Resumed from gen {state["generation"]}: '
              f'meta={len(self.meta_pool)}, pools=[{pool_str}]', flush=True)
        return state['generation']

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
        start_gen:         int   = 0,
        mutation_sigma:    float = 0.05,
        mutation_rate:     float = 0.3,
        sigma_decay:       float = 0.97,
        sigma_min:         float = 0.01,
        num_continuations: int   = 1,
        save_every:        int   = 10,
        output_dir:        str   = '.',
        poll_interval:     float = 2.0,
        verbose:           bool  = False,
        seed:              int   = 42,
        crowding_threshold: float = 0.0,
        algorithm:         str   = 'nsgaii',
        n_reference_divisions: int = 12,
    ) -> List[Individual]:

        dist_on = torch.distributed.is_initialized()
        rank    = torch.distributed.get_rank() if dist_on else 0
        is_main = rank == 0

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
        def _done_path(g):        return os.path.join(_gen_dir(g), 'done')
        def _tasks_ready_path(g): return os.path.join(_gen_dir(g), 'tasks_ready')
        def _count_path(g):       return os.path.join(_gen_dir(g), 'task_count.json')

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
        print(f'Dataset chunks: {_num_chunks} × {_chunk_size} prompts '
              f'(N = {_num_chunks}, pool lifetime = {_num_chunks} gens)', flush=True)

        self.N    = _num_chunks
        if start_gen == 0:
            self.pools = {}

        # ── Eval helper ───────────────────────────────────────────────────────
        def _eval_individual(params, chunk_idx, label=''):
            log(f'eval [{label}] chunk={chunk_idx}')
            loader = chunk_loaders[chunk_idx]
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
        def _worker_loop(gen, n_tasks, exit_signal):
            while True:
                if is_main:
                    if all(os.path.exists(_result_path(gen, i)) for i in range(n_tasks)):
                        break
                else:
                    if os.path.exists(exit_signal):
                        break
                claimed = False
                for i in range(n_tasks):
                    if not os.path.exists(_task_path(gen, i)):  continue
                    if os.path.exists(_result_path(gen, i)):    continue
                    if not _try_claim(gen, i):                  continue
                    claimed = True
                    with open(_task_path(gen, i)) as f: task = json.load(f)
                    r = _eval_individual(np.array(task['child_params']),
                                         task['chunk_idx'], label=f'g{gen}/t{i}')
                    _write_result(gen, i, r)
                    break
                if not claimed:
                    time.sleep(poll_interval)

        def _collect_fitness(gen, task_id) -> np.ndarray:
            with open(_result_path(gen, task_id)) as f:
                r = np.array(json.load(f)['reward_vec'])
            self._update_z_star(r)
            return r

        # ── Generational loop ─────────────────────────────────────────────────
        # When resuming, fast-forward sigma to match the checkpoint generation
        sigma = max(sigma_min, mutation_sigma * (sigma_decay ** start_gen))

        for gen in range(start_gen, num_generations + 1):
            log(f'gen {gen}/{num_generations} start')
            chunk_idx = gen % self.N   # current chunk for all pools this generation

            if is_main:
                os.makedirs(_gen_dir(gen), exist_ok=True)

                # ── 1. Create P new children → pools[gen] ────────────────────
                # Donor schedule: before meta forms use all active pools; once meta
                # has members, require active pools to have ≥ required_chunks evaluated
                # (linearly increasing from 2→N over gens N…2N-1), so that by gen 2N-1
                # only meta pool is used.  This removes noisy young-pool donors as
                # meta accumulates reliable graduates.
                if not self.meta_pool:
                    active_donors = [ind for pool in self.pools.values() for ind in pool]
                else:
                    required_chunks = max(1, gen - self.N + 2)
                    active_donors   = [
                        ind for pool in self.pools.values()
                        if pool and len(pool[0].chunk_fitness) >= required_chunks
                        for ind in pool
                    ]
                all_inds     = active_donors + self.meta_pool
                donor_params = [ind.params for ind in all_inds] if all_inds else [self.base_params]
                donor_size   = len(donor_params)

                new_children: List[Individual] = []
                for _ in range(self.P):
                    pi1 = np.random.randint(donor_size)
                    pi2 = np.random.randint(donor_size)
                    child_p, _ = self._crossover(donor_params[pi1], donor_params[pi2])
                    child_p    = self._mutate(child_p, sigma, mutation_rate)
                    new_children.append(Individual(child_p, entry_gen=gen))
                self.pools[gen] = new_children

                # ── 2. Build flat task list: ALL active pools on chunk_idx ───
                # Ordered oldest → newest so task IDs are stable across re-runs.
                ordered_pool_gens = sorted(self.pools.keys())
                to_eval: List[Individual] = []
                for pg in ordered_pool_gens:
                    to_eval.extend(self.pools[pg])
                n_tasks = len(to_eval)
                n_hist  = n_tasks - self.P   # individuals from older pools

                log(f'gen {gen}: chunk={chunk_idx} tasks={n_tasks} '
                    f'({self.P} new + {n_hist} hist from {len(ordered_pool_gens)-1} older pools)')

                for i, ind in enumerate(to_eval):
                    _write_task(gen, i, {'task_id': i, 'chunk_idx': chunk_idx,
                                          'child_params': ind.params.tolist()})
                with open(_count_path(gen), 'w') as f:
                    json.dump({'n_tasks': n_tasks}, f)
                open(_tasks_ready_path(gen), 'w').close()

            else:
                while not os.path.exists(_tasks_ready_path(gen)):
                    time.sleep(poll_interval)
                with open(_count_path(gen)) as f:
                    n_tasks = json.load(f)['n_tasks']

            _worker_loop(gen, n_tasks, _done_path(gen))

            if is_main:
                # ── 3. Collect results → update chunk_fitness ─────────────────
                for i, ind in enumerate(to_eval):
                    raw = _collect_fitness(gen, i)
                    ind.chunk_fitness[chunk_idx] = raw   # record for this individual's gen

                # ── 4. Cross-gen + Intra-gen selection ───────────────────────
                # Both triggered for EVERY active pool at the current chunk.
                #
                # Cross-gen (for each pool P_k, oldest→newest):
                #   Mutual chunks = P_k's full evaluated chunk set.  Older pools have
                #   been active longer and include all of P_k's chunks, so P_k's chunk
                #   set is exactly the intersection.  Both sides use mean_fitness over
                #   these mutual chunks.  Dominated members of P_k are eliminated;
                #   older pools are never touched.
                #
                # Intra-gen (for each pool P_k):
                #   NSGA-II select by that pool's full mean fitness.
                #   For the graduating pool this serves as the final quality filter
                #   using all N chunks before members enter meta.
                ordered_pool_gens = sorted(self.pools.keys())  # oldest → newest
                for k_idx, pg in enumerate(ordered_pool_gens):
                    pool_k = self.pools[pg]
                    if not pool_k:
                        continue

                    # Cross-gen: compare pool_k against all older active pools + meta pool
                    # Meta pool members have all N chunks so always cover mutual_chunks.
                    older_inds = ([ind for opg in ordered_pool_gens[:k_idx]
                                   for ind in self.pools[opg]]
                                  + self.meta_pool)
                    if older_inds:
                        # mutual_chunks = pool_k's own evaluated chunk set;
                        # all members of pool_k were born at the same gen and have
                        # identical chunk_fitness keys, so we read from pool_k[0].
                        mutual_chunks = list(pool_k[0].chunk_fitness.keys())
                        self.pools[pg] = [
                            c for c in pool_k
                            if (c.mean_fitness(chunks=mutual_chunks) is not None and
                                not any(
                                    _dominates(o.mean_fitness(chunks=mutual_chunks),
                                               c.mean_fitness(chunks=mutual_chunks))
                                    for o in older_inds
                                    if o.mean_fitness(chunks=mutual_chunks) is not None))
                        ]
                        pool_k = self.pools[pg]

                    # Intra-gen: keep non-dominated members by full mean fitness
                    if len(pool_k) > 1:
                        self.pools[pg] = [
                            ind for ind in pool_k
                            if (ind.mean_fitness() is not None and
                                not any(_dominates(o.mean_fitness(), ind.mean_fitness())
                                        for o in pool_k
                                        if o is not ind and o.mean_fitness() is not None))
                        ]

                n_survived_new = len(self.pools.get(gen, []))

                # ── 5. Graduate oldest pool → meta ────────────────────────────
                graduate_gen = gen - self.N + 1   # pool born this many gens ago
                graduates: List[Individual] = []
                if graduate_gen in self.pools:
                    graduates = list(self.pools.pop(graduate_gen))
                    self.meta_pool.extend(graduates)
                    if graduates:
                        self.meta_pool = _pareto_prune_individuals(
                            self.meta_pool, crowding_threshold)

                # ── 6. Bookkeeping ────────────────────────────────────────────
                sigma = max(sigma_min, sigma * sigma_decay)

                self.fitness_history.append(
                    [ind.mean_fitness().tolist() for ind in self.meta_pool]
                    if self.meta_pool else []
                )

                all_params_list = ([ind.params for pool in self.pools.values() for ind in pool]
                                   + [ind.params for ind in self.meta_pool])
                diversity = (np.std(np.array(all_params_list), axis=0).mean()
                             if len(all_params_list) > 1 else 0.0)

                pool_str  = ' '.join(f'g{pg}:{len(self.pools[pg])}'
                                     for pg in sorted(self.pools))
                meta_best = (np.round(
                                 np.array([ind.mean_fitness() for ind in self.meta_pool]).max(axis=0), 3)
                             if self.meta_pool else 'n/a')
                print(
                    f'Gen {gen:4d}/{num_generations} | chunk={chunk_idx} | '
                    f'pools=[{pool_str}] | meta={len(self.meta_pool)} | '
                    f'surv_new={n_survived_new}/{self.P} | grad={len(graduates)} | '
                    f'meta_best={meta_best} | '
                    f'z*={np.round(self.z_star, 3)} | σ={sigma:.5f} | div={diversity:.5f}',
                    flush=True,
                )

                log_path = os.path.join(output_dir, 'population_log.json')
                try:
                    with open(log_path) as f: pop_log = json.load(f)
                except FileNotFoundError:
                    pop_log = {}
                pop_log[f'gen_{gen:04d}'] = {
                    'chunk':         chunk_idx,
                    'n_tasks':       n_tasks,
                    'n_new':         self.P,
                    'n_hist':        n_hist,
                    'surv_new':      n_survived_new,
                    'graduates':     len(graduates),
                    'graduate_pool': graduate_gen if graduates else None,
                    'meta_size':     len(self.meta_pool),
                    'active_pools':  {str(pg): len(self.pools[pg])
                                      for pg in sorted(self.pools)},
                }
                with open(log_path, 'w') as f:
                    json.dump(pop_log, f, indent=2)

                if gen % save_every == 0:
                    self._save_checkpoint(output_dir, gen)

                open(_done_path(gen), 'w').close()

            else:
                while not os.path.exists(_done_path(gen)):
                    time.sleep(poll_interval)

        return self.meta_pool


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
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

dataset = dataset.with_format("numpy")
data_collator = DataCollatorWithPadding(tokenizer=sft_tokenizer)
print(f'Dataset size: {len(dataset)} | eval_prompts per call: {script_args.eval_prompts}')

_max_new_tokens = (script_args.max_new_tokens if script_args.max_new_tokens > 0
                   else (128 if script_args.dataset_name in {
                       'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K',
                       'nvidia/HelpSteer', 'nvidia/HelpSteer2'} else 48))
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
print(f'lm_hidden_size = {lm_hidden_size}')

template_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=len(expert_models),
)

moe_model = MoEForCausalLM(expert_models, template_net).to(device)
moe_model.eval()
print(f'MoEForCausalLM: {len(expert_models)} experts × {moe_model.num_layers} layers')

nsgaii = NSGAII(
    template_net=template_net,
    num_objectives=n_objectives,
    population_size=script_args.population_size,
    device=device,
)

start_gen = 0
if script_args.resume_from:
    start_gen = nsgaii._load_checkpoint(
        script_args.resume_from,
        lm_hidden_size=lm_hidden_size,
        num_experts=len(expert_models),
    ) + 1
    print(f'Resuming from gen {start_gen} (checkpoint: {script_args.resume_from})')

print(f'\nStarting Chunk-Based Incremental NSGA-II — {script_args.num_generations} generations, '
      f'P={nsgaii.P} new children/gen, N={script_args.eval_prompts}-prompt chunks …')
nsgaii.run(
    dataset=dataset, data_collator=data_collator,
    eval_prompts=script_args.eval_prompts, eval_batch_size=script_args.eval_batch_size,
    moe_model=moe_model, sft_tokenizer=sft_tokenizer,
    reward_models=reward_models, instructions=instructions,
    generation_kwargs=generation_kwargs, gpu_id=gpu_id,
    num_generations=script_args.num_generations,
    start_gen=start_gen,
    mutation_sigma=script_args.mutation_sigma, mutation_rate=script_args.mutation_rate,
    sigma_decay=script_args.sigma_decay, sigma_min=script_args.sigma_min,
    num_continuations=script_args.num_continuations,
    save_every=script_args.save_every, output_dir=output_dir,
    verbose=script_args.verbose, seed=script_args.seed,
    crowding_threshold=script_args.crowding_threshold,
    algorithm=script_args.algorithm,
    n_reference_divisions=script_args.n_reference_divisions,
)

is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
if is_main:
    print(f'\nSaving final meta pool to {output_dir}/final …')
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    for i, ind in enumerate(nsgaii.meta_pool):
        subdir = os.path.join(final_dir, f'ind_{i:03d}')
        net    = params_to_net(ind.params, template_net, 'cpu')
        save_gating_network(net, subdir)
        mf = ind.mean_fitness()
        with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
            json.dump({
                'mean_fitness':  mf.tolist() if mf is not None else None,
                'chunk_fitness': {str(k): v.tolist() for k, v in ind.chunk_fitness.items()},
                'entry_gen':     ind.entry_gen,
            }, f, indent=2)

    with open(os.path.join(output_dir, 'nsgaii_meta.json'), 'w') as f:
        json.dump({
            'reward_names':    reward_names,
            'z_star':          nsgaii.z_star.tolist(),
            'final_fitness':   [ind.mean_fitness().tolist() for ind in nsgaii.meta_pool
                                if ind.mean_fitness() is not None],
            'fitness_history': nsgaii.fitness_history,
            'num_generations': script_args.num_generations,
            'population_size': nsgaii.P,
            'lm_hidden_size':  lm_hidden_size,
        }, f, indent=2)

    print(f'\nDone. Meta pool: {len(nsgaii.meta_pool)} individuals | '
          f'z*={np.round(nsgaii.z_star, 4)}')
