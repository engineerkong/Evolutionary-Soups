"""_retention.py — ES evolution with dual-front retention selection.

Mirrors es_train.py, with one additional selection mode:

  - When `use_dual_front=True`, selection combines two fronts:
      • front1 = non-dominated front over raw merged fitness (parents+children)
      • front2 = non-dominated front over a baseline-stability-boosted set:
          parent_baseline · (1 + min(eval_count · bonus, cap))   ⨃  child_fitness
    The intersection front1 ∩ front2 is filled first via greedy HVC (carries
    over individuals that look good both on this generation AND on average),
    then front2 \\ front1, then deeper front2 fronts — until P individuals
    are chosen. This biases retention toward parents whose averaged fitness
    holds up across re-evaluations.

  - When `use_dual_front=False`, behavior is identical to es_train.py.

`parent_baseline` is the cumulative mean of a parent's raw reward vector
across all re-evaluations; `parent_eval_count` tracks how many times each
parent has been evaluated. Both are checkpointed and restored on resume.

Supported datasets: Anthropic/hh-rlhf, openai/summarize_from_feedback,
PKU-Alignment/PKU-SafeRLHF-10K.
"""

import datetime
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

from es_architecture import GatingNetwork, MoEForCausalLM, SimpleGatingNetwork, SimpleMoEForCausalLM, load_shared_experts
from es_utils import (
    Instructions, Instructions_summary, REWARD_PATHS, RewardModels,
    build_dataset_ppo, build_dataset_summary_ppo, build_dataset_beaver_ppo,
    generate_and_score, generate_reference_points, greedy_hvc_select, hv,
    load_gating_network, load_main_tokenizer, load_simple_gating_network,
    make_onehot_params, net_to_params, non_dominated_sort, nsga2_select,
    nsga3_select, params_to_net, save_gating_network,
)


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:      str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths:   List[str] = field(default_factory=list)
    reward_names:         str       = 'harmless,helpful'
    dataset_name:         str       = 'Anthropic/hh-rlhf'
    do_sample:            bool      = False
    num_continuations:    int       = 1
    eval_prompts:         int       = 8192
    eval_batch_size:      int       = 128
    max_new_tokens:       int       = -1
    normalize_rewards:    bool      = False  # online z-score fitness via running mean/std (computed during the run)
    warm_start_path:      str       = ''

    # Algorithm selection
    algorithm:            str       = 'greedy_hvc'   # 'nsgaii' | 'nsgaiii' | 'greedy_hvc'
    n_reference_divisions: int      = 12
    use_greedy_hvc:       bool      = True

    # Evolutionary hyper-parameters
    population_size:      int       = 20
    num_generations:      int       = 100
    convergence_window:   int       = 10
    mutation_sigma:       float     = 0.05
    mutation_rate:        float     = 0.5
    sigma_decay:          float     = 0.99
    sigma_min:            float     = 0.03

    # Dual-front retention
    use_dual_front:         bool    = True
    parent_stability_bonus: float   = 0.01
    parent_stability_cap:   float   = 0.10

    # Gating template
    fixed_alpha:          float     = 1.2
    gating_type:          str       = 'per_layer'   # 'per_layer' | 'simple'
    normalize_fitness:    bool      = True

    # GPU / I/O
    gpu_id:               int       = -1
    save_directory:       str       = './models/ES/'
    run_name:             str       = 'es_retention'
    save_every:           int       = 5
    seed:                 int       = 8888
    verbose:              bool      = False


# ---------------------------------------------------------------------------
# ES evolutionary algorithm class
# ---------------------------------------------------------------------------

class ES:
    """ES with dual-front retention selection (es_train.ES + parent_baseline tracking)."""

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

        self.fitness           = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.parent_baseline   = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.parent_eval_count = [0] * self.P
        self.z_star            = np.full(self.M, -np.inf, dtype=np.float32)

        # Online (Welford) running mean/std per objective for normalize_rewards.
        # Computed DURING the run from observed reward vectors — not given upfront.
        self._rew_count = 0
        self._rew_mean  = np.zeros(self.M, dtype=np.float64)
        self._rew_M2    = np.zeros(self.M, dtype=np.float64)

    def _update_z_star(self, r: np.ndarray):
        improved = r > self.z_star
        self.z_star[improved] = r[improved]

    def _running_zscore(self, r: np.ndarray) -> np.ndarray:
        """Welford online update of per-objective mean/std, then z-score `r`.

        Stats accumulate across every reward vector seen this run (a PPO-style
        running normalizer), so the normalization baseline shifts as more data
        arrives. Until variance is defined (count < 2), only centering applies.
        """
        self._rew_count += 1
        delta = r - self._rew_mean
        self._rew_mean += delta / self._rew_count
        self._rew_M2   += delta * (r - self._rew_mean)
        if self._rew_count < 2:
            return r - self._rew_mean
        std = np.sqrt(self._rew_M2 / (self._rew_count - 1))
        std = np.where(std < 1e-6, 1.0, std)
        return (r - self._rew_mean) / std

    def _reward_stats_dict(self) -> dict:
        std = (np.sqrt(self._rew_M2 / (self._rew_count - 1))
               if self._rew_count >= 2 else np.ones(self.M))
        return {
            'count': self._rew_count,
            'mean':  self._rew_mean.tolist(),
            'M2':    self._rew_M2.tolist(),
            'std':   np.where(std < 1e-6, 1.0, std).tolist(),
        }

    def _restore_reward_stats(self, rs: dict) -> None:
        if not rs:
            return
        self._rew_count = int(rs.get('count', 0))
        self._rew_mean  = np.array(rs.get('mean', np.zeros(self.M)), dtype=np.float64)
        self._rew_M2    = np.array(rs.get('M2',   np.zeros(self.M)), dtype=np.float64)

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

    def _save_checkpoint(self, output_dir: str, gen: int, sigma: float,
                         bounds: dict = None):
        ckpt_dir = os.path.join(output_dir, f'gen_{gen:04d}')
        os.makedirs(ckpt_dir, exist_ok=True)
        for i in range(self.P):
            subdir = os.path.join(ckpt_dir, f'ind_{i:03d}')
            net    = params_to_net(self.population[i], self.template, 'cpu')
            save_gating_network(net, subdir)
            with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
                json.dump({'fitness': self.fitness[i].tolist()}, f, indent=2)
        meta = {
            'generation':        gen,
            'sigma':             sigma,
            'z_star':            self.z_star.tolist(),
            'fitness':           [f.tolist() for f in self.fitness],
            'parent_baseline':   [b.tolist() for b in self.parent_baseline],
            'parent_eval_count': list(self.parent_eval_count),
            'reward_stats':      self._reward_stats_dict(),
        }
        if bounds and bounds.get('min') is not None:
            meta['bounds'] = {
                'min':   bounds['min'].tolist(),
                'range': bounds['range'].tolist(),
            }
        with open(os.path.join(ckpt_dir, 'es_state.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  Checkpoint saved → {ckpt_dir}', flush=True)

    @staticmethod
    def _find_latest_checkpoint(output_dir: str):
        """Return (gen, ckpt_dir) of the latest gen_XXXX checkpoint, or (-1, None)."""
        gen_dirs = sorted([
            d for d in os.listdir(output_dir)
            if d.startswith('gen_') and d[4:].isdigit()
            and os.path.isfile(os.path.join(output_dir, d, 'es_state.json'))
        ])
        if not gen_dirs:
            return -1, None
        latest = gen_dirs[-1]
        return int(latest[4:]), os.path.join(output_dir, latest)

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
        use_greedy_hvc:          bool  = True,
        algorithm:               str   = 'greedy_hvc',
        n_reference_divisions:   int   = 12,
        normalize_fitness:       bool  = False,
        normalize_rewards:       bool  = False,
        use_dual_front:          bool  = True,
        parent_stability_bonus:  float = 0.01,
        parent_stability_cap:    float = 0.10,
        resume:                  bool  = False,
        resume_dir:              str   = None,
    ) -> List[np.ndarray]:

        dist_on = torch.distributed.is_initialized()
        rank    = torch.distributed.get_rank() if dist_on else 0
        is_main = rank == 0

        # normalize_rewards (online z-score) and normalize_fitness (expert min-max)
        # are alternative fitness-normalization schemes; enabling both would
        # compound them in inconsistent spaces.
        assert not (normalize_rewards and normalize_fitness), \
            'normalize_rewards and normalize_fitness are alternative schemes — enable at most one.'

        # ── Algorithm-specific setup ──────────────────────────────────────────
        assert algorithm in ('nsgaii', 'nsgaiii', 'greedy_hvc'), \
            f"algorithm must be 'nsgaii', 'nsgaiii', or 'greedy_hvc', got {algorithm!r}"
        if algorithm == 'nsgaiii':
            ref_pts = generate_reference_points(self.M, n_reference_divisions)
            _select = lambda fit, n: nsga3_select(fit, n, ref_pts)
            if is_main:
                print(f'NSGA-III: {len(ref_pts)} reference points '
                      f'(M={self.M}, divisions={n_reference_divisions})')
        elif algorithm == 'greedy_hvc':
            _select = greedy_hvc_select
            if is_main:
                print(f'greedy_hvc: Hypervolume Computing utility selection (M={self.M})')
        else:
            _select = nsga2_select

        if use_greedy_hvc:
            _select = greedy_hvc_select

        if is_main and use_dual_front:
            print(f'Dual-front retention: bonus={parent_stability_bonus}, '
                  f'cap={parent_stability_cap}')

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

        # ── Resume: load latest checkpoint ────────────────────────────────────
        resume_gen = -1
        sigma      = mutation_sigma
        bounds     = {'min': None, 'range': None}

        if resume and is_main:
            resume_gen, ckpt_dir = self._find_latest_checkpoint(resume_dir or output_dir)
            if resume_gen >= 0:
                with open(os.path.join(ckpt_dir, 'es_state.json')) as f:
                    state = json.load(f)

                for i in range(self.P):
                    ind_dir = os.path.join(ckpt_dir, f'ind_{i:03d}')
                    g = load_gating_network(ind_dir, lm_hidden_size=self.template.lm_hidden_size,
                                            num_experts=self.template.num_experts,
                                            num_layers=getattr(self.template, 'num_layers', 32),
                                            device='cpu')
                    if g is None:
                        g = load_simple_gating_network(ind_dir,
                                                       lm_hidden_size=self.template.lm_hidden_size,
                                                       num_experts=self.template.num_experts,
                                                       device='cpu')
                    self.population[i] = net_to_params(g)
                    self.fitness[i]    = np.array(state['fitness'][i])

                self.z_star = np.array(state['z_star'])
                sigma       = state.get('sigma', mutation_sigma)
                if 'parent_baseline' in state:
                    self.parent_baseline = [np.array(b) for b in state['parent_baseline']]
                if 'parent_eval_count' in state:
                    self.parent_eval_count = list(state['parent_eval_count'])

                lb = state.get('bounds')
                if not lb:
                    bounds_path = os.path.join(resume_dir or output_dir, 'bounds.json')
                    if os.path.exists(bounds_path):
                        with open(bounds_path) as f:
                            lb = json.load(f)
                        print(f'  bounds restored from bounds.json', flush=True)
                if lb:
                    bounds['min']   = np.array(lb['min'])
                    bounds['range'] = np.array(lb['range'])
                    r_max = bounds['min'] + bounds['range']
                    print(f'  reward_min={np.round(bounds["min"], 3)}', flush=True)
                    print(f'  reward_max={np.round(r_max, 3)}', flush=True)

                # Restore online reward-normalizer stats (es_state, then reward_stats.json)
                rs = state.get('reward_stats')
                if not rs:
                    rs_path = os.path.join(resume_dir or output_dir, 'reward_stats.json')
                    if os.path.exists(rs_path):
                        with open(rs_path) as f:
                            rs = json.load(f)
                self._restore_reward_stats(rs)
                if normalize_rewards and self._rew_count > 0:
                    print(f'  reward_stats restored (count={self._rew_count}, '
                          f'mean={np.round(self._rew_mean, 3)})', flush=True)

                print(f'Resuming: next gen={resume_gen + 1}  '
                      f'(σ={sigma:.5f}, z*={np.round(self.z_star, 3)})', flush=True)
            else:
                print('No checkpoint found — starting from scratch.', flush=True)
                resume_gen = -1

        if resume and not is_main:
            resume_gen, _ckpt_dir = self._find_latest_checkpoint(resume_dir or output_dir)
            if normalize_fitness and _ckpt_dir is not None:
                with open(os.path.join(_ckpt_dir, 'es_state.json')) as f:
                    st = json.load(f)
                lb = st.get('bounds')
                if lb is None:
                    bounds_path = os.path.join(resume_dir or output_dir, 'bounds.json')
                    if os.path.exists(bounds_path):
                        with open(bounds_path) as f:
                            lb = json.load(f)
                if lb is not None:
                    bounds['min']   = np.array(lb['min'])
                    bounds['range'] = np.array(lb['range'])

        # ── Eval helper ───────────────────────────────────────────────────────
        def _eval_individual(params, chunk_idx, label=''):
            log(f'eval [{label}] chunk={chunk_idx % _num_chunks}')
            loader = chunk_loaders[chunk_idx % _num_chunks]
            net    = params_to_net(params, self.template, self.device)
            net.eval()
            moe_model.gating_net = net
            reward_vecs = []
            for batch in loader:
                # Workers always produce RAW reward vectors; the main rank applies
                # the online z-score (normalize_rewards) at collection time.
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

        def _collect_fitness(gen, task_id) -> np.ndarray:
            with open(_result_path(gen, task_id)) as f:
                r = np.array(json.load(f)['reward_vec'])      # raw reward vector
            if normalize_rewards:
                r = self._running_zscore(r)                   # online z-score (running mean/std)
            elif bounds['range'] is not None:
                r = (r - bounds['min']) / bounds['range']     # expert min-max
            self._update_z_star(r)
            return r

        # ── Expert reward bounds (normalize_fitness=True) ─────────────────────
        _BG = -1
        if normalize_fitness and bounds['min'] is not None:
            if is_main:
                os.makedirs(_gen_dir(_BG), exist_ok=True)
                open(_done_path(_BG), 'w').close()
            else:
                while not os.path.exists(_done_path(_BG)):
                    time.sleep(poll_interval)

        if normalize_fitness and bounds['min'] is None:
            if is_main:
                print(f'Computing expert reward bounds '
                      f'({self.template.num_experts} experts, chunk 0) …', flush=True)
                os.makedirs(_gen_dir(_BG), exist_ok=True)
                for exp_idx in range(self.template.num_experts):
                    _write_task(_BG, exp_idx, {
                        'task_id':      exp_idx,
                        'chunk_idx':    0,
                        'child_params': make_onehot_params(self.template, exp_idx).tolist(),
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
                with open(os.path.join(output_dir, 'bounds.json'), 'w') as f:
                    json.dump({'min': r_min.tolist(), 'range': r_range.tolist()}, f)
                open(_done_path(_BG), 'w').close()
            else:
                while not os.path.exists(_done_path(_BG)):
                    time.sleep(poll_interval)

        # ── Phase 0: initial population (skipped on resume) ──────────────────
        if resume_gen < 0:
            gen = 0
            if is_main:
                print(f'ES (dual-front) — initialising population ({self.P} individuals) …')
                os.makedirs(_gen_dir(gen), exist_ok=True)
                for i in range(self.P):
                    _write_task(gen, i, {'task_id': i,
                                         'chunk_idx': 0,
                                         'child_params': self.population[i].tolist()})

            _worker_loop(gen, 0, self.P, _done_path(gen))

            if is_main:
                for i in range(self.P):
                    self.fitness[i]           = _collect_fitness(gen, i)
                    self.parent_baseline[i]   = self.fitness[i].copy()
                    self.parent_eval_count[i] = 1
                open(_done_path(gen), 'w').close()
                log(f'gen 0 done, z*={np.round(self.z_star, 3)}')
                print(f'Gen {gen:4d}/{num_generations}')
            else:
                while not os.path.exists(_done_path(gen)):
                    time.sleep(poll_interval)

        # ── Generational loop ─────────────────────────────────────────────────
        start_gen = resume_gen + 1 if resume_gen >= 0 else 1
        for gen in range(start_gen, num_generations + 1):
            log(f'gen {gen}/{num_generations} start')

            if is_main:
                os.makedirs(_gen_dir(gen), exist_ok=True)
                chunk_idx = gen % _num_chunks
                diversity = np.std(np.array(self.population), axis=0).mean()

                parent_params = list(self.population)

                # Parent re-eval tasks (0..P-1)
                for i in range(self.P):
                    _write_task(gen, i, {'task_id': i, 'chunk_idx': chunk_idx,
                                         'child_params': self.population[i].tolist()})

                # Crossover → child tasks (P..2P-1)
                child_params_list = []
                for i in range(self.P):
                    pi1 = np.random.randint(self.P)
                    pi2 = np.random.randint(self.P)
                    child, _ = self._crossover(parent_params[pi1], parent_params[pi2])
                    child = self._mutate(child, sigma, mutation_rate)
                    child_params_list.append(child)
                    _write_task(gen, self.P + i, {'task_id': self.P + i,
                                                   'chunk_idx': chunk_idx,
                                                   'child_params': child.tolist()})
                log(f'gen {gen} crossover: parents={self.P}, chunk={chunk_idx}')

                open(_tasks_ready_path(gen), 'w').close()
            else:
                while not os.path.exists(_tasks_ready_path(gen)):
                    time.sleep(poll_interval)

            _worker_loop(gen, 0, 2 * self.P, _done_path(gen))

            n_intersection = 0
            if is_main:
                # Parent re-evals: update both raw fitness AND running baseline
                for i in range(self.P):
                    raw = _collect_fitness(gen, i)
                    self.fitness[i] = raw
                    n = self.parent_eval_count[i]
                    if n == 0 or np.all(self.parent_baseline[i] == -np.inf):
                        self.parent_baseline[i] = raw.copy()
                    else:
                        self.parent_baseline[i] = (self.parent_baseline[i] * n + raw) / (n + 1)
                    self.parent_eval_count[i] = n + 1

                child_fitness  = [_collect_fitness(gen, self.P + i) for i in range(self.P)]
                merged_params  = parent_params + child_params_list
                merged_fitness = np.vstack([np.array(self.fitness), np.array(child_fitness)])

                # Greedy HVC fill over `merged_fitness` from a seed set.
                def _greedy_hvc_fill(seed: List[int], pool: List[int], n: int) -> List[int]:
                    if n <= 0 or not pool:
                        return []
                    all_pts = np.array([merged_fitness[k] for k in seed + pool])
                    ref     = all_pts.min(axis=0) - 0.1
                    chosen  = [merged_fitness[k] for k in seed]
                    hv_base = hv(np.array(chosen), ref) if chosen else 0.0
                    result, remaining = [], list(range(len(pool)))
                    for _ in range(min(n, len(remaining))):
                        gains    = [hv(np.array(chosen + [merged_fitness[pool[loc]]]), ref) - hv_base
                                    for loc in remaining]
                        best_pos = int(np.argmax(gains))
                        best_loc = remaining[best_pos]
                        hv_base += gains[best_pos]
                        chosen.append(merged_fitness[pool[best_loc]])
                        result.append(pool[best_loc])
                        remaining.remove(best_loc)
                    return result

                if not use_dual_front:
                    if use_greedy_hvc:
                        selected = _greedy_hvc_fill([], list(range(len(merged_fitness))), self.P)
                    else:
                        selected = _select(merged_fitness, self.P)
                else:
                    # ── Dual-front selection ──────────────────────────────────
                    front1_set = set(non_dominated_sort(merged_fitness)[0])

                    parent_front2 = [
                        self.parent_baseline[i] * (1.0 + min(
                            self.parent_eval_count[i] * parent_stability_bonus,
                            parent_stability_cap))
                        for i in range(self.P)
                    ]
                    merged_front2 = np.vstack([np.array(parent_front2), np.array(child_fitness)])
                    front2_fronts = non_dominated_sort(merged_front2)
                    front2_set    = set(front2_fronts[0])

                    intersection   = [k for k in front2_set if k in front1_set]
                    n_intersection = len(intersection)
                    if len(intersection) > self.P:
                        intersection = _greedy_hvc_fill([], intersection, self.P)
                    selected = intersection + _greedy_hvc_fill(
                        intersection,
                        [k for k in front2_set if k not in front1_set],
                        self.P - len(intersection),
                    )
                    for deeper_front in front2_fronts[1:]:
                        if len(selected) >= self.P: break
                        pool = [k for k in deeper_front if k not in set(selected)]
                        selected += _greedy_hvc_fill(selected, pool, self.P - len(selected))

                ind_labels     = [f'{j+1}-parent{k+1}' if k < self.P else f'{j+1}-child'
                                  for j, k in enumerate(selected)]
                n_parents_kept = sum(1 for k in selected if k < self.P)

                # Carry baseline + eval_count through selection
                new_baseline, new_eval_count = [], []
                for k in selected:
                    if k < self.P:
                        new_baseline.append(self.parent_baseline[k].copy())
                        new_eval_count.append(self.parent_eval_count[k])
                    else:
                        new_baseline.append(merged_fitness[k].copy())
                        new_eval_count.append(1)

                self.population        = [merged_params[k] for k in selected]
                self.fitness           = [merged_fitness[k] for k in selected]
                self.parent_baseline   = new_baseline
                self.parent_eval_count = new_eval_count

            # ── Post-selection bookkeeping and logging ────────────────────────
            if is_main:
                sigma = max(sigma_min, sigma * sigma_decay)
                fit_arr  = np.array(self.fitness)
                base_arr = np.array(self.parent_baseline)

                log_path = os.path.join(output_dir, 'population_log.json')
                try:
                    with open(log_path) as f:
                        pop_log = json.load(f)
                except FileNotFoundError:
                    pop_log = {}
                pop_log[f'gen_{gen:04d}'] = {
                    label: {'raw': fit_arr[j].tolist(),
                            'baseline': base_arr[j].tolist()}
                    for j, label in enumerate(ind_labels)
                }
                with open(log_path, 'w') as f:
                    json.dump(pop_log, f, indent=2)

                live_state: dict = {
                    'generation': gen,
                    'sigma':      sigma,
                    'z_star':     self.z_star.tolist(),
                }
                if bounds.get('min') is not None:
                    live_state['bounds'] = {
                        'min':   bounds['min'].tolist(),
                        'range': bounds['range'].tolist(),
                    }
                live_path = os.path.join(output_dir, 'live_state.json')
                tmp_live  = live_path + '.tmp'
                with open(tmp_live, 'w') as f:
                    json.dump(live_state, f)
                os.replace(tmp_live, live_path)

                # Online reward-normalizer stats every generation (crash-safe resume)
                if normalize_rewards:
                    rs_path = os.path.join(output_dir, 'reward_stats.json')
                    tmp_rs  = rs_path + '.tmp'
                    with open(tmp_rs, 'w') as f:
                        json.dump(self._reward_stats_dict(), f)
                    os.replace(tmp_rs, rs_path)

                alpha_str = f'entmax_α={self.template.fixed_alpha:.3f}(fixed)'
                front_str = f'intersect={n_intersection}/{self.P} | ' if use_dual_front else ''
                print(
                    f'Gen {gen:4d}/{num_generations} | '
                    f'chunk={chunk_idx % _num_chunks} | '
                    f'parents_kept={n_parents_kept}/{self.P} | '
                    f'{front_str}'
                    f'mean_fit={np.round(fit_arr.mean(axis=0), 3)} | '
                    f'best_fit={np.round(fit_arr.max(axis=0), 3)} | '
                    f'z*={np.round(self.z_star, 3)} | '
                    f'σ={sigma:.5f} | '
                    f'{alpha_str} | '
                    f'div={diversity:.5f}',
                    flush=True,
                )

                if gen % save_every == 0:
                    self._save_checkpoint(output_dir, gen, sigma, bounds)

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
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}. '
                     f'Choose from: Anthropic/hh-rlhf, openai/summarize_from_feedback, '
                     f'PKU-Alignment/PKU-SafeRLHF-10K')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

dataset = dataset.with_format("numpy")
data_collator = DataCollatorWithPadding(tokenizer=sft_tokenizer)
print(f'Dataset size: {len(dataset)} | eval_prompts per call: {script_args.eval_prompts}')

_max_new_tokens = (script_args.max_new_tokens if script_args.max_new_tokens > 0
                   else (128 if script_args.dataset_name in {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K'} else 48))
generation_kwargs = {
    'max_new_tokens': _max_new_tokens, 'min_length': -1,
    'top_k': 0, 'top_p': 0.9, 'temperature': 0.7, 'do_sample': script_args.do_sample,
}

expert_models = load_shared_experts(
    script_args.base_model_name, script_args.expert_model_paths, device,
    tokenizer=sft_tokenizer)

lm_hidden_size = expert_models[0].config.hidden_size
_num_layers    = len(expert_models[0].model.layers)
print(f'lm_hidden_size = {lm_hidden_size}, num_layers = {_num_layers}')

if script_args.gating_type == 'simple':
    template_net = SimpleGatingNetwork(
        lm_hidden_size=lm_hidden_size,
        num_experts=len(expert_models),
        fixed_alpha=script_args.fixed_alpha,
    )
    print(f'Gating type: SimpleGatingNetwork')
else:
    template_net = GatingNetwork(
        lm_hidden_size=lm_hidden_size,
        num_experts=len(expert_models),
        num_layers=_num_layers,
        alpha_init=script_args.fixed_alpha,
        fixed_alpha=script_args.fixed_alpha,
    )
    print(f'Gating type: GatingNetwork (per_layer)')

# warm_start_path: try resume first (gen_XXXX checkpoints), fallback to single-net template load
_resume     = False
_resume_dir = None
if script_args.warm_start_path:
    wsp      = script_args.warm_start_path.rstrip('/')
    wsp_name = os.path.basename(wsp)
    if (wsp_name.startswith('gen_') and wsp_name[4:].isdigit()
            and os.path.isfile(os.path.join(wsp, 'es_state.json'))):
        _resume     = True
        _resume_dir = os.path.dirname(os.path.abspath(wsp))
        _ckpt_gen   = int(wsp_name[4:])
        print(f'Resume detected: checkpoint {wsp_name} (gen={_ckpt_gen}) — '
              f'resume_dir={_resume_dir}')
    else:
        _ckpt_gen, _ = ES._find_latest_checkpoint(wsp)
        if _ckpt_gen >= 0:
            _resume     = True
            _resume_dir = wsp
            print(f'Resume detected: latest checkpoint gen_{_ckpt_gen:04d} in {wsp}')
        else:
            if script_args.gating_type == 'simple':
                loaded = load_simple_gating_network(wsp,
                                                    lm_hidden_size=lm_hidden_size,
                                                    num_experts=len(expert_models), device='cpu')
            else:
                loaded = load_gating_network(wsp,
                                             lm_hidden_size=lm_hidden_size,
                                             num_experts=len(expert_models), num_layers=_num_layers,
                                             device='cpu')
            if loaded is not None:
                template_net = loaded.cpu().bfloat16()
                print(f'Warm-start (template) from {wsp}')

if script_args.gating_type == 'simple':
    moe_model = SimpleMoEForCausalLM(expert_models, template_net).to(device)
    print(f'SimpleMoEForCausalLM: {len(expert_models)} experts')
else:
    moe_model = MoEForCausalLM(expert_models, template_net).to(device)
    print(f'MoEForCausalLM: {len(expert_models)} experts × {moe_model.num_layers} layers')
moe_model.eval()

es = ES(
    template_net=template_net,
    num_objectives=n_objectives,
    population_size=script_args.population_size,
    device=device,
)

print(f'\nStarting dual-front ES — {script_args.num_generations} generations, '
      f'P={es.P} …')
final_population = es.run(
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
    use_greedy_hvc=script_args.use_greedy_hvc,
    algorithm=script_args.algorithm,
    n_reference_divisions=script_args.n_reference_divisions,
    normalize_fitness=script_args.normalize_fitness,
    normalize_rewards=script_args.normalize_rewards,
    use_dual_front=script_args.use_dual_front,
    parent_stability_bonus=script_args.parent_stability_bonus,
    parent_stability_cap=script_args.parent_stability_cap,
    resume=_resume,
    resume_dir=_resume_dir,
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
            json.dump({'fitness': es.fitness[i].tolist()}, f, indent=2)

    fronts = non_dominated_sort(np.array(es.fitness))
    with open(os.path.join(output_dir, 'es_meta.json'), 'w') as f:
        json.dump({
            'reward_names':    reward_names,
            'z_star':          es.z_star.tolist(),
            'final_fitness':   [f.tolist() for f in es.fitness],
            'num_generations': script_args.num_generations,
            'population_size': es.P,
            'lm_hidden_size':  lm_hidden_size,
        }, f, indent=2)

    print(f'\nDone. Non-dominated front: {len(fronts[0])}/{es.P} individuals')
    print(f'Final ideal point z* = {np.round(es.z_star, 4)}')
