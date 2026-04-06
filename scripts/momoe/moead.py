"""moead.py — Evolve GatingNetwork parameters using MOEA/D on raw dataset.

Works directly on the raw HuggingFace dataset (Anthropic/hh-rlhf or
openai/summarize_from_feedback).

Algorithm: MOEA/D (Zhang & Li, 2007) with Chebyshev decomposition.
  - N sub-problems, each defined by a preference vector λ_i on the simplex.
  - One individual (flat GatingNetwork parameter vector) per sub-problem.
  - Fitness: Chebyshev gap on LIVE rewards from merged-expert inference.

Evaluation approximation (for computational tractability):
  A child is evaluated ONCE under its parent's λ_i to obtain a reward vector
  r_child.  That same r_child is then scalarised with each neighbour's λ_j
  for the neighbourhood update.  This avoids T × (generation cost) per step
  while still propagating useful gradient information across the neighbourhood.
  Set --full_neighbor_eval to override and re-evaluate the child under every
  neighbour's λ.
"""

import copy
import datetime
import time
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
)
from trl import set_seed
from peft import PeftModel

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_ppo, build_dataset_summary_ppo,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
    get_clean_data, load_main_tokenizer,
)
from moead_architecture import (
    GatingNetwork,
    MoEForCausalLM,
)
from moead_utils import (
    save_gating_network,
    load_base_model,
    get_simplex_samples,
    REWARD_PATHS,
)


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    # ── Dataset (same as collect_rewards.py) ──────────────────────────────
    sft_model_name:       str       = './models/sft/model/'
    expert_model_paths:   List[str] = field(default_factory=list)
    reward_names:         str       = 'harmless,helpful'
    exp_type:             str       = 'assistant'    # 'assistant' | 'summary'
    split:                str       = 'train'
    do_sample:            bool      = True           # stochastic generation
    num_continuations:    int       = 3              # reward averaging over K generations
    eval_prompts:         int       = 8192             # total prompts sampled per fitness evaluation
    eval_batch_size:      int       = 64               # generation batch size within each eval
    max_new_tokens:       int       = 128

    # ── GatingNetwork architecture ────────────────────────────────────────
    warm_start_path:      str       = ''             # optional pre-trained checkpoint

    # ── MOEA/D hyper-parameters ───────────────────────────────────────────
    pref_step:            float     = 0.1            # simplex grid step
    neighborhood_size:    int       = 5
    num_generations:      int       = 200
    mutation_sigma:       float     = 0.02
    mutation_rate:        float     = 0.05
    sigma_decay:          float     = 0.99
    full_neighbor_eval:   bool      = False          # re-eval child per neighbour λ
    mu:                   int       = 1              # individuals per sub-problem
    full_eval_every:      int       = 10             # use full dataset every N gens (0=off)

    # ── Hardware ──────────────────────────────────────────────────────────
    gpu_id:               int       = -1            # GPU node id; -1 → use accelerator rank

    # ── Output ────────────────────────────────────────────────────────────
    save_directory:       str       = './models/moead/'
    run_name:             str       = 'moead_gating'
    save_every:           int       = 10             # checkpoint every N generations
    seed:                 int       = 42
    verbose:              bool      = False          # detailed per-batch debug logs


# ---------------------------------------------------------------------------
# Genome ↔ GatingNetwork helpers
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
            p.copy_(torch.tensor(
                params[offset:offset + n].reshape(p.shape),
                dtype=p.dtype, device=device))
            offset += n
    return net


# ---------------------------------------------------------------------------
# Fitness evaluation helpers
# ---------------------------------------------------------------------------

def generate_and_score(
    moe_model:           MoEForCausalLM,
    prompt_input_ids:    torch.Tensor,
    prompt_attention:    torch.Tensor,
    prompt_texts_decoded: List[str],
    sft_tokenizer,
    reward_models,
    instructions,
    generation_kwargs:   dict,
    gpu_id:              int,
    num_continuations:   int = 1,
) -> np.ndarray:
    """Generate responses via MoEForCausalLM and return mean reward vector.

    Gating weights are computed dynamically per layer inside moe_model.generate().

    Returns
    -------
    mean_rewards : (num_rewards,) ndarray — mean reward per objective.
    """
    accumulated = None   # list[list[float]] — [prompt][reward]

    for _ in range(num_continuations):
        outputs = moe_model.generate(
            prompt_input_ids.to(f'cuda:{gpu_id}'),
            attention_mask=prompt_attention.to(f'cuda:{gpu_id}'),
            **generation_kwargs,
        )
        responses       = sft_tokenizer.batch_decode(outputs.cpu())
        prompts_decoded = sft_tokenizer.batch_decode(prompt_input_ids.cpu())
        del outputs

        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(instructions.get_input(r), instructions.get_response(r))
                 for r in responses_clean]
        if hasattr(instructions, 'get_post'):
            scores_per_reward = reward_models.get_reward_model_scores(
                pairs, instructions.get_post, normalize_rewards=False, round_digits=None)
        else:
            scores_per_reward = reward_models.get_reward_model_scores(
                pairs, normalize_rewards=False, round_digits=None)
        # scores_per_reward: list[num_rewards] of list[num_prompts]

        n_prompts  = len(prompts_clean)
        n_rewards  = len(scores_per_reward)
        if accumulated is None:
            accumulated = [[[] for _ in range(n_rewards)] for _ in range(n_prompts)]
        for p in range(n_prompts):
            for k in range(n_rewards):
                accumulated[p][k].append(scores_per_reward[k][p])

        torch.cuda.empty_cache()

    # Mean per prompt, then mean over prompts
    per_prompt_mean = np.array(
        [[np.mean(accumulated[p][k]) for k in range(n_rewards)]
         for p in range(n_prompts)]
    )                                          # (n_prompts, n_rewards)
    return per_prompt_mean.mean(axis=0)        # (n_rewards,)


def chebyshev_scalar(reward_vec: np.ndarray, lam: np.ndarray,
                     z_star: np.ndarray) -> float:
    """Chebyshev fitness: higher = better (negated gap)."""
    return -float(np.max(lam * np.abs(reward_vec - z_star)))


# ---------------------------------------------------------------------------
# MOEA/D
# ---------------------------------------------------------------------------

class MOEAD:
    """MOEA/D for GatingNetwork evolution via live reward feedback.

    Population: N flat parameter arrays, one per sub-problem λ_i.
    Fitness is computed via actual expert-model generation + reward scoring.
    Reward vectors (not just scalars) are cached per individual so that the
    same r can be re-scalarised with different λ_j without extra generation.
    """

    def __init__(
        self,
        template_net:      GatingNetwork,
        weight_vectors:    np.ndarray,
        neighborhood_size: int  = 5,
        mu:                int  = 1,
        device:            str  = 'cpu',
    ):
        self.template  = template_net.eval()
        self.lambdas   = weight_vectors.astype(np.float64)
        self.N         = len(weight_vectors)
        self.T         = min(neighborhood_size, self.N)
        self.mu        = mu
        self.device    = device
        self.M         = weight_vectors.shape[1]

        dists   = cdist(self.lambdas, self.lambdas, metric='euclidean')
        self.B  = [np.argsort(dists[i])[:self.T].tolist() for i in range(self.N)]

        base_params    = net_to_params(template_net)
        self.param_dim = len(base_params)
        # population[i][k]: k-th individual of sub-problem i
        self.population = [
            [base_params + np.random.randn(self.param_dim) * 0.05
             for _ in range(self.mu)]
            for _ in range(self.N)
        ]

        self.z_star       = np.full(self.M, -np.inf, dtype=np.float64)
        # reward_cache[i][k] / fitness[i][k]: per-individual cache
        self.reward_cache = [[None] * self.mu for _ in range(self.N)]
        self.fitness      = [np.full(self.mu, -np.inf) for _ in range(self.N)]

        # Per-generation snapshots for post-hoc analysis
        # z_star_history[g]  : z* at the end of generation g  (shape M)
        # fitness_history[g] : best fitness per sub-problem at end of generation g  (shape N)
        self.z_star_history   = []
        self.fitness_history  = []

    @staticmethod
    def _log(rank: int, msg: str, verbose: bool = True):
        """Timestamped, flushed debug print. Suppressed when verbose=False."""
        if not verbose:
            return
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'[{ts}] rank{rank} {msg}', flush=True)

    def _update_z_star(self, r: np.ndarray):
        improved          = r > self.z_star
        self.z_star[improved] = r[improved]

    @staticmethod
    def _crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        beta = np.random.uniform(0.0, 1.0, size=p1.shape)
        return beta * p1 + (1.0 - beta) * p2

    @staticmethod
    def _mutate(x: np.ndarray, sigma: float, rate: float) -> np.ndarray:
        mask  = np.random.random(x.shape) < rate
        noise = np.random.normal(0.0, sigma, x.shape)
        return x + mask.astype(np.float64) * noise

    def run(
        self,
        # Dataset for random sampling each evaluation
        dataset,
        data_collator,
        eval_prompts:         int,            # total prompts to sample per eval call
        eval_batch_size:      int,            # generation batch size within each eval
        # MoE model & generation
        moe_model:            MoEForCausalLM,
        sft_tokenizer,
        reward_models,
        instructions,
        generation_kwargs:    dict,
        gpu_id:               int,
        # Hyper-parameters
        num_generations:      int   = 100,
        mutation_sigma:       float = 0.02,
        mutation_rate:        float = 0.05,
        sigma_decay:          float = 0.995,
        num_continuations:    int   = 1,
        full_neighbor_eval:   bool  = False,
        save_every:           int   = 10,
        output_dir:           str   = '.',
        poll_interval:        float = 2.0,    # seconds between queue polls
        verbose:              bool  = False,  # detailed per-batch debug logs
        eval_seed:            int   = 42,     # base seed; gen g uses eval_seed+g
        full_eval_every:      int   = 10,     # use full dataset every N gens (0=off)
    ) -> List[np.ndarray]:
        """Run MOEA/D with an async file-based work queue.

        Rank 0 is the coordinator: it owns the population, generates tasks at the
        start of each generation, also participates as a worker, and applies all
        results once every task is done.

        All other ranks are pure workers: they poll for unclaimed tasks, evaluate
        them, and write results.  No barriers, no all_reduce/all_gather — zero
        idle time.

        Queue layout (under output_dir/queue/):
            gen_{g:04d}/
                task_{i:03d}.json      written by rank 0: child params + metadata
                claimed_{i:03d}        O_EXCL lock file: atomically created to claim task i
                result_{i:03d}.json    written by worker after eval (via tmp+rename)
                done                   written by rank 0 after applying all results
        """

        # ── Distributed setup ─────────────────────────────────────────────
        dist_on    = torch.distributed.is_initialized()
        rank       = torch.distributed.get_rank()       if dist_on else 0
        is_main    = (rank == 0)

        def log(msg):
            self._log(rank, msg, verbose=verbose)

        queue_root = os.path.join(output_dir, 'queue')
        if is_main:
            os.makedirs(queue_root, exist_ok=True)

        # ── Queue path helpers ────────────────────────────────────────────

        def _gen_dir(gen):
            return os.path.join(queue_root, f'gen_{gen:04d}')

        def _task_path(gen, i):
            return os.path.join(_gen_dir(gen), f'task_{i:03d}.json')

        def _claim_path(gen, i):
            return os.path.join(_gen_dir(gen), f'claimed_{i:03d}')

        def _result_path(gen, i):
            return os.path.join(_gen_dir(gen), f'result_{i:03d}.json')

        def _done_path(gen):
            return os.path.join(_gen_dir(gen), 'done')

        def _try_claim(gen, i):
            """Atomically claim task i. Returns True on success, False if already taken."""
            try:
                fd = os.open(_claim_path(gen, i), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(rank).encode())
                os.close(fd)
                return True
            except FileExistsError:
                return False

        def _write_task(gen, tid, task):
            """Write task atomically via tmp-file + rename (params can be several MB)."""
            tmp = _task_path(gen, tid) + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(task, f)
            os.replace(tmp, _task_path(gen, tid))

        def _write_result(gen, i, r):
            """Write result atomically via tmp-file + rename."""
            tmp = _result_path(gen, i) + f'.tmp_rank{rank}'
            with open(tmp, 'w') as f:
                json.dump({'task_id': i, 'reward_vec': r.tolist()}, f)
            os.replace(tmp, _result_path(gen, i))

        # ── Eval helper ───────────────────────────────────────────────────

        def _sample_loader(seed=None, full=False):
            if full or eval_prompts >= len(dataset):
                return DataLoader(dataset, batch_size=eval_batch_size,
                                  collate_fn=data_collator, drop_last=False)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(dataset), size=eval_prompts, replace=False)
            batch_ds = dataset.select(idx.tolist())
            return DataLoader(batch_ds, batch_size=eval_batch_size,
                              collate_fn=data_collator, drop_last=False)

        def _eval_individual(params, label='', seed=None, full=False):
            log( f'eval start [{label}]{"[FULL]" if full else ""}')
            loader = _sample_loader(seed=seed, full=full)
            net    = params_to_net(params, self.template, self.device)
            net.eval()
            moe_model.gating_net = net
            reward_vecs = []
            for b_idx, batch in enumerate(loader):
                log( f'eval batch {b_idx+1} — generate+score [{label}]')
                input_ids      = batch['input_ids']
                attention_mask = batch['attention_mask']
                prompts_dec    = sft_tokenizer.batch_decode(input_ids)
                r = generate_and_score(
                    moe_model, input_ids, attention_mask, prompts_dec,
                    sft_tokenizer, reward_models, instructions,
                    generation_kwargs, gpu_id, num_continuations)
                log( f'eval batch {b_idx+1} done, r={np.round(r,3)} [{label}]')
                reward_vecs.append(r)
            result = np.mean(reward_vecs, axis=0)
            log( f'eval done, mean_r={np.round(result,3)} [{label}]')
            return result

        # ── Worker loop ───────────────────────────────────────────────────

        def _worker_loop(gen, num_tasks):
            """Claim and evaluate tasks until this generation is complete.

            Rank 0 exits when all num_tasks result files exist (then applies results and
            writes the done file).  All other ranks exit when the done file appears.
            """
            gen_seed = eval_seed + gen   # all tasks in this gen share the same eval subset
            is_full  = (full_eval_every > 0 and gen > 0 and gen % full_eval_every == 0)
            if is_main and is_full:
                print(f'  [gen {gen}] full-dataset eval', flush=True)
            log( f'gen {gen} worker loop start (num_tasks={num_tasks}, seed={gen_seed}, full={is_full})')
            while True:
                # Exit condition differs by role
                if is_main:
                    if all(os.path.exists(_result_path(gen, i)) for i in range(num_tasks)):
                        break
                else:
                    if os.path.exists(_done_path(gen)):
                        break

                # Scan for an unclaimed task
                claimed = False
                for i in range(num_tasks):
                    if not os.path.exists(_task_path(gen, i)):
                        continue                        # task not written yet (race on startup)
                    if os.path.exists(_result_path(gen, i)):
                        continue                        # already done
                    if not _try_claim(gen, i):
                        continue                        # another rank got it
                    claimed = True
                    with open(_task_path(gen, i)) as f:
                        task = json.load(f)
                    child_params = np.array(task['child_params'])
                    log( f'gen {gen} task {i} (λ={task["lambda"]}) claimed')
                    r = _eval_individual(child_params, label=f'g{gen}/t{i}',
                                         seed=gen_seed, full=is_full)
                    _write_result(gen, i, r)
                    log( f'gen {gen} task {i} result written, r={np.round(r,3)}')
                    break   # restart scan from beginning after each eval (avoids stale cache)

                if not claimed:
                    time.sleep(poll_interval)

            log( f'gen {gen} worker loop done')

        # ── Phase 0: initial population evaluation ────────────────────────
        # Write N*mu tasks (one per individual), apply results into population[i][k].
        gen = 0
        num_init_tasks = self.N * self.mu
        if is_main:
            print(f'MOEA/D — initialising population ({num_init_tasks} individuals) …')
            os.makedirs(_gen_dir(gen), exist_ok=True)
            for i in range(self.N):
                for k in range(self.mu):
                    tid = i * self.mu + k
                    task = {'task_id':      tid,
                            'sub_idx':      i,
                            'ind_idx':      k,
                            'lambda':       self.lambdas[i].tolist(),
                            'child_params': self.population[i][k].tolist()}
                    _write_task(gen, tid, task)
            log( f'gen {gen} all {num_init_tasks} init tasks written')

        _worker_loop(gen, num_init_tasks)

        if is_main:
            log( 'gen 0 applying init results')
            for i in range(self.N):
                for k in range(self.mu):
                    tid = i * self.mu + k
                    with open(_result_path(gen, tid)) as f:
                        r = np.array(json.load(f)['reward_vec'])
                    self.reward_cache[i][k] = r
                    self._update_z_star(r)
                    self.fitness[i][k] = chebyshev_scalar(r, self.lambdas[i], self.z_star)
            # Signal workers that gen 0 is complete
            open(_done_path(gen), 'w').close()
            log( f'gen 0 done, z*={np.round(self.z_star,3)}')
        else:
            # Non-main ranks: wait for done before entering gen loop
            while not os.path.exists(_done_path(gen)):
                time.sleep(poll_interval)

        # ── Generational loop ─────────────────────────────────────────────
        sigma = mutation_sigma
        for gen in range(1, num_generations + 1):
            log( f'gen {gen}/{num_generations} start')

            if is_main:
                os.makedirs(_gen_dir(gen), exist_ok=True)
                for i in range(self.N):
                    # Parents drawn from the combined pool of all individuals in the neighborhood
                    pool = [self.population[j][k]
                            for j in self.B[i] for k in range(self.mu)]
                    p1, p2 = pool[np.random.randint(len(pool))], pool[np.random.randint(len(pool))]
                    child  = self._crossover(p1, p2)
                    child  = self._mutate(child, sigma, mutation_rate)
                    task   = {'task_id':      i,
                              'sub_idx':      i,
                              'lambda':       self.lambdas[i].tolist(),
                              'neighbors':    self.B[i],
                              'child_params': child.tolist()}
                    _write_task(gen, i, task)
                log( f'gen {gen} all {self.N} tasks written')

            # All ranks evaluate tasks
            _worker_loop(gen, self.N)

            # Rank 0 applies results and advances population
            if is_main:
                log( f'gen {gen} applying results')
                for i in range(self.N):
                    with open(_task_path(gen, i)) as f:
                        task = json.load(f)
                    with open(_result_path(gen, i)) as f:
                        r_child = np.array(json.load(f)['reward_vec'])
                    child_params = np.array(task['child_params'])
                    self._update_z_star(r_child)
                    for j in task['neighbors']:
                        f_j = chebyshev_scalar(r_child, self.lambdas[j], self.z_star)
                        # Replace the worst individual in neighbor j's pool if child is better
                        worst_k = int(np.argmin(self.fitness[j]))
                        if f_j > self.fitness[j][worst_k]:
                            self.population[j][worst_k]   = child_params.copy()
                            self.reward_cache[j][worst_k] = r_child
                            self.fitness[j][worst_k]      = f_j

                sigma *= sigma_decay
                self.z_star_history.append(self.z_star.copy())
                # Store best fitness per sub-problem for history
                self.fitness_history.append(
                    np.array([np.max(self.fitness[i]) for i in range(self.N)]))

                all_fit = [self.fitness[i][k]
                           for i in range(self.N) for k in range(self.mu)]
                print(
                    f'Gen {gen:4d}/{num_generations} | '
                    f'best_fit={max(all_fit):.4f} | '
                    f'mean_fit={np.mean(all_fit):.4f} | '
                    f'z*={np.round(self.z_star, 3)} | '
                    f'σ={sigma:.5f}',
                    flush=True,
                )
                if gen % save_every == 0:
                    self._save_checkpoint(output_dir, gen)

                # Signal workers this generation is complete
                open(_done_path(gen), 'w').close()
                log( f'gen {gen} done signal written')
            else:
                # Workers wait for rank 0 to finish applying results before next gen
                while not os.path.exists(_done_path(gen)):
                    time.sleep(poll_interval)

        return self.population

    def _save_checkpoint(self, output_dir: str, gen: int):
        """Save checkpoint. Should only be called on rank 0."""
        ckpt_dir = os.path.join(output_dir, f'gen_{gen:04d}')
        os.makedirs(ckpt_dir, exist_ok=True)
        for i in range(self.N):
            lam_str = '_'.join(f'{v:.2f}' for v in self.lambdas[i])
            for k in range(self.mu):
                subdir = os.path.join(ckpt_dir, f'sub_{i:03d}_ind_{k:02d}_lam_{lam_str}')
                net    = params_to_net(self.population[i][k], self.template, 'cpu')
                save_gating_network(net, subdir)
        meta = {
            'generation':     gen,
            'z_star':         self.z_star.tolist(),
            'fitness':        [self.fitness[i].tolist() for i in range(self.N)],
            'reward_cache':   [[r.tolist() if r is not None else None
                                for r in self.reward_cache[i]]
                               for i in range(self.N)],
            'weight_vectors': self.lambdas.tolist(),
        }
        with open(os.path.join(ckpt_dir, 'moead_state.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  Checkpoint saved → {ckpt_dir}', flush=True)


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
        backend='nccl', timeout=datetime.timedelta(minutes=60))
accelerator = Accelerator()
gpu_id      = (script_args.gpu_id if script_args.gpu_id >= 0
               else accelerator.local_process_index)
device      = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_objectives = len(reward_names)

# ── Reward models ───────────────────────────────────────────────────────────
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

# ── Raw dataset (same as collect_rewards.py) ────────────────────────────────
sft_tokenizer              = load_main_tokenizer(script_args.sft_model_name)
sft_tokenizer.padding_side = 'left'

if script_args.exp_type == 'assistant':
    if script_args.split == 'test':
        dataset = build_dataset_eval_ppo(
            'Anthropic/hh-rlhf', sft_tokenizer,
            reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_ppo(
            'Anthropic/hh-rlhf', sft_tokenizer,
            reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions()
else:
    if script_args.split == 'test':
        dataset = build_dataset_summary_eval_ppo(
            'openai/summarize_from_feedback', sft_tokenizer,
            reward_models.rm_tokenizers, split='test')
    else:
        dataset = build_dataset_summary_ppo(
            'openai/summarize_from_feedback', sft_tokenizer,
            reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

data_collator = DataCollatorWithPadding(tokenizer=sft_tokenizer)
print(f'Dataset size: {len(dataset)} | eval_prompts per call: {script_args.eval_prompts}')

generation_kwargs = {
    'max_new_tokens': script_args.max_new_tokens,
    'min_length':     -1,
    'top_k':          0,
    'top_p':          0.9,
    'temperature':    0.7,
    'do_sample':      script_args.do_sample,
}

# ── Load all expert models (kept separate — no merging) ─────────────────────
print(f'Loading {len(script_args.expert_model_paths)} expert models …')
expert_models = []
for i, path in enumerate(script_args.expert_model_paths):
    print(f'  Expert {i+1}: {path}')
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device)
    m.resize_token_embeddings(len(sft_tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    expert_models.append(m)
print(f'  All {len(expert_models)} experts loaded on {device}.')

# lm_hidden_size comes from the expert model config (gating net reads transformer layers directly)
lm_hidden_size = expert_models[0].config.hidden_size
print(f'lm_hidden_size = {lm_hidden_size}')

# ── GatingNetwork template ───────────────────────────────────────────────────
template_net = GatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=n_objectives,
)

if script_args.warm_start_path:
    from moead_utils import load_gating_network
    loaded = load_gating_network(
        script_args.warm_start_path,
        lm_hidden_size=lm_hidden_size,
        num_experts=n_objectives,
        device='cpu',
    )
    if loaded is not None:
        template_net = loaded.cpu()
        print(f'Warm-start from {script_args.warm_start_path}')

# ── Build MoEForCausalLM (experts loaded, gating_net slot filled by MOEAD) ───
moe_model = MoEForCausalLM(expert_models, template_net).to(device)
moe_model.eval()
print(f'MoEForCausalLM: {len(expert_models)} experts × {moe_model.num_layers} layers')

# ── MOEA/D weight vectors ────────────────────────────────────────────────────
simplex = get_simplex_samples(n_objectives, step=script_args.pref_step)
weight_vectors = np.array(simplex, dtype=np.float64)
print(f'Sub-problems (N={len(weight_vectors)}):\n{np.round(weight_vectors, 3)}')

# ── Run MOEA/D ───────────────────────────────────────────────────────────────
moead = MOEAD(
    template_net=template_net,
    weight_vectors=weight_vectors,
    neighborhood_size=script_args.neighborhood_size,
    mu=script_args.mu,
    device=device,
)

print(f'\nStarting MOEA/D — {script_args.num_generations} generations …')
final_population = moead.run(
    dataset=dataset,
    data_collator=data_collator,
    eval_prompts=script_args.eval_prompts,
    eval_batch_size=script_args.eval_batch_size,
    moe_model=moe_model,
    sft_tokenizer=sft_tokenizer,
    reward_models=reward_models,
    instructions=instructions,
    generation_kwargs=generation_kwargs,
    gpu_id=gpu_id,
    num_generations=script_args.num_generations,
    mutation_sigma=script_args.mutation_sigma,
    mutation_rate=script_args.mutation_rate,
    sigma_decay=script_args.sigma_decay,
    num_continuations=script_args.num_continuations,
    full_neighbor_eval=script_args.full_neighbor_eval,
    save_every=script_args.save_every,
    output_dir=output_dir,
    verbose=script_args.verbose,
    eval_seed=script_args.seed,
    full_eval_every=script_args.full_eval_every,
)

# ── Save final population (rank 0 only) ──────────────────────────────────────
is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

if is_main:
    print(f'\nSaving final population to {output_dir}/final …')
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    for i in range(len(weight_vectors)):
        lam_str = '_'.join(f'{v:.2f}' for v in weight_vectors[i])
        for k in range(moead.mu):
            subdir = os.path.join(final_dir, f'sub_{i:03d}_ind_{k:02d}_lam_{lam_str}')
            net    = params_to_net(final_population[i][k], template_net, 'cpu')
            save_gating_network(net, subdir)
            with open(os.path.join(subdir, 'lambda.json'), 'w') as f:
                json.dump({'lambda':  weight_vectors[i].tolist(),
                           'fitness': float(moead.fitness[i][k]),
                           'z_star':  moead.z_star.tolist()}, f, indent=2)

    best_i = max(range(len(weight_vectors)), key=lambda i: float(np.max(moead.fitness[i])))
    best_k = int(np.argmax(moead.fitness[best_i]))
    best_net = params_to_net(final_population[best_i][best_k], template_net, 'cpu')
    best_dir = os.path.join(output_dir, 'best')
    save_gating_network(best_net, best_dir)
    with open(os.path.join(best_dir, 'lambda.json'), 'w') as f:
        json.dump({'lambda':        weight_vectors[best_i].tolist(),
                   'fitness':       float(moead.fitness[best_i][best_k]),
                   'subproblem_id': best_i,
                   'ind_id':        best_k,
                   'z_star':        moead.z_star.tolist()}, f, indent=2)

    with open(os.path.join(output_dir, 'moead_meta.json'), 'w') as f:
        json.dump({
            'reward_names':      reward_names,
            'weight_vectors':    weight_vectors.tolist(),
            'z_star':            moead.z_star.tolist(),
            'final_fitness':     [moead.fitness[i].tolist() for i in range(len(weight_vectors))],
            'fitness_history':   [f.tolist() for f in moead.fitness_history],
            'z_star_history':    [z.tolist() for z in moead.z_star_history],
            'reward_cache':      [[r.tolist() if r is not None else None
                                   for r in moead.reward_cache[i]]
                                  for i in range(len(weight_vectors))],
            'num_generations':   script_args.num_generations,
            'eval_prompts':      script_args.eval_prompts,
            'lm_hidden_size':    lm_hidden_size,
            'mu':                moead.mu,
        }, f, indent=2)

    print(f'\nDone. Best sub-problem: idx={best_i}, ind={best_k}, '
          f'λ={weight_vectors[best_i].round(3)}, '
          f'fitness={moead.fitness[best_i][best_k]:.4f}')
    print(f'Final ideal point z* = {np.round(moead.z_star, 4)}')
