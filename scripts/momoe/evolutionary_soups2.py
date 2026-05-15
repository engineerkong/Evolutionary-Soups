"""evolutionary_soups2.py — Fine-tune a single GatingNetwork on last hidden states.

The GatingNetwork merges expert models' last hidden states (after all transformer
layers, before LM head) using a single lightweight MLP:
  - Input:  mean-pooled last hidden state  (B, lm_hidden_size)  [from first expert]
  - Output: merging coefficients           (B, num_experts)      via softmax

Only the GatingNetwork is trained; all expert LLMs remain frozen.
Dataset and tokenizer setup mirrors scripts/fine-tuning/sft.py.
"""

import copy
import datetime
import time
import json
import shutil
import sys
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import DataCollatorForCompletionOnlyLM, set_seed

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_sft, build_dataset_summary_sft, build_dataset_news_summary_sft,
    build_dataset_beaver_sft, build_dataset_steer_sft, build_dataset_ultrafeedback_sft,
    build_dataset_ppo, build_dataset_summary_ppo, build_dataset_news_summary_ppo,
    build_dataset_beaver_ppo, build_dataset_steer_ppo, build_dataset_ultrafeedback_ppo,
    get_clean_data, load_main_tokenizer,
)
from scripts.utils.multi_reward_models import RewardModels
from nsgaii_utils import REWARD_PATHS

tqdm.pandas()


# =========================================================================
# Script arguments
# =========================================================================

@dataclass
class ScriptArguments:
    base_model_name:    str       = field(default='meta-llama/Llama-2-7b-hf')
    expert_model_paths: List[str] = field(default_factory=list,
                                          metadata={"help": "paths to LoRA adapter dirs for each expert"})
    dataset_name:       str       = field(default='Anthropic/hh-rlhf')
    log_with:           str       = field(default='none', metadata={"help": "'wandb' or 'none'"})
    save_directory:     str       = field(default='./models/gating_sft/')
    wandb_name:         str       = field(default='gating_sft')
    entropy_coef:       float     = field(default=0.1,
                                          metadata={"help": "coefficient for gating entropy regularization (prevents collapse)"})
    gating_hidden_size: int       = field(default=256,
                                          metadata={"help": "hidden size of the gating MLP"})
    # ---- Stage 2: NSGA-II evolution ----
    reward_names:           str   = field(default='beaver_reward,beaver_cost')
    do_sample:              bool  = field(default=False)
    num_continuations:      int   = field(default=1)
    eval_prompts:           int   = field(default=1024)
    eval_batch_size:        int   = field(default=128)
    max_new_tokens:         int   = field(default=-1)
    normalize_rewards:      bool  = field(default=False)
    warm_start_path:        str   = field(default='')
    algorithm:              str   = field(default='nsga4')
    population_size:        int   = field(default=20)
    num_generations:        int   = field(default=100)
    convergence_window:     int   = field(default=10)
    mutation_sigma:         float = field(default=0.05)
    mutation_rate:          float = field(default=0.5)
    sigma_decay:            float = field(default=0.99)
    sigma_min:              float = field(default=0.03)
    parent_stability_bonus: float = field(default=0.01)
    parent_stability_cap:   float = field(default=0.10)
    use_dual_front:         bool  = field(default=True)
    normalize_fitness:      bool  = field(default=True)
    es_save_every:          int   = field(default=5)
    es_verbose:             bool  = field(default=False)
    es_seed:                int   = field(default=8888)


# =========================================================================
# Simple Gating Network
# =========================================================================

class SimpleGatingNetwork(nn.Module):
    """Single MLP: prompt-token mean-pooled hidden state → per-sequence expert merging coefficients.

    Input:  (B, lm_hidden_size)  [already mean-pooled over prompt tokens by caller]
    Output: (B, num_experts)     [softmax merging coefficients — one set per sequence]
    """

    def __init__(self, lm_hidden_size: int, num_experts: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
        ).to(torch.bfloat16)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (B, lm_hidden_size)  mean of prompt-token hidden states
        Returns:
            (B, num_experts)  softmax coefficients
        """
        logits = self.net(pooled.to(self.net[0].weight.dtype))
        return F.softmax(logits.float(), dim=-1).to(logits.dtype)


# =========================================================================
# Token sampling helper (used by generate)
# =========================================================================

def _sample_token(logits: torch.Tensor, temperature: float,
                  top_p: float, top_k: int) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        top_vals, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < top_vals[:, -1:]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float('-inf')
        logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


# =========================================================================
# MoE model: frozen experts + trainable gating over last hidden states
# =========================================================================

class MoELastHiddenForCausalLM(nn.Module):
    """Merge expert last hidden states with a trainable SimpleGatingNetwork.

    Expert transformer layers run under torch.no_grad().
    Gradients flow only through the gating network parameters.
    """

    def __init__(self, expert_models: list, gating_net: SimpleGatingNetwork,
                 entropy_coef: float = 0.1):
        super().__init__()
        self.experts      = nn.ModuleList(expert_models)
        self.gating_net   = gating_net
        self.config       = expert_models[0].config
        self.entropy_coef = entropy_coef

        for exp in self.experts:
            for p in exp.parameters():
                p.requires_grad = False

        # Averaged lm_head: removes structural bias toward any single expert.
        # Computed once at init from frozen weights; registered as a plain nn.Linear.
        with torch.no_grad():
            avg_weight = torch.stack(
                [exp.lm_head.weight.float() for exp in expert_models], dim=0).mean(dim=0)
            avg_bias = None
            if expert_models[0].lm_head.bias is not None:
                avg_bias = torch.stack(
                    [exp.lm_head.bias.float() for exp in expert_models], dim=0).mean(dim=0)
        lm_h, lm_w = avg_weight.shape
        self.shared_lm_head = nn.Linear(lm_w, lm_h, bias=avg_bias is not None,
                                        dtype=expert_models[0].lm_head.weight.dtype)
        self.shared_lm_head.weight = nn.Parameter(avg_weight.to(expert_models[0].lm_head.weight.dtype),
                                                   requires_grad=False)
        if avg_bias is not None:
            self.shared_lm_head.bias = nn.Parameter(avg_bias.to(expert_models[0].lm_head.weight.dtype),
                                                     requires_grad=False)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=128,
                 do_sample=False, temperature=1.0, top_p=0.9, top_k=0, **kwargs):
        """Autoregressive generation with per-expert KV caches.

        Gating coefficients are computed once during prefill from the mean of all
        experts' last hidden states, then reused for every decode step.
        """
        device = input_ids.device
        B      = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated        = input_ids.clone()
        cur_attn         = attention_mask.clone()
        expert_past_kvs  = [None] * len(self.experts)
        seq_coefficients = None   # frozen after prefill

        for step in range(max_new_tokens):
            cur_ids = generated if step == 0 else generated[:, -1:]

            expert_hidden = []
            for i, expert in enumerate(self.experts):
                out = expert.model(
                    input_ids=cur_ids,
                    attention_mask=cur_attn,
                    past_key_values=expert_past_kvs[i],
                    use_cache=True,
                )
                expert_past_kvs[i] = out.past_key_values
                expert_hidden.append(out.last_hidden_state)   # (B, cur_len, H)

            if seq_coefficients is None:
                # Prefill: gate on mean-pooled hidden states over the whole prompt
                avg_hidden       = torch.stack(expert_hidden, dim=0).mean(dim=0)
                pooled           = avg_hidden.mean(dim=1)          # (B, H)
                seq_coefficients = self.gating_net(pooled)         # (B, num_experts)

            merged      = sum(seq_coefficients[:, i].view(B, 1, 1) * expert_hidden[i]
                              for i in range(len(self.experts)))
            next_logits = self.shared_lm_head(merged)[:, -1, :]    # (B, vocab)

            if do_sample:
                next_token = _sample_token(next_logits, temperature, top_p, top_k)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            cur_attn  = torch.cat(
                [cur_attn, torch.ones(B, 1, dtype=cur_attn.dtype, device=device)], dim=1)

            eos = getattr(self.config, 'eos_token_id', None)
            if eos is not None and (next_token.squeeze(-1) == eos).all():
                break

        return generated

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # --- Expert forward passes (frozen, no grad) ---
        expert_hidden = []
        for expert in self.experts:
            with torch.no_grad():
                h = expert.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).last_hidden_state   # (B, seq, H)
            expert_hidden.append(h)

        # --- Per-sequence gating: mean-pool over prompt tokens only ---
        # labels == -100 marks prompt positions (DataCollatorForCompletionOnlyLM convention).
        # This matches inference behaviour where only prompt tokens are available.
        avg_hidden = torch.stack(expert_hidden, dim=0).mean(dim=0)  # (B, seq, H)
        if labels is not None:
            prompt_mask = (labels == -100)                               # (B, seq)
            if attention_mask is not None:
                prompt_mask = prompt_mask & attention_mask.bool()
        else:
            prompt_mask = attention_mask.bool() if attention_mask is not None \
                          else torch.ones(avg_hidden.shape[:2], dtype=torch.bool,
                                         device=avg_hidden.device)
        weight = prompt_mask.float().unsqueeze(-1)                       # (B, seq, 1)
        pooled = (avg_hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)  # (B, H)
        coefficients = self.gating_net(pooled)                           # (B, num_experts)

        # --- Merge last hidden states ---
        B = input_ids.shape[0]
        merged = sum(
            coefficients[:, i].view(B, 1, 1) * expert_hidden[i]
            for i in range(len(self.experts))
        )   # (B, seq, H)

        # --- Shared averaged lm_head (unbiased toward any expert) ---
        logits = self.shared_lm_head(merged)   # (B, seq, vocab)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().long()
            nll_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Entropy regularization: maximise gating entropy to prevent collapse.
            # H = -Σ p_i log(p_i);  adding +entropy_coef * H penalises one-hot solutions.
            entropy = -(coefficients * (coefficients + 1e-8).log()).sum(dim=-1).mean()
            loss = nll_loss - self.entropy_coef * entropy

        return CausalLMOutputWithPast(loss=loss, logits=logits)


# =========================================================================
# Stage 2 — genome helpers
# =========================================================================

def net_to_params(net: SimpleGatingNetwork) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().float().numpy().ravel() for p in net.parameters()]
    )


def params_to_net(params: np.ndarray, template: SimpleGatingNetwork,
                  device: str = 'cpu') -> SimpleGatingNetwork:
    net    = copy.deepcopy(template).to(device)
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(torch.tensor(params[offset:offset + n].reshape(p.shape),
                                 dtype=p.dtype, device=device))
            offset += n
    return net


def _make_onehot_params(template: SimpleGatingNetwork, expert_idx: int) -> np.ndarray:
    """Params that force gating to output one-hot[expert_idx] for any input."""
    net = copy.deepcopy(template)
    with torch.no_grad():
        last = net.net[-1]
        last.weight.zero_()
        last.bias.fill_(-100.0)
        last.bias[expert_idx] = 100.0
    return net_to_params(net)


def _save_simple_gating_network(net: SimpleGatingNetwork, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(save_path, 'gating_network.pt'))
    config = {
        'type':           'SimpleGatingNetwork',
        'lm_hidden_size': net.net[0].in_features,
        'num_experts':    net.net[-1].out_features,
        'hidden_size':    net.net[0].out_features,
    }
    with open(os.path.join(save_path, 'gating_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


# =========================================================================
# Stage 2 — reward evaluation
# =========================================================================

def generate_and_score(
    moe_model, prompt_input_ids, prompt_attention,
    sft_tokenizer, reward_models, instructions,
    generation_kwargs, gpu_id, num_continuations=1,
    normalize_rewards=False,
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
                normalize_rewards=normalize_rewards, round_digits=None)
        else:
            scores = reward_models.get_reward_model_scores(
                pairs, normalize_rewards=normalize_rewards, round_digits=None)

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


# =========================================================================
# Stage 2 — NSGA-II / NSGA-HVC selection
# =========================================================================

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def _non_dominated_sort(rewards: np.ndarray) -> List[List[int]]:
    P          = len(rewards)
    dom_count  = np.zeros(P, dtype=int)
    dominates  = [[] for _ in range(P)]
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


try:
    from pymoo.indicators.hv import HV as _PymooHV

    def _hv(points: np.ndarray, ref: np.ndarray) -> float:
        ind = _PymooHV(ref_point=-ref)
        return float(ind(-points))

    def _hypervolume_contribution(candidates: np.ndarray, ref: np.ndarray) -> np.ndarray:
        n        = len(candidates)
        total_hv = _hv(candidates, ref)
        hvc      = np.zeros(n)
        for i in range(n):
            hvc[i] = total_hv - _hv(np.delete(candidates, i, axis=0), ref)
        return hvc

    def _nsga4_select(rewards: np.ndarray, pop_size: int) -> List[int]:
        fronts   = _non_dominated_sort(rewards)
        selected: List[int] = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                n_needed   = pop_size - len(selected)
                candidates = np.array([rewards[i] for i in front])
                all_pts    = np.vstack([rewards[selected], candidates]) if selected else candidates
                ref        = all_pts.min(axis=0) - 0.1
                hvc        = _hypervolume_contribution(candidates, ref)
                ranked     = [x for _, x in sorted(zip(-hvc, front))]
                selected.extend(ranked[:n_needed])
                break
        return selected

    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False


# =========================================================================
# Stage 2 — NSGAII evolutionary algorithm
# =========================================================================

class NSGAII:
    """Evolutionary algorithm for SimpleGatingNetwork evolution.

    Identical in structure to evolutionary_soups.py's NSGAII class,
    adapted for SimpleGatingNetwork (no per-layer alpha, no num_layers).
    """

    def __init__(self, template_net: SimpleGatingNetwork, num_objectives: int,
                 population_size: int = 20, device: str = 'cpu'):
        self.template  = template_net.eval()
        self.M         = num_objectives
        self.P         = population_size
        self.device    = device

        base_params    = net_to_params(template_net)
        self.param_dim = len(base_params)
        self.population = [
            base_params + np.random.randn(self.param_dim) * 0.05
            for _ in range(self.P)
        ]
        self.fitness           = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.parent_baseline   = [np.full(self.M, -np.inf) for _ in range(self.P)]
        self.parent_eval_count = [0] * self.P
        self.z_star            = np.full(self.M, -np.inf, dtype=np.float32)

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
            _save_simple_gating_network(net, subdir)
            with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
                json.dump({'fitness': self.fitness[i].tolist()}, f, indent=2)
        meta = {'generation': gen, 'z_star': self.z_star.tolist(),
                'fitness':    [f.tolist() for f in self.fitness]}
        with open(os.path.join(ckpt_dir, 'nsgaii_state.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  Checkpoint saved → {ckpt_dir}', flush=True)

    def run(
        self,
        dataset, data_collator, eval_prompts: int, eval_batch_size: int,
        moe_model: MoELastHiddenForCausalLM, sft_tokenizer,
        reward_models, instructions, generation_kwargs: dict, gpu_id: int,
        num_generations: int = 100, mutation_sigma: float = 0.05,
        mutation_rate: float = 0.3, sigma_decay: float = 0.999,
        num_continuations: int = 1, save_every: int = 10,
        output_dir: str = '.', poll_interval: float = 2.0,
        verbose: bool = False, seed: int = 42,
        sigma_min: float = 0.005, convergence_window: int = 10,
        parent_stability_bonus: float = 0.01, parent_stability_cap: float = 0.10,
        use_dual_front: bool = True, algorithm: str = 'nsga4',
        normalize_rewards: bool = False, normalize_fitness: bool = False,
    ) -> List[np.ndarray]:

        dist_on  = torch.distributed.is_initialized()
        rank     = torch.distributed.get_rank() if dist_on else 0
        is_main  = rank == 0

        assert algorithm in ('nsgaii', 'nsga4'), \
            f"algorithm must be 'nsgaii' or 'nsga4', got {algorithm!r}"
        if algorithm == 'nsga4' and not _PYMOO_AVAILABLE:
            print('pymoo not available — falling back to nsgaii', flush=True)
            algorithm = 'nsgaii'
        _select = _nsga4_select if algorithm == 'nsga4' else _nsga2_select

        def log(msg): self._log(rank, msg, verbose)

        queue_root = os.path.join(output_dir, 'queue')
        if is_main:
            if os.path.exists(queue_root):
                shutil.rmtree(queue_root)
            os.makedirs(queue_root)

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
                    generation_kwargs, gpu_id, num_continuations, normalize_rewards)
                reward_vecs.append(r)
            return np.mean(reward_vecs, axis=0)

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

        bounds: dict = {'min': None, 'range': None}

        def _collect_fitness(gen, task_id) -> np.ndarray:
            with open(_result_path(gen, task_id)) as f:
                r = np.array(json.load(f)['reward_vec'])
            if bounds['range'] is not None:
                r = (r - bounds['min']) / bounds['range']
            self._update_z_star(r)
            return r

        if normalize_fitness:
            _BG = -1
            if is_main:
                print(f'Computing expert reward bounds ({len(list(self.template.net.parameters())[-1].data)} experts) …', flush=True)
                os.makedirs(_gen_dir(_BG), exist_ok=True)
                n_exp = self.template.net[-1].out_features
                for exp_idx in range(n_exp):
                    _write_task(_BG, exp_idx, {
                        'task_id':      exp_idx,
                        'chunk_idx':    0,
                        'child_params': _make_onehot_params(self.template, exp_idx).tolist(),
                    })
            n_exp = self.template.net[-1].out_features
            _worker_loop(_BG, 0, n_exp, _done_path(_BG))
            if is_main:
                expert_rewards = []
                for exp_idx in range(n_exp):
                    with open(_result_path(_BG, exp_idx)) as f:
                        r = np.array(json.load(f)['reward_vec'])
                    expert_rewards.append(r)
                    print(f'  expert {exp_idx}: {np.round(r, 3)}', flush=True)
                r_min   = np.min(expert_rewards, axis=0)
                r_max   = np.max(expert_rewards, axis=0)
                r_range = np.maximum(r_max - r_min, 1e-6)
                bounds['min']   = r_min
                bounds['range'] = r_range
                open(_done_path(_BG), 'w').close()
            else:
                while not os.path.exists(_done_path(_BG)):
                    time.sleep(poll_interval)

        # ── Phase 0: initial population ──────────────────────────────────────
        gen = 0
        if is_main:
            print(f'Stage 2 NSGA-II — {self.P} individuals …')
            os.makedirs(_gen_dir(gen), exist_ok=True)
            for i in range(self.P):
                _write_task(gen, i, {'task_id': i, 'chunk_idx': 0,
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
        sigma                = mutation_sigma
        parents_kept_hist:   list = []
        diversity_hist:      list = []
        z_star_hist:         list = []

        for gen in range(1, num_generations + 1):
            log(f'gen {gen}/{num_generations} start')

            if is_main:
                os.makedirs(_gen_dir(gen), exist_ok=True)
                chunk_idx = gen % _num_chunks
                diversity = np.std(np.array(self.population), axis=0).mean()
                parent_params = list(self.population)

                for i in range(self.P):
                    _write_task(gen, i, {'task_id': i, 'chunk_idx': chunk_idx,
                                         'child_params': self.population[i].tolist()})
                child_params_list = []
                for i in range(self.P):
                    pi1 = np.random.randint(len(parent_params))
                    pi2 = np.random.randint(len(parent_params))
                    child, _ = self._crossover(parent_params[pi1], parent_params[pi2])
                    child = self._mutate(child, sigma, mutation_rate)
                    child_params_list.append(child)
                    _write_task(gen, self.P + i, {'task_id': self.P + i,
                                                   'chunk_idx': chunk_idx,
                                                   'child_params': child.tolist()})
                open(_tasks_ready_path(gen), 'w').close()
            else:
                while not os.path.exists(_tasks_ready_path(gen)):
                    if os.path.exists(_converged_path): break
                    time.sleep(poll_interval)
                if os.path.exists(_converged_path): break

            _worker_loop(gen, 0, 2 * self.P, _done_path(gen))

            if is_main:
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

                if not use_dual_front:
                    selected       = _select(merged_fitness, self.P)
                    n_intersection = len(selected)
                else:
                    front1_set = set(_non_dominated_sort(merged_fitness)[0])
                    parent_front2 = [
                        self.parent_baseline[i] * (1.0 + min(self.parent_eval_count[i] * parent_stability_bonus,
                                                              parent_stability_cap))
                        for i in range(self.P)
                    ]
                    merged_front2  = np.vstack([np.array(parent_front2), np.array(child_fitness)])
                    front2_fronts  = _non_dominated_sort(merged_front2)
                    front2_set     = set(front2_fronts[0])

                    def _greedy_hvc_fill(seed_set, pool, n):
                        if n <= 0 or not pool: return []
                        all_pts = np.array([merged_fitness[k] for k in seed_set + pool])
                        ref     = all_pts.min(axis=0) - 0.1
                        chosen  = [merged_fitness[k] for k in seed_set]
                        hv_base = _hv(np.array(chosen), ref) if chosen else 0.0
                        result, remaining = [], list(range(len(pool)))
                        for _ in range(min(n, len(remaining))):
                            gains    = [_hv(np.array(chosen + [merged_fitness[pool[loc]]]), ref) - hv_base
                                        for loc in remaining]
                            best_pos = int(np.argmax(gains))
                            best_loc = remaining[best_pos]
                            hv_base += gains[best_pos]
                            chosen.append(merged_fitness[pool[best_loc]])
                            result.append(pool[best_loc])
                            remaining.remove(best_loc)
                        return result

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

                new_population    = [merged_params[k] for k in selected]
                new_fitness       = [merged_fitness[k] for k in selected]
                new_baseline, new_eval_count = [], []
                for k in selected:
                    if k < self.P:
                        new_baseline.append(self.parent_baseline[k].copy())
                        new_eval_count.append(self.parent_eval_count[k])
                    else:
                        new_baseline.append(merged_fitness[k].copy())
                        new_eval_count.append(1)

                self.population        = new_population
                self.fitness           = new_fitness
                self.parent_baseline   = new_baseline
                self.parent_eval_count = new_eval_count

            if is_main:
                sigma = max(sigma_min, sigma * sigma_decay)
                fit_arr  = np.array(self.fitness)
                base_arr = np.array(self.parent_baseline)

                log_path = os.path.join(output_dir, 'population_log.json')
                try:
                    with open(log_path) as f: pop_log = json.load(f)
                except FileNotFoundError:
                    pop_log = {}
                pop_log[f'gen_{gen:04d}'] = {
                    label: {'raw': fit_arr[j].tolist(), 'baseline': base_arr[j].tolist()}
                    for j, label in enumerate(ind_labels)
                }
                with open(log_path, 'w') as f:
                    json.dump(pop_log, f, indent=2)

                print(
                    f'Gen {gen:4d}/{num_generations} | '
                    f'parents_kept={n_parents_kept}/{self.P} | '
                    f'intersect={n_intersection}/{self.P} | '
                    f'mean_fit={np.round(fit_arr.mean(axis=0), 3)} | '
                    f'best_fit={np.round(fit_arr.max(axis=0), 3)} | '
                    f'z*={np.round(self.z_star, 3)} | '
                    f'σ={sigma:.5f} | div={diversity:.5f}',
                    flush=True,
                )

                if gen % save_every == 0:
                    self._save_checkpoint(output_dir, gen)

                converged = False
                if convergence_window > 0:
                    parents_kept_hist.append(n_parents_kept)
                    diversity_hist.append(diversity)
                    z_star_hist.append(self.z_star.copy())
                    if len(parents_kept_hist) >= convergence_window:
                        avg_kept  = np.mean(parents_kept_hist[-convergence_window:])
                        delta_div = abs(diversity_hist[-1] - diversity_hist[-convergence_window])
                        z_stable  = all(np.array_equal(z_star_hist[-convergence_window], z)
                                        for z in z_star_hist[-convergence_window:])
                        if avg_kept >= 37.5 and delta_div < 0.002 and z_stable:
                            print(f'  *** Convergence at gen {gen} ***', flush=True)
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
                if os.path.exists(_converged_path): break

        return self.population


# =========================================================================
# Main
# =========================================================================

parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
training_args.output_dir = output_dir
training_args.run_name   = script_args.wandb_name
training_args.report_to  = [] if script_args.log_with == 'none' else [script_args.log_with]
os.makedirs(output_dir, exist_ok=True)

print(f"Script arguments: {script_args}")
print(f"Training config: {training_args}")

set_seed(training_args.seed if training_args.seed else 8888)

accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id
device      = f'cuda:{gpu_id}'
print(f'process: {process_id}, gpu id: {gpu_id}')

# =========================================================================
# Load tokenizer and dataset
# =========================================================================

tokenizer              = load_main_tokenizer(script_args.expert_model_paths[0])
tokenizer.padding_side = 'left'

if script_args.dataset_name == 'Anthropic/hh-rlhf':
    dataset = build_dataset_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    dataset = build_dataset_summary_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    dataset = build_dataset_beaver_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]
else:
    raise ValueError(
        f'Unsupported dataset_name: {script_args.dataset_name!r}. '
        f'Choose from: Anthropic/hh-rlhf, openai/summarize_from_feedback, PKU-Alignment/PKU-SafeRLHF-10K'
    )

train_dataset = dataset.shuffle()
# Trainer needs only input_ids; collator adds attention_mask and labels
train_dataset = train_dataset.remove_columns(
    [c for c in train_dataset.column_names if c not in ('input_ids',)]
)
print(f"Train set size: {len(train_dataset)}")

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
    mlm=False,
)

# =========================================================================
# Load expert models (frozen)
# =========================================================================

print(f"Loading {len(script_args.expert_model_paths)} expert models …")
expert_models = []
for i, path in enumerate(script_args.expert_model_paths):
    print(f"  Expert {i + 1}: {path}")
    base = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.resize_token_embeddings(len(tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    expert_models.append(m)
print(f"  All {len(expert_models)} experts loaded on {device}.")

lm_hidden_size = expert_models[0].config.hidden_size
print(f"lm_hidden_size = {lm_hidden_size}")

# =========================================================================
# Build gating network and MoE model
# =========================================================================

gating_net = SimpleGatingNetwork(
    lm_hidden_size=lm_hidden_size,
    num_experts=len(expert_models),
    hidden_size=script_args.gating_hidden_size,
).to(device)

model = MoELastHiddenForCausalLM(expert_models, gating_net,
                                 entropy_coef=script_args.entropy_coef).to(device)
model.train()

if process_id == 0:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable / total:.6f}%")

# =========================================================================
# Diagnostics callback
# =========================================================================

class GatingDiagnosticsCallback(TrainerCallback):
    """Logs gating coefficient distribution and grad norm every log_steps steps.

    Check 1: are coefficients stuck near uniform (0.5, 0.5)?
    Check 2: is the learning rate / warmup reasonable?
    """

    def __init__(self, gating_net: SimpleGatingNetwork, log_steps: int = 10):
        self._coeff_buffer: list = []
        self._hook = gating_net.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        # output: (B, num_experts) — detach to avoid holding grad graph
        self._coeff_buffer.append(output.detach().float().cpu())

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self._coeff_buffer or state.global_step == 0:
            return
        coeffs = torch.cat(self._coeff_buffer, dim=0)   # (N_samples, num_experts)
        self._coeff_buffer.clear()

        mean = coeffs.mean(dim=0)    # (num_experts,)
        std  = coeffs.std(dim=0)     # (num_experts,)
        mn   = coeffs.min(dim=0).values
        mx   = coeffs.max(dim=0).values

        print(
            f"[gating step {state.global_step}] "
            f"mean={mean.numpy().round(4).tolist()}  "
            f"std={std.numpy().round(4).tolist()}  "
            f"min={mn.numpy().round(4).tolist()}  "
            f"max={mx.numpy().round(4).tolist()}",
            flush=True,
        )

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._hook.remove()


# =========================================================================
# Trainer — only gating parameters receive gradients
# =========================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[GatingDiagnosticsCallback(gating_net, log_steps=training_args.logging_steps)],
)

print("Training gating network …")
trainer.train()

# =========================================================================
# Save
# =========================================================================

if process_id == 0:
    print("Saving gating network …")
    sft_save_path = os.path.join(output_dir, 'gating_network')
    _save_simple_gating_network(gating_net, sft_save_path)
    tokenizer.save_pretrained(sft_save_path)
    print(f"Stage 1 saved to {sft_save_path}")

# =========================================================================
# Stage 2 — NSGA-II evolution from SFT-pretrained gating network
# =========================================================================

reward_names       = [x.strip() for x in script_args.reward_names.split(',')]
n_objectives       = len(reward_names)
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models_obj  = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

tokenizer.padding_side = 'left'

if script_args.dataset_name == 'Anthropic/hh-rlhf':
    eval_dataset = build_dataset_ppo(
        script_args.dataset_name, tokenizer,
        reward_models_obj.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    eval_dataset = build_dataset_summary_ppo(
        script_args.dataset_name, tokenizer,
        reward_models_obj.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    eval_dataset = build_dataset_beaver_ppo(
        script_args.dataset_name, tokenizer,
        rm_tokenizer=reward_models_obj.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    eval_dataset = build_dataset_steer_ppo(
        script_args.dataset_name, tokenizer,
        rm_tokenizer=reward_models_obj.rm_tokenizers[0], split='train')
    instructions = Instructions()
elif script_args.dataset_name == 'argilla/news-summary':
    eval_dataset = build_dataset_news_summary_ppo(
        script_args.dataset_name, tokenizer,
        reward_models_obj.rm_tokenizers[0], split='test')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'openbmb/UltraFeedback':
    eval_dataset = build_dataset_ultrafeedback_ppo(
        script_args.dataset_name, tokenizer,
        rm_tokenizer=reward_models_obj.rm_tokenizers[0], split='train')
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name for Stage 2: {script_args.dataset_name!r}')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in eval_dataset.column_names:
        eval_dataset = eval_dataset.remove_columns(key)
eval_dataset  = eval_dataset.with_format('numpy')
eval_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(f'Stage 2 eval dataset size: {len(eval_dataset)} | eval_prompts per call: {script_args.eval_prompts}')

_max_new_tokens = (script_args.max_new_tokens if script_args.max_new_tokens > 0
                   else (128 if script_args.dataset_name in {
                       'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K'} else 48))
generation_kwargs = {
    'max_new_tokens': _max_new_tokens, 'min_length': -1,
    'do_sample': script_args.do_sample,
}

es_output_dir = os.path.join(output_dir, 'es2')
os.makedirs(es_output_dir, exist_ok=True)

np.random.seed(script_args.es_seed)

# Warm-start: optionally load a different gating network as the template
template_net = gating_net.cpu()
if script_args.warm_start_path:
    ws_state = torch.load(os.path.join(script_args.warm_start_path, 'gating_network.pt'),
                          map_location='cpu')
    template_net = SimpleGatingNetwork(
        lm_hidden_size=lm_hidden_size,
        num_experts=len(expert_models),
        hidden_size=script_args.gating_hidden_size,
    )
    template_net.load_state_dict(ws_state)
    print(f'Stage 2 warm-start from {script_args.warm_start_path}')

model.eval()
model.gating_net = template_net.to(device)

nsgaii = NSGAII(
    template_net=template_net,
    num_objectives=n_objectives,
    population_size=script_args.population_size,
    device=device,
)

print(f'\nStarting Stage 2 — {script_args.num_generations} generations, '
      f'P={nsgaii.P}, algorithm={script_args.algorithm} …')
final_population = nsgaii.run(
    dataset=eval_dataset, data_collator=eval_collator,
    eval_prompts=script_args.eval_prompts, eval_batch_size=script_args.eval_batch_size,
    moe_model=model, sft_tokenizer=tokenizer,
    reward_models=reward_models_obj, instructions=instructions,
    generation_kwargs=generation_kwargs, gpu_id=gpu_id,
    num_generations=script_args.num_generations,
    mutation_sigma=script_args.mutation_sigma, mutation_rate=script_args.mutation_rate,
    sigma_decay=script_args.sigma_decay, sigma_min=script_args.sigma_min,
    num_continuations=script_args.num_continuations,
    save_every=script_args.es_save_every, output_dir=es_output_dir,
    verbose=script_args.es_verbose, seed=script_args.es_seed,
    convergence_window=script_args.convergence_window,
    parent_stability_bonus=script_args.parent_stability_bonus,
    parent_stability_cap=script_args.parent_stability_cap,
    use_dual_front=script_args.use_dual_front,
    algorithm=script_args.algorithm,
    normalize_rewards=script_args.normalize_rewards,
    normalize_fitness=script_args.normalize_fitness,
)

is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
if is_main:
    print(f'\nSaving final Stage 2 population to {es_output_dir}/final …')
    final_dir = os.path.join(es_output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    for i, params in enumerate(final_population):
        subdir = os.path.join(final_dir, f'ind_{i:03d}')
        net    = params_to_net(params, template_net, 'cpu')
        _save_simple_gating_network(net, subdir)
        with open(os.path.join(subdir, 'fitness.json'), 'w') as f:
            json.dump({'fitness': nsgaii.fitness[i].tolist()}, f, indent=2)

    fronts = _non_dominated_sort(np.array(nsgaii.fitness))
    with open(os.path.join(es_output_dir, 'nsgaii_meta.json'), 'w') as f:
        json.dump({
            'reward_names':    reward_names,
            'z_star':          nsgaii.z_star.tolist(),
            'final_fitness':   [f.tolist() for f in nsgaii.fitness],
            'num_generations': script_args.num_generations,
            'population_size': nsgaii.P,
            'lm_hidden_size':  lm_hidden_size,
        }, f, indent=2)

    print(f'\nDone. Non-dominated front: {len(fronts[0])}/{nsgaii.P} individuals')
    print(f'Final ideal point z* = {np.round(nsgaii.z_star, 4)}')
