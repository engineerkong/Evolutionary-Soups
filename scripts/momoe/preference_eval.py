"""preference_eval.py — Per-preference comparison across CIEA, Rewarded Soups, HOE, MORLHF.

For each preference λ (step=0.1 over the simplex):
  1. CIEA:          best individual from meta pool by argmax(λ · mean_fitness)
  2. RewardedSoups: expert LoRA adapters blended with weights λ
  3. HOE:           preference-conditioned model; set λ at inference time
  4. MORLHF:        model trained for preference λ

TO RUN (beaver example):
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./scripts/momoe/preference_eval.py \
    --expert_model_paths './models/ppo/ppo_beaver_reward_2204/best_model' \
                         './models/ppo/ppo_beaver_cost_2204/best_model' \
    --nsgaii_gating_dir  './models/nsgaii/nsgaii_beaver_0105/gen_0035/meta' \
    --hoe_path           './results/hoe/hoe_beaver_train_2304/model' \
    --morlhf_dir         './results/morlhf/' \
    --morlhf_prefix      'morlhf_beaver_2704_pref' \
    --reward_names       'beaver_reward,beaver_cost' \
    --dataset_name       'PKU-Alignment/PKU-SafeRLHF-10K' \
    --run_name           'preference_eval_0105' \
    2>&1 | tee ./logs/preference_eval_0105.log
"""

import datetime
import gc
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from torch.utils.data import DataLoader

script_dir      = Path(__file__).resolve().parent
project_root    = script_dir.parent.parent
workspace_root  = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(project_root / 'scripts' / 'hoe'))

# HOE uses hacked PEFT / MoLA modules from its own repo — must precede any peft import
_HoE_ROOT = workspace_root / 'anonymous-repo-for-HoE' / 'Code' / 'HoE'
for _p in [str(_HoE_ROOT / 'src'), str(_HoE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_beaver_eval, build_dataset_eval,
    build_dataset_summary_eval, build_dataset_steer_eval,
    build_dataset_ultrafeedback_eval,
    get_clean_data, load_main_tokenizer,
)
from nsgaii_architecture import GatingNetwork, MoEForCausalLM
from nsgaii_utils import REWARD_PATHS, load_gating_network, get_simplex_samples


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    base_model_name:    str       = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'beaver_reward,beaver_cost'
    dataset_name:       str       = 'PKU-Alignment/PKU-SafeRLHF-10K'
    eval_prompts:       int       = 0
    batch_size:         int       = 128
    num_continuations:  int       = 1
    pref_step:          float     = 0.1

    # CIEA / NSGAII meta pool
    nsgaii_gating_dir:  str       = ''   # dir with ind_XXX/ subdirs (each has fitness.json + gating_network.pt)

    # Rewarded Soups (uses expert_model_paths directly)

    # HOE
    hoe_path:           str       = ''   # trained HOE checkpoint dir (adapter_model.bin + v_head*.pt)
    hoe_n_layers:       int       = 32   # number of LLaMA transformer layers (32 for 7B)

    # MORLHF
    morlhf_dir:         str       = ''   # parent dir containing morlhf_*_pref*/ runs
    morlhf_prefix:      str       = ''   # e.g. 'morlhf_beaver_2704_pref'
    morlhf_subdir:      str       = 'best_model'   # sub-dir inside each pref run

    gpu_id:             int       = -1
    save_directory:     str       = './results/preference_eval/'
    run_name:           str       = 'preference_eval'
    seed:               int       = 8888


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',')]


def generate_and_score(model, input_ids, attention_mask, tokenizer,
                       reward_models, instructions, generation_kwargs,
                       gpu_id, num_continuations):
    device      = f'cuda:{gpu_id}'
    accumulated = None
    # HOE models are wrapped by accelerator; unwrap for direct generate call
    gen_model = accelerator.unwrap_model(model) if hasattr(model, 'module') else model
    for _ in range(num_continuations):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs     = gen_model.generate(input_ids.to(device),
                                             attention_mask=attention_mask.to(device),
                                             **generation_kwargs)
        responses       = tokenizer.batch_decode(outputs.cpu())
        prompts_decoded = tokenizer.batch_decode(input_ids.cpu())
        del outputs
        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(instructions.get_input(r), instructions.get_response(r))
                 for r in responses_clean]
        scores = (reward_models.get_reward_model_scores(pairs, instructions.get_post,
                      normalize_rewards=False)
                  if hasattr(instructions, 'get_post') else
                  reward_models.get_reward_model_scores(pairs, normalize_rewards=False))
        n_p, n_r = len(prompts_clean), len(scores)
        if accumulated is None:
            accumulated = [[[] for _ in range(n_r)] for _ in range(n_p)]
        for p in range(n_p):
            for k in range(n_r):
                accumulated[p][k].append(scores[k][p])
        torch.cuda.empty_cache()
    per_prompt = np.array([[np.mean(accumulated[p][k]) for k in range(n_r)]
                           for p in range(n_p)])
    return per_prompt.mean(axis=0).tolist()


def eval_model(model, loader, tokenizer, reward_models, instructions,
               generation_kwargs, gpu_id, num_continuations, label,
               cache_path=None, rank=0):
    """Evaluate one model over the full loader; return mean reward vector."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f'  rank{rank} cached: {label}  {cached["rewards"]}', flush=True)
        return cached['rewards']

    model.eval()
    batch_rewards = []
    for batch in loader:
        r = generate_and_score(model, batch['input_ids'], batch['attention_mask'],
                                tokenizer, reward_models, instructions,
                                generation_kwargs, gpu_id, num_continuations)
        batch_rewards.append(r)
    mean_r = np.mean(batch_rewards, axis=0).tolist()
    print(f'  rank{rank} done: {label}  {mean_r}', flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'label': label, 'rewards': mean_r}, f)
    return mean_r


def _load_adapter_sd(adapter_dir: str) -> dict:
    p = Path(adapter_dir)
    st = p / 'adapter_model.safetensors'
    bn = p / 'adapter_model.bin'
    if st.exists():
        from safetensors import safe_open
        sd = {}
        with safe_open(str(st), framework='pt', device='cpu') as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        return sd
    elif bn.exists():
        return torch.load(str(bn), map_location='cpu')
    raise FileNotFoundError(f'No adapter weights in {adapter_dir}')


def _build_rewarded_soup(base_model_name, expert_sds, peft_config, coeffs, device):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    blended = {k: sum(coeffs[i] * expert_sds[i][k].to(torch.bfloat16)
                      for i in range(len(coeffs)))
               for k in expert_sds[0]}
    m = get_peft_model(base, peft_config)
    set_peft_model_state_dict(m, blended)
    m = m.merge_and_unload()
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _morlhf_path(morlhf_dir, prefix, pref, subdir):
    """Resolve MORLHF model path for a given preference vector."""
    pref_str = '_'.join(f'{w:.1f}' for w in pref)
    path = os.path.join(morlhf_dir, f'{prefix}{pref_str}', subdir)
    if not os.path.isdir(path):
        return None
    return path


def _load_ciea_meta(gating_dir):
    """Load {ind_path: fitness} for all individuals in an NSGAII gen directory.

    Supports two fitness.json layouts:
      - newer runs: {"mean_fitness": [...], "chunk_fitness": {...}}
      - older runs: {"fitness": [...]}
    """
    meta = {}
    for ind_dir in sorted(glob.glob(os.path.join(gating_dir, 'ind_*'))):
        fit_path = os.path.join(ind_dir, 'fitness.json')
        if not os.path.exists(fit_path):
            continue
        with open(fit_path) as f:
            data = json.load(f)
        mf = data.get('mean_fitness') or data.get('fitness')
        if mf is not None:
            meta[ind_dir] = np.array(mf)
    return meta


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

args: Args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
torch.manual_seed(args.seed)
np.random.seed(args.seed)

output_dir = os.path.join(args.save_directory, args.run_name)
os.makedirs(output_dir, exist_ok=True)

if 'RANK' in os.environ:
    torch.distributed.init_process_group(
        backend='nccl', timeout=datetime.timedelta(minutes=240))
accelerator = Accelerator()
gpu_id     = args.gpu_id if args.gpu_id >= 0 else accelerator.local_process_index
rank       = accelerator.process_index
world_size = accelerator.num_processes
is_main    = accelerator.is_main_process
device     = f'cuda:{gpu_id}'

reward_names = [x.strip() for x in args.reward_names.split(',')]
M = len(reward_names)

tokenizer              = load_main_tokenizer(args.expert_model_paths[0])
tokenizer.padding_side = 'left'

_rm_paths     = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(_rm_paths, _rm_paths, gpu_id)

_max_new_tokens = 128 if args.dataset_name in {
    'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K',
    'nvidia/HelpSteer', 'nvidia/HelpSteer2'} else 48

# Per-method generation kwargs aligned with each method's original eval script
# CIEA / RS (nsgaii_test.py): greedy, no min_length
generation_kwargs_ciea = dict(max_new_tokens=_max_new_tokens, do_sample=False)
# HOE (eval_hoe.py): greedy + min_length=-1
generation_kwargs_hoe  = dict(max_new_tokens=_max_new_tokens, min_length=-1, do_sample=False)
# MORLHF (eval_morlhf.py): greedy + min_length=-1
generation_kwargs_morlhf = dict(max_new_tokens=_max_new_tokens, min_length=-1, do_sample=False)

# Dataset
if args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    ds = build_dataset_beaver_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions()
elif args.dataset_name == 'Anthropic/hh-rlhf':
    ds = build_dataset_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions()
elif args.dataset_name == 'openai/summarize_from_feedback':
    ds = build_dataset_summary_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions_summary()
elif args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    ds = build_dataset_steer_eval(
        args.dataset_name, tokenizer, reward_models.rm_tokenizers,
        split='test', size=args.eval_prompts or None)
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset: {args.dataset_name!r}')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in ds.column_names:
        ds = ds.remove_columns(key)
ds = ds.with_format('numpy')
loader = DataLoader(ds, batch_size=args.batch_size,
                    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
                    drop_last=False)
if is_main:
    print(f'{len(ds)} test prompts', flush=True)

# Preference simplex
preferences = get_simplex_samples(M, step=args.pref_step)
n_prefs = len(preferences)

# ---------------------------------------------------------------------------
# Load expert models (needed by CIEA MoE and Rewarded Soups)
# ---------------------------------------------------------------------------

if is_main:
    print(f'\nLoading {len(args.expert_model_paths)} expert models …', flush=True)
experts = []
for i, path in enumerate(args.expert_model_paths):
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, torch_dtype=torch.bfloat16, device_map=device)
    m = PeftModel.from_pretrained(base, path).merge_and_unload()
    m.resize_token_embeddings(len(tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    experts.append(m)
    if is_main:
        print(f'  [{i}] {path}', flush=True)
lm_hidden_size = experts[0].config.hidden_size

# ---------------------------------------------------------------------------
# Pre-select CIEA individuals per preference (no model loading yet)
# ---------------------------------------------------------------------------

ciea_meta = {}   # {ind_dir: mean_fitness}
if args.nsgaii_gating_dir:
    ciea_meta = _load_ciea_meta(args.nsgaii_gating_dir)
    if is_main:
        print(f'\nLoaded {len(ciea_meta)} CIEA meta pool members from {args.nsgaii_gating_dir}',
              flush=True)

ind_dirs  = list(ciea_meta.keys())
fitnesses = np.array(list(ciea_meta.values())) if ciea_meta else None

def best_ciea_for_pref(lam):
    if fitnesses is None:
        return None
    scores = fitnesses @ np.array(lam)
    return ind_dirs[int(np.argmax(scores))]

# ---------------------------------------------------------------------------
# HOE model — build once, reuse across preferences
# ---------------------------------------------------------------------------

hoe_model = None
if args.hoe_path:
    try:
        from hoe_utils import (build_hoe_model, load_hoe_checkpoint,
                                make_number_experts_str, parse_comma_int_list)
        if is_main:
            print(f'\nBuilding HOE model from {args.hoe_path} …', flush=True)
        _n_exp   = len(args.expert_model_paths)
        _exp_str = make_number_experts_str(_n_exp, args.hoe_n_layers)
        _num_exp = parse_comma_int_list(_exp_str)
        hoe_model = build_hoe_model(
            base_model_name=args.base_model_name,
            expert_model_paths=args.expert_model_paths,
            number_experts=_num_exp,
            top_k=_num_exp[:],
            num_rewards=M,
            device=device,
        )
        load_hoe_checkpoint(hoe_model, args.hoe_path)
        hoe_model.resize_token_embeddings(len(tokenizer))
        hoe_model.eval()
        hoe_model = accelerator.prepare(hoe_model)
        if is_main:
            print('  HOE model ready', flush=True)
    except Exception as e:
        if is_main:
            print(f'  [WARN] HOE model failed to load: {e}', flush=True)
        hoe_model = None

# Rewarded Soups adapter state dicts (loaded once)
rs_expert_sds  = None
rs_peft_config = None
if args.expert_model_paths:
    rs_expert_sds  = [_load_adapter_sd(p) for p in args.expert_model_paths]
    rs_peft_config = LoraConfig.from_pretrained(args.expert_model_paths[0])

# ---------------------------------------------------------------------------
# Per-preference evaluation (each rank handles a subset of preferences)
# ---------------------------------------------------------------------------

results = {}   # {pref_idx: {method: rewards}}
my_prefs = [i for i in range(n_prefs) if i % world_size == rank]

for pref_idx in my_prefs:
    lam  = preferences[pref_idx]
    lam_str = str(list(lam))
    results[pref_idx] = {}

    # ── 1. CIEA ──────────────────────────────────────────────────────────────
    if ciea_meta:
        ind_dir = best_ciea_for_pref(lam)
        ind_name = os.path.basename(ind_dir)
        cache_p = os.path.join(output_dir, 'cache', f'ciea_pref{pref_idx}.json')
        gating = load_gating_network(ind_dir, lm_hidden_size=lm_hidden_size,
                                     num_experts=len(experts), device=device)
        if gating is not None:
            moe = MoEForCausalLM(experts, gating).to(device)
            r = eval_model(moe, loader, tokenizer, reward_models, instructions,
                           generation_kwargs_ciea, gpu_id, args.num_continuations,
                           label=f'CIEA {lam_str} [{ind_name}]',
                           cache_path=cache_p, rank=rank)
            results[pref_idx]['ciea'] = {'rewards': r, 'ind': ind_name,
                                         'train_fitness': ciea_meta[ind_dir].tolist()}
            del moe
            gc.collect(); torch.cuda.empty_cache()

    # ── 2. Rewarded Soups ────────────────────────────────────────────────────
    if rs_expert_sds is not None:
        cache_p = os.path.join(output_dir, 'cache', f'rs_pref{pref_idx}.json')
        rs_model = _build_rewarded_soup(
            args.base_model_name, rs_expert_sds, rs_peft_config, lam, device)
        rs_model.resize_token_embeddings(len(tokenizer))
        r = eval_model(rs_model, loader, tokenizer, reward_models, instructions,
                       generation_kwargs_ciea, gpu_id, args.num_continuations,
                       label=f'RewardedSoups {lam_str}',
                       cache_path=cache_p, rank=rank)
        results[pref_idx]['rewarded_soups'] = {'rewards': r}
        del rs_model
        gc.collect(); torch.cuda.empty_cache()

    # ── 3. HOE ───────────────────────────────────────────────────────────────
    if hoe_model is not None:
        cache_p = os.path.join(output_dir, 'cache', f'hoe_pref{pref_idx}.json')
        lam_t = torch.tensor(lam, dtype=torch.bfloat16).unsqueeze(0).to(device)
        accelerator.unwrap_model(hoe_model).dynamic_weights.set_dynamic_weights(
            lam_t, batch_size=1)
        r = eval_model(hoe_model, loader, tokenizer, reward_models, instructions,
                       generation_kwargs_hoe, gpu_id, args.num_continuations,
                       label=f'HOE {lam_str}',
                       cache_path=cache_p, rank=rank)
        results[pref_idx]['hoe'] = {'rewards': r}

    # ── 4. MORLHF ────────────────────────────────────────────────────────────
    if args.morlhf_dir and args.morlhf_prefix:
        model_path = _morlhf_path(args.morlhf_dir, args.morlhf_prefix,
                                   lam, args.morlhf_subdir)
        if model_path:
            cache_p = os.path.join(output_dir, 'cache', f'morlhf_pref{pref_idx}.json')
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model_name, torch_dtype=torch.bfloat16, device_map=gpu_id)
            morlhf_m = PeftModel.from_pretrained(base, model_path).merge_and_unload()
            morlhf_m.resize_token_embeddings(len(tokenizer))
            morlhf_m.eval()
            for p in morlhf_m.parameters():
                p.requires_grad = False
            r = eval_model(morlhf_m, loader, tokenizer, reward_models, instructions,
                           generation_kwargs_morlhf, gpu_id, args.num_continuations,
                           label=f'MORLHF {lam_str}',
                           cache_path=cache_p, rank=rank)
            results[pref_idx]['morlhf'] = {'rewards': r, 'path': model_path}
            del morlhf_m
            gc.collect(); torch.cuda.empty_cache()
        else:
            if is_main:
                print(f'  [WARN] No MORLHF model found for pref {lam_str}', flush=True)

# ---------------------------------------------------------------------------
# Gather and print comparison table
# ---------------------------------------------------------------------------

if world_size > 1:
    all_results = [None] * world_size
    torch.distributed.all_gather_object(all_results, results)
else:
    all_results = [results]

if is_main:
    merged = {}
    for r in all_results:
        merged.update(r)

    out_path = os.path.join(output_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump({'preferences': [list(p) for p in preferences],
                   'reward_names': reward_names,
                   'results': {str(k): v for k, v in merged.items()}}, f, indent=2)
    print(f'\nResults saved → {out_path}', flush=True)

    methods = ['ciea', 'rewarded_soups', 'hoe', 'morlhf']
    method_labels = {'ciea': 'CIEA', 'rewarded_soups': 'RewardedSoups',
                     'hoe': 'HOE', 'morlhf': 'MORLHF'}
    rn_width = 10

    header = f'{"λ":<20}  {"method":<16}  ' + '  '.join(f'{rn:>{rn_width}}' for rn in reward_names)
    print('\n' + '=' * len(header))
    print(header)
    print('=' * len(header))

    for pref_idx in sorted(merged.keys()):
        lam = preferences[pref_idx]
        lam_str = str(list(lam))
        row = merged[pref_idx]
        first = True
        for method in methods:
            if method not in row:
                continue
            r = row[method]['rewards']
            ind_note = f'  [{row[method]["ind"]}]' if method == 'ciea' and 'ind' in row[method] else ''
            label = f'{method_labels[method]}{ind_note}'
            prefix = f'{lam_str:<20}' if first else ' ' * 20
            print(f'{prefix}  {label:<16}  ' + '  '.join(f'{v:>{rn_width}.4f}' for v in r))
            first = False
        print()
