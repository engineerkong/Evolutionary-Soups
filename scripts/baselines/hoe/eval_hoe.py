"""HoE evaluation across the preference simplex.

For each preference vector on a uniform simplex grid, sets the model's DynamicWeights
and generates responses with the trained HoE router. Evaluates per-objective rewards
and saves one CSV per preference point plus an overall summary CSV.

Output files:
    eval_pref{w0}_{w1}.csv    per-preference generation + reward data
    eval_summary.csv          per-preference mean rewards + hypervolume indicator
"""
import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

script_dir = Path(__file__).resolve().parent          # .../MOMoE/scripts/hoe
project_root = script_dir.parent.parent               # .../MOMoE
workspace_root = project_root.parent                  # .../MOMOE (contains anonymous-repo-for-HoE)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

HoE_ROOT = workspace_root / 'anonymous-repo-for-HoE' / 'Code' / 'HoE'
for _p in [str(HoE_ROOT / 'src'), str(HoE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.baselines.hoe.hoe_utils import (
    build_hoe_model,
    load_hoe_checkpoint,
    load_router_weights,
    make_number_experts_str,
    parse_comma_int_list,
    parse_comma_str_list as _parse_paths,
)
from scripts.baselines.utils.multi_reward_models import RewardModels
from scripts.baselines.utils.utils import (
    Instructions,
    Instructions_summary,
    build_dataset_eval,
    build_dataset_summary_eval,
    build_dataset_news_summary_ppo,
    build_dataset_beaver_eval,
    build_dataset_steer_eval,
    get_clean_data,
    load_config,
    load_main_tokenizer,
    sample_preferences_uniform,
)

os.environ['WANDB_DISABLED'] = 'true'

_URM = 'urm://LxzGordon/URM-LLaMa-3.1-8B'
REWARD_PATHS = {
    'harmless':          'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':           'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':           'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':           'Tristan/gpt2_reward_summarization',
    'faithful':          'CogComp/bart-faithful-summary-detector',
    'humor':             'mohameddhiab/humor-no-humor',
    'beaver_reward':     'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':       'PKU-Alignment/beaver-7b-v1.0-cost',
    'steer_helpfulness': f'{_URM}#0',
    'steer_correctness': f'{_URM}#1',
    'steer_coherence':   f'{_URM}#2',
    'steer_complexity':  f'{_URM}#3',
    'steer_verbosity':   f'{_URM}#4',
}

_DEFAULT_REWARD_NAMES = {
    'Anthropic/hh-rlhf':                    'harmless,helpful',
    'openai/summarize_from_feedback':        'summary,faithful',
    'argilla/news-summary':                  'summary,faithful',
    'PKU-Alignment/PKU-SafeRLHF-10K':        'beaver_reward,beaver_cost',
    'nvidia/HelpSteer':                      'steer_helpfulness,steer_correctness',
    'nvidia/HelpSteer2':                     'steer_helpfulness,steer_correctness',
}


@dataclass
class ScriptArguments:
    base_model_name: str = field(default='meta-llama/Llama-2-7b-hf', metadata={'help': 'Local path or HF id of the base LLaMA model'})
    expert_model_paths: str = field(default='', metadata={'help': 'Comma-separated paths to per-objective PPO LoRA adapters, one per reward (order must match reward_names)'})
    checkpoint_path: str = field(default='', metadata={'help': 'Path to Stage-2 trained HoE checkpoint directory (contains adapter_model.bin)'})
    pretrained_moe_path: str = field(default='', metadata={'help': 'Path to Stage-1 router checkpoint directory (contains router_weights.pt). Use this to evaluate the router before Stage-2 PPO.'})
    dataset_name: str = field(default='Anthropic/hh-rlhf', metadata={'help': 'Dataset: Anthropic/hh-rlhf, openai/summarize_from_feedback, argilla/news-summary, PKU-Alignment/PKU-SafeRLHF-10K, nvidia/HelpSteer, nvidia/HelpSteer2'})
    reward_names: str = field(default='', metadata={'help': 'Comma-separated reward names; auto-selected from dataset_name if empty'})
    num_pref_samples: int = field(default=11, metadata={'help': 'Number of preference points to evaluate'})
    num_eval_samples: int = field(default=0, metadata={'help': 'Limit eval dataset size (0 = all)'})
    save_directory: str = field(default='./results/hoe/', metadata={'help': 'Output directory'})
    wandb_name: str = field(default='hoe_assistant_eval', metadata={'help': 'Experiment name'})
    batch_size: int = field(default=32, metadata={'help': 'Evaluation batch size'})
    n_layers: int = field(default=32, metadata={'help': 'Number of LLaMA transformer layers'})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config(script_dir / 'config.yaml')
print(f'Script arguments: {script_args}')

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id
print(f'Process {process_id} | GPU {gpu_id}')

# ========== reward models ==========
_rn_str = script_args.reward_names.strip() or _DEFAULT_REWARD_NAMES.get(script_args.dataset_name, 'harmless,helpful')
reward_names = [x.strip() for x in _rn_str.split(',')]
num_rewards = len(reward_names)
if any(n not in REWARD_PATHS for n in reward_names):
    raise ValueError(f'Unknown reward name(s): {[n for n in reward_names if n not in REWARD_PATHS]}')
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(reward_model_paths, reward_model_paths, 'cpu')

# ========== build and load HoE model ==========
expert_paths = _parse_paths(script_args.expert_model_paths)
n_experts = len(expert_paths)
experts_str = make_number_experts_str(n_experts, script_args.n_layers)
number_experts = parse_comma_int_list(experts_str)
top_k = number_experts[:]

model = build_hoe_model(
    base_model_name=script_args.base_model_name,
    expert_model_paths=expert_paths,
    number_experts=number_experts,
    top_k=top_k,
    router_type=cfg.get('router_version', 'v1'),
    num_rewards=num_rewards,
    router_hidden_dim=int(cfg.get('router_hidden_dim', 32)),
)

if script_args.pretrained_moe_path:
    # Stage-1 router checkpoint (router_weights.pt from train_hoe_router.py)
    load_router_weights(model, script_args.pretrained_moe_path)
elif script_args.checkpoint_path:
    # Stage-2 PPO checkpoint (adapter_model.bin from train_hoe.py)
    load_hoe_checkpoint(model, script_args.checkpoint_path)

model.eval()

# ========== dataset and dataloader ==========
tokenizer = load_main_tokenizer(expert_paths[0])
tokenizer.padding_side = 'left'

if script_args.dataset_name == 'Anthropic/hh-rlhf':
    valid_dataset = build_dataset_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    valid_dataset = build_dataset_summary_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'argilla/news-summary':
    valid_dataset = build_dataset_news_summary_ppo(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    valid_dataset = build_dataset_beaver_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    valid_dataset = build_dataset_steer_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}')

if script_args.num_eval_samples > 0:
    valid_dataset = valid_dataset.select(
        range(min(script_args.num_eval_samples, len(valid_dataset))))

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

print(f'Evaluation dataset size: {len(valid_dataset)}')
# numpy format avoids pyarrow DLPack readonly-buffer error with torch>=2.8
valid_dataset = valid_dataset.with_format("numpy")
model.resize_token_embeddings(len(tokenizer))
model = accelerator.prepare(model)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
eval_dataloader = DataLoader(
    valid_dataset, batch_size=script_args.batch_size, drop_last=False, collate_fn=data_collator
)
eval_dataloader = accelerator.prepare(eval_dataloader)

# ========== preference grid ==========
preferences = sample_preferences_uniform(num_rewards, script_args.num_pref_samples)
print(f'\n{len(preferences)} preferences to evaluate:')
for p in preferences:
    pref_str = ', '.join(f'{reward_names[k]}={p[k]:.2f}' for k in range(num_rewards))
    print(f'  [{pref_str}]')

_long_response_datasets = {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K', 'nvidia/HelpSteer', 'nvidia/HelpSteer2'}
generation_kwargs = {
    'max_new_tokens': 128 if script_args.dataset_name in _long_response_datasets else 48,
    'min_length': -1,
    'do_sample': False,
}

# ========== evaluation loop ==========
print('\nEvaluating...')
summary_rows = []

for pref in preferences:
    pref_tensor = torch.tensor(pref, dtype=torch.bfloat16).unsqueeze(0)
    accelerator.unwrap_model(model).dynamic_weights.set_dynamic_weights(pref_tensor, batch_size=1)

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

    pref_str = ', '.join(f'{reward_names[k]}={pref[k]:.2f}' for k in range(num_rewards))
    print(f'\nPreference: [{pref_str}]')

    full_responses, full_prompts = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in tqdm(eval_dataloader, desc=f'pref={[round(p, 2) for p in pref]}'):
            input_ids = batch['input_ids'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            response_tensors = accelerator.unwrap_model(model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs
            )
            full_responses.extend(tokenizer.batch_decode(response_tensors))
            full_prompts.extend(tokenizer.batch_decode(input_ids))

    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)
    queries_responses = [
        (instructions.get_input(r), instructions.get_response(r)) for r in full_responses
    ]

    reward_models.to_device(f'cuda:{gpu_id}')
    rewards_list = reward_models.get_reward_model_scores(
        queries_responses, getattr(instructions, 'get_post', None), normalize_rewards=False
    )
    reward_models.to_device('cpu')
    torch.cuda.empty_cache()

    all_rewards = [accelerator.gather_for_metrics(rewards_list[i]) for i in range(num_rewards)]
    all_prompts = accelerator.gather_for_metrics(full_prompts)
    all_responses = accelerator.gather_for_metrics(full_responses)

    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        result = {'prompt': all_prompts, 'response': all_responses}
        summary_row = {'type': 'preference'}
        for k, name in enumerate(reward_names):
            result[f'reward_{name}'] = all_rewards[k]
            mean_r = float(np.mean(all_rewards[k]))
            summary_row[f'pref_{name}'] = pref[k]
            summary_row[f'mean_reward_{name}'] = mean_r
            print(f'  {name}: {mean_r:.4f}')

        df = pd.DataFrame(result)
        if num_rewards == 2:
            fname = f'eval_pref{pref[0]:.1f}_{pref[1]:.1f}.csv'
        else:
            fname = 'eval_pref_' + '_'.join(f'{p:.1f}' for p in pref) + '.csv'
        df.to_csv(os.path.join(output_dir, fname), escapechar='\\', index=False)
        summary_rows.append(summary_row)

if process_id == 0:
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, 'eval_summary.csv'), index=False
    )
    print(f'\nResults saved to {output_dir}')
    print('  - eval_pref*.csv (per-preference generation + reward data)')
    print('  - eval_summary.csv (mean rewards per preference)')

# ---------------------------------------------------------------------------
# Rewarded Soups evaluation
# ---------------------------------------------------------------------------

def _load_adapter_sd(adapter_dir: str) -> dict:
    p = Path(adapter_dir)
    st_path = p / 'adapter_model.safetensors'
    bin_path = p / 'adapter_model.bin'
    if st_path.exists():
        from safetensors import safe_open
        sd = {}
        with safe_open(str(st_path), framework='pt', device='cpu') as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        return sd
    elif bin_path.exists():
        return torch.load(str(bin_path), map_location='cpu')
    raise FileNotFoundError(f'No adapter weights found in {adapter_dir}')


print('\n' + '=' * 60)
print('Rewarded Soups evaluation')
print('=' * 60)

expert_sds    = [_load_adapter_sd(p) for p in expert_paths]
peft_cfg_rs   = LoraConfig.from_pretrained(expert_paths[0])
rs_summary_rows = []

for pref in preferences:
    pref_str = ', '.join(f'{reward_names[k]}={pref[k]:.2f}' for k in range(num_rewards))
    print(f'\n[RS] Preference: [{pref_str}]')

    blended_sd = {
        k: sum(pref[i] * expert_sds[i][k].to(torch.bfloat16) for i in range(num_rewards))
        for k in expert_sds[0]
    }

    rs_base = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, torch_dtype=torch.bfloat16,
        device_map=f'cuda:{gpu_id}')
    rs_model = get_peft_model(rs_base, peft_cfg_rs)
    set_peft_model_state_dict(rs_model, blended_sd)
    rs_model = rs_model.merge_and_unload()
    rs_model.resize_token_embeddings(len(tokenizer))
    rs_model.eval()

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

    rs_responses, rs_prompts = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in tqdm(eval_dataloader, desc=f'RS pref={[round(p, 2) for p in pref]}'):
            input_ids = batch['input_ids'].to(f'cuda:{gpu_id}')
            attention_mask = batch['attention_mask'].to(f'cuda:{gpu_id}')
            rs_responses.extend(tokenizer.batch_decode(
                rs_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)))
            rs_prompts.extend(tokenizer.batch_decode(input_ids))

    rs_prompts, rs_responses = get_clean_data(rs_responses, rs_prompts)
    qr_rs = [(instructions.get_input(r), instructions.get_response(r)) for r in rs_responses]
    reward_models.to_device(f'cuda:{gpu_id}')
    rewards_rs = reward_models.get_reward_model_scores(
        qr_rs, getattr(instructions, 'get_post', None), normalize_rewards=False
    )
    reward_models.to_device('cpu')
    torch.cuda.empty_cache()

    all_rewards_rs  = [accelerator.gather_for_metrics(rewards_rs[i]) for i in range(num_rewards)]
    all_rs_prompts  = accelerator.gather_for_metrics(rs_prompts)
    all_rs_responses = accelerator.gather_for_metrics(rs_responses)

    del rs_model, rs_base, blended_sd
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        rs_row = {'type': 'rewarded_soups'}
        for k, name in enumerate(reward_names):
            mean_r = float(np.mean(all_rewards_rs[k]))
            rs_row[f'pref_{name}'] = pref[k]
            rs_row[f'mean_reward_{name}'] = mean_r
            print(f'  {name}: {mean_r:.4f}')
        rs_summary_rows.append(rs_row)

        result_rs = {'prompt': all_rs_prompts, 'response': all_rs_responses}
        for k, name in enumerate(reward_names):
            result_rs[f'reward_{name}'] = all_rewards_rs[k]
        df_rs = pd.DataFrame(result_rs)
        if num_rewards == 2:
            fname_rs = f'rs_pref{pref[0]:.1f}_{pref[1]:.1f}.csv'
        else:
            fname_rs = 'rs_pref_' + '_'.join(f'{p:.1f}' for p in pref) + '.csv'
        df_rs.to_csv(os.path.join(output_dir, fname_rs), escapechar='\\', index=False)

if process_id == 0:
    pd.DataFrame(rs_summary_rows).to_csv(
        os.path.join(output_dir, 'rs_summary.csv'), index=False)
    print(f'\nRewarded Soups results saved to {output_dir}')
    print('  - rs_pref*.csv (per-preference generation + reward data)')
    print('  - rs_summary.csv (mean rewards per preference)')
