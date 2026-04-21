"""HoE training via preference-conditioned PPO. (MyMORLHF-style in HoE)

Trains the router layers and value heads of a pre-initialized MoLA/HoE model using PPO.
Expert LoRA weights are frozen throughout. Rewards are blended per batch as:

    r_blend = blend_ratio * r_preference + (1 - blend_ratio) * r_uniform

where r_preference is the dot product of the sampled preference vector with the
per-objective rewards, and r_uniform uses equal weights. This encourages both
preference-specific and general capabilities.

Preference weights are sampled from a bimodal distribution (peaks at 0 and 1,
mixed with uniform) via WeightSampler. Only router params and value heads are updated.
"""
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from peft import PeftModel as _StdPeft
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import set_seed

script_dir = Path(__file__).resolve().parent          # .../MOMoE/scripts/hoe
project_root = script_dir.parent.parent               # .../MOMoE
workspace_root = project_root.parent                  # .../MOMOE (contains anonymous-repo-for-HoE)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

HoE_ROOT = workspace_root / 'anonymous-repo-for-HoE' / 'Code' / 'HoE'
for _p in [str(HoE_ROOT / 'src'), str(HoE_ROOT), str(HoE_ROOT / 'trl_utils')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trl_utils.trainer import PPOConfig
from hoe_architecture import ConditionedMOEModelWithValueHead, WeightSampler
from hoe_utils import (
    build_hoe_model,
    clean_response,
    make_number_experts_str,
    parse_comma_int_list,
    parse_comma_str_list,
    parse_comma_str_list as _parse_paths,
)
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions,
    Instructions_summary,
    build_dataset_ppo,
    build_dataset_summary_ppo,
    build_dataset_news_summary_ppo,
    build_dataset_beaver_ppo,
    build_dataset_steer_ppo,
    load_config,
    load_main_tokenizer,
)

_ARMORM = 'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1'
REWARD_PATHS = {
    'harmless':          'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':           'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':           'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':           'Tristan/gpt2_reward_summarization',
    'faithful':          'CogComp/bart-faithful-summary-detector',
    'humor':             'mohameddhiab/humor-no-humor',
    'beaver_reward':     'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':       'PKU-Alignment/beaver-7b-v1.0-cost',
    'steer_helpfulness': f'{_ARMORM}#0',
    'steer_correctness': f'{_ARMORM}#1',
    'steer_coherence':   f'{_ARMORM}#2',
    'steer_complexity':  f'{_ARMORM}#3',
    'steer_verbosity':   f'{_ARMORM}#4',
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
    sft_model_name: str = field(default='', metadata={'help': 'Path to the SFT LoRA adapter (tokenizer lives here too)'})
    expert_model_paths: str = field(default='', metadata={'help': 'Comma-separated paths to per-objective PPO LoRA adapters, one per reward (order must match reward_names)'})
    dataset_name: str = field(default='Anthropic/hh-rlhf', metadata={'help': 'Dataset: Anthropic/hh-rlhf, openai/summarize_from_feedback, argilla/news-summary, PKU-Alignment/PKU-SafeRLHF-10K, nvidia/HelpSteer, nvidia/HelpSteer2'})
    reward_names: str = field(default='', metadata={'help': 'Comma-separated reward names; auto-selected from dataset_name if empty'})
    save_directory: str = field(default='./results/hoe/', metadata={'help': 'Output directory'})
    wandb_name: str = field(default='hoe_assistant', metadata={'help': 'Experiment name'})
    log_with: str = field(default='none', metadata={'help': "'wandb' to enable W&B logging"})
    n_layers: int = field(default=32, metadata={'help': 'Number of LLaMA transformer layers'})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config(script_dir / 'config.yaml')
print(f'Script arguments: {script_args}')
print(f'Training config: {cfg}')

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

if script_args.log_with != 'wandb':
    os.environ['WANDB_DISABLED'] = 'true'

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
reward_models = RewardModels(reward_model_paths, reward_model_paths, gpu_id)
_first_non_armorm = next((p for p in reward_model_paths if not p.startswith('armorm://')), reward_model_paths[0])
rm_tokenizer = AutoTokenizer.from_pretrained(_first_non_armorm)

# ========== build HoE model ==========
expert_paths = _parse_paths(script_args.expert_model_paths)
n_experts = len(expert_paths)
experts_str = make_number_experts_str(n_experts, script_args.n_layers)
number_experts = parse_comma_int_list(experts_str)
top_k = number_experts[:]

hoe_base = build_hoe_model(
    base_model_name=script_args.base_model_name,
    sft_model_name=script_args.sft_model_name,
    expert_model_paths=expert_paths,
    number_experts=number_experts,
    top_k=top_k,
    router_type=cfg.get('router_version', 'v1'),
    num_rewards=num_rewards,
    router_hidden_dim=int(cfg.get('router_hidden_dim', 32)),
)

# trainable_module="all" makes every param name match ('a'/'l' appear in almost all LLaMA names)
# matching Mymorlhf.py which passes trainable_module="all" to train everything
moemodel = ConditionedMOEModelWithValueHead(
    model=hoe_base, num_rewards=num_rewards, trainable_module="all"
)

# ========== reference model (frozen SFT for KL penalty) ==========
# Mirrors ppo.py: load base then apply SFT LoRA adapter
ref_model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name,
    torch_dtype=torch.bfloat16,
)
ref_model = _StdPeft.from_pretrained(ref_model, script_args.sft_model_name)
ref_model = ref_model.merge_and_unload()
ref_model.config.update({'use_cache': True, 'pad_token_id': ref_model.config.eos_token_id})

# ========== dataset ==========
# Tokenizer lives in the SFT adapter directory (same as ppo.py sft_model_name)
tokenizer = load_main_tokenizer(script_args.sft_model_name)
if script_args.dataset_name == 'Anthropic/hh-rlhf':
    dataset = build_dataset_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    dataset = build_dataset_summary_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'argilla/news-summary':
    dataset = build_dataset_news_summary_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='test')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    dataset = build_dataset_beaver_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    dataset = build_dataset_steer_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}')
print(f'Dataset size: {len(dataset)}')
# numpy format avoids the pyarrow DLPack readonly-buffer error with torch>=2.8
dataset = dataset.with_format("numpy")


def collator(data):
    result = {}
    for k in data[0]:
        vals = [d[k] for d in data]
        v0 = vals[0]
        if hasattr(v0, 'dtype') and np.issubdtype(v0.dtype, np.integer):
            result[k] = [torch.tensor(v) for v in vals]
        else:
            result[k] = [v.item() if hasattr(v, 'item') else v for v in vals]
    return result


# ========== PPO config ==========
ppo_config = PPOConfig(
    model_name=script_args.wandb_name,
    learning_rate=float(cfg['learning_rate']),
    log_with=script_args.log_with if script_args.log_with != 'none' else None,
    mini_batch_size=int(cfg['mini_batch_size']),
    batch_size=int(cfg['batch_size']),
    gradient_accumulation_steps=int(cfg['gradient_accumulation_steps']),
    early_stopping=bool(cfg.get('early_stopping', True)),
    target=float(cfg.get('target_kl', 9.0)),
    max_grad_norm=float(cfg.get('max_grad_norm', 0.5)),
    optimize_cuda_cache=True,
    init_kl_coef=float(cfg.get('init_kl_coef', 0.2)),
    kl_penalty=cfg.get('kl_penalty', 'abs'),
    tracker_project_name='hoe_training',
    tracker_kwargs={'wandb': {'name': script_args.wandb_name}},
    ppo_epochs=int(cfg.get('ppo_epochs', 3)),
)

from trl_utils.trainer.My_ppo_trainer import MyPPOTrainer

from peft import prepare_model_for_int8_training
if not cfg.get('load_8bit', False):
    # called even without 8-bit to enable gradient checkpointing hooks, matching Mymorlhf.py
    moemodel.model = prepare_model_for_int8_training(moemodel.model)

ref_model.resize_token_embeddings(len(tokenizer))
moemodel.model.resize_token_embeddings(len(tokenizer))
moemodel.train()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, moemodel.parameters()),
    lr=ppo_config.learning_rate,
)

ppo_trainer = MyPPOTrainer(
    ppo_config, moemodel, ref_model=ref_model, tokenizer=tokenizer,
    dataset=dataset, data_collator=collator, optimizer=optimizer,
)
weight_sampler = WeightSampler(num_rewards, uniform_ratio=float(cfg.get('uniform_ratio', 1.0)))

_long_response_datasets = {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K', 'nvidia/HelpSteer', 'nvidia/HelpSteer2'}
generation_kwargs = {
    'max_new_tokens': 128 if script_args.dataset_name in _long_response_datasets else 48,
    'min_length': -1,
    'do_sample': False,
    'num_beams': 1,
    'repetition_penalty': 1.2,
}

# ========== training loop ==========
blend_ratio = float(cfg.get('preference_blend_ratio', 0.8))
epochs = int(cfg.get('epochs', 1))
batch_size = int(cfg['batch_size'])
uniform_w = 1.0 / num_rewards

metrics = {k: [] for k in [
    'kl_mean', 'reward_mean', 'reward_std', 'batch_time', 'text_sample', 'weights', 'val_error',
    *[f'reward{k}_mean' for k in range(num_rewards)],
    *[f'reward{k}_sample' for k in range(num_rewards)],
    *[f'reward{k}_partition' for k in range(num_rewards)],
]}

print('Training...')
moemodel.test_grad()

for epoch in range(epochs):
    pbar = tqdm(total=len(dataset) // batch_size // accelerator.num_processes)

    for i, batch in enumerate(ppo_trainer.dataloader):
        t0 = time.time()
        query_tensors = batch['input_ids']

        # sample preference weights for this batch
        weights = weight_sampler.generate_weights(batch_size)

        moemodel.model.config.use_cache = True
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                query_tensors, weights, batch_size=8, return_prompt=False, **generation_kwargs
            )

        full_responses = tokenizer.batch_decode(response_tensors)
        clean_texts = [clean_response(r) for r in full_responses]
        clean_ids = [tokenizer.encode(t) for t in clean_texts]
        lengths = [len(ids) for ids in clean_ids]
        response_tensors = [response_tensors[j][:max(lengths[j], 2)] for j in range(len(response_tensors))]
        batch['response'] = clean_texts

        # ========== compute multi-objective rewards ==========
        texts_merge = [q + r for q, r in zip(batch['query'], batch['response'])]
        queries_responses = [
            (instructions.get_input(t), instructions.get_response(t)) for t in texts_merge
        ]
        if hasattr(instructions, 'get_post'):
            rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post, normalize_rewards=False) # no normalization during training as well to keep reward scales consistent
        else:
            rewards_list = reward_models.get_reward_model_scores(queries_responses, normalize_rewards=False)

        # blend: blend_ratio * preference-weighted + (1-blend_ratio) * uniform
        rewards = []
        for j in range(len(queries_responses)):
            r_pref = sum(weights[j][k] * rewards_list[k][j] for k in range(num_rewards))
            r_unif = sum(uniform_w * rewards_list[k][j] for k in range(num_rewards))
            rewards.append(blend_ratio * r_pref + (1.0 - blend_ratio) * r_unif)
        rewards_tensor = [r.detach().clone().to(gpu_id) if isinstance(r, torch.Tensor) else torch.tensor(float(r)).to(gpu_id) for r in rewards]

        print(f'Epoch {epoch} | Batch {i} | mean reward: {torch.mean(torch.tensor(rewards)).item():.4f}')

        # ========== PPO update ==========
        moemodel.model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, weights, response_tensors, rewards_tensor)

        # ========== gather metrics across GPUs ==========
        rewards_t = torch.stack(rewards).to('cuda')
        kl_t = torch.tensor([float(np.asarray(stats['objective/kl']).flat[0])]).to('cuda')
        val_err_t = torch.tensor([float(np.asarray(stats['ppo/val/error']).flat[0])]).to('cuda')
        multi_r_t = torch.stack([torch.tensor(r) for r in rewards_list]).T.to('cuda')
        weights_t = torch.stack(weights).to('cuda')

        all_rewards = accelerator.gather_for_metrics(rewards_t).cpu().numpy()
        all_kl = accelerator.gather_for_metrics(kl_t).cpu().numpy()
        all_val_err = accelerator.gather_for_metrics(val_err_t).cpu().numpy()
        all_multi_r = accelerator.gather_for_metrics(multi_r_t).cpu()
        all_weights = accelerator.gather_for_metrics(weights_t).cpu()

        if process_id == 0:
            mean_multi_r = all_multi_r.numpy().mean(axis=0)
            reward_partition = np.mean(
                np.einsum('bn,bn->bn', all_weights.numpy(), all_multi_r.numpy()), axis=0
            )
            metrics['kl_mean'].append(float(np.mean(all_kl)))
            metrics['reward_mean'].append(float(np.mean(all_rewards)))
            metrics['reward_std'].append(float(np.std(all_rewards)))
            metrics['batch_time'].append(time.time() - t0)
            metrics['text_sample'].append(batch['response'][:3])
            metrics['weights'].append(torch.stack(weights[:3]).cpu().numpy().tolist())
            metrics['val_error'].append(float(np.mean(all_val_err)))
            for k in range(num_rewards):
                metrics[f'reward{k}_mean'].append(float(mean_multi_r[k]))
                metrics[f'reward{k}_sample'].append(np.array(rewards_list)[k, :3].tolist())
                metrics[f'reward{k}_partition'].append(float(reward_partition[k]))

            pd.DataFrame(metrics).to_csv(os.path.join(output_dir, 'data.csv'))

        accelerator.wait_for_everyone()
        pbar.update(1)

        if ppo_trainer.accelerator.is_main_process and i % 5 == 0 and i > 0:
            ckpt = os.path.join(output_dir, f'epoch_{epoch}_batch_{i}')
            ppo_trainer.save_pretrained(ckpt)
            print(f'Checkpoint saved to {ckpt}')

    pbar.close()

    if ppo_trainer.accelerator.is_main_process:
        ckpt = os.path.join(output_dir, f'epoch_{epoch}_final')
        ppo_trainer.save_pretrained(ckpt)
        print(f'Epoch {epoch} complete. Final checkpoint saved to {ckpt}')
