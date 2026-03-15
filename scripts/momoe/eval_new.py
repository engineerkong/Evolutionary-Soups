"""Step 4: Evaluate the trained GatingNetwork.
For each (prompt, preference), predict merging weights, merge models, generate response,
and score rewards. Follows eval_ppo_rs pattern: rank0 merges+saves, all ranks load+eval.
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
from transformers import (AutoModelForCausalLM, DataCollatorWithPadding,
                          HfArgumentParser)
from trl import set_seed

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_eval_ppo, build_dataset_summary_eval_ppo,
    get_clean_data, load_main_tokenizer, save_configs,
    sample_preferences_uniform,
)
from new_architecture import GatingNetwork
from new_utils import (compute_hypervolume, load_base_model,
                       load_gating_network, merge_and_save_weights)

REWARD_PATHS = {
    'harmless': 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':  'Tristan/gpt2_reward_summarization',
    'faithful': 'CogComp/bart-faithful-summary-detector',
    'humor':    'mohameddhiab/humor-no-humor',
}


@dataclass
class ScriptArguments:
    sft_model_name: str = './models/sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = field(default='')
    manual_expert_weights: Optional[str] = field(
        default='0.5,0.5',
        metadata={'help': 'fallback weights if no checkpoint, e.g. "0.5,0.5"'})
    num_pref_samples: int = 10
    reward_names: str = 'harmless,helpful'
    exp_type: str = 'assistant'
    save_directory: str = './results/new/'
    wandb_name: str = 'new_assistant_eval'
    hidden_dim: int = 256


def parse_manual_weights(spec, num_experts):
    w = [float(v.strip()) for v in spec.split(',') if v.strip()]
    if len(w) != num_experts:
        raise ValueError(f'Expected {num_experts} weights, got {len(w)}')
    s = sum(w)
    return [v / s for v in w]


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index
gpu_id = process_id

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_experts = len(reward_names)
reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models = RewardModels(reward_model_paths, reward_model_paths, gpu_id)

save_configs({'sft_model_name': script_args.sft_model_name,
              'expert_model_paths': str(script_args.expert_model_paths)}, output_dir)

tokenizer = load_main_tokenizer(script_args.sft_model_name)

# Load gating network
if script_args.checkpoint_path:
    # Need lm_hidden_size — infer from expert model
    _tmp = load_base_model(script_args.expert_model_paths[0], target_device=f'cuda:{gpu_id}')
    with torch.no_grad():
        _dummy = tokenizer('hello', return_tensors='pt').to(f'cuda:{gpu_id}')
        _out = _tmp(**_dummy, output_hidden_states=True)
        lm_hidden_size = _out.hidden_states[-1].shape[-1]
    del _tmp
    gc.collect()
    torch.cuda.empty_cache()

    gating_net = load_gating_network(
        script_args.checkpoint_path,
        lm_hidden_size=lm_hidden_size,
        num_experts=num_experts,
        device=f'cuda:{gpu_id}',
    )
    if gating_net is None:
        print('Warning: could not load gating network, falling back to manual weights')
        gating_net = None
else:
    gating_net = None

if gating_net is None:
    manual_weights = parse_manual_weights(script_args.manual_expert_weights, num_experts)
    print(f'Using manual_expert_weights={manual_weights}')

# Load frozen expert models for prompt encoding (only needed if gating_net loaded)
expert_models = []
if gating_net is not None:
    for path in script_args.expert_model_paths:
        m = load_base_model(path, target_device=f'cuda:{gpu_id}')
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        expert_models.append(m)

# Dataset
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_ppo(
        'Anthropic/hh-rlhf', tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_ppo(
        'openai/summarize_from_feedback', tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

print(f'Eval dataset size: {len(valid_dataset)}')

sampled_preferences = sample_preferences_uniform(num_experts, script_args.num_pref_samples)


def get_prompt_hidden(expert_models, input_ids, attention_mask):
    all_hidden = []
    with torch.no_grad():
        for model in expert_models:
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
            h = out.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1)
            all_hidden.append(pooled)
    return torch.stack(all_hidden).mean(0)


def predict_weights(preference, input_ids, attention_mask):
    """Return expert weights for this batch. Uses gating_net if available."""
    if gating_net is not None:
        pref_t = torch.tensor(preference, dtype=torch.float32,
                              device=f'cuda:{gpu_id}').unsqueeze(0).expand(input_ids.shape[0], -1)
        hidden = get_prompt_hidden(expert_models, input_ids, attention_mask)
        with torch.no_grad():
            weights = gating_net(hidden, pref_t)   # (B, num_experts)
        # For merging we use the batch-mean weights (one merged model per preference)
        return weights.mean(dim=0).cpu().tolist()
    else:
        return manual_weights


def evaluate_model(temp_save_path):
    """Load merged model, generate responses, score rewards. Identical to eval_ppo_rs."""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(valid_dataset, 8, drop_last=True, collate_fn=data_collator)
    model = AutoModelForCausalLM.from_pretrained(
        temp_save_path, torch_dtype=torch.bfloat16, device_map=gpu_id)
    model.resize_token_embeddings(len(tokenizer))
    _accelerator = Accelerator()
    model, loader = _accelerator.prepare(model, loader)

    generation_kwargs = {
        'max_new_tokens': 128 if script_args.exp_type == 'assistant' else 48,
        'min_length': -1, 'top_k': 0.0, 'top_p': 0.9, 'do_sample': False,
    }
    tokenizer.padding_side = 'left'

    full_responses, full_prompts = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Generating', leave=False):
            out = _accelerator.unwrap_model(model).generate(
                batch['input_ids'], attention_mask=batch['attention_mask'],
                **generation_kwargs)
            full_responses.extend(out)
            full_prompts.extend(batch['input_ids'])

    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts   = tokenizer.batch_decode(full_prompts)
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    qr = [(instructions.get_input(r), instructions.get_response(r)) for r in full_responses]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(
            qr, instructions.get_post, normalize_rewards=False)
    else:
        rewards_list = reward_models.get_reward_model_scores(qr, normalize_rewards=False)

    all_rewards   = [_accelerator.gather_for_metrics(r) for r in rewards_list]
    all_prompts   = _accelerator.gather_for_metrics(full_prompts)
    all_responses = _accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_prompts, all_responses


# ========== Main eval loop ==========
all_results = []

for k, preference in enumerate(sampled_preferences):
    print(f'\nPref {k+1}/{len(sampled_preferences)}: {[round(p,2) for p in preference]}')

    # Predict weights (use mean over a small probe batch if gating_net is loaded)
    if gating_net is not None:
        # Use the first batch of the dataset as a representative probe
        probe_loader = DataLoader(
            valid_dataset.select(range(min(32, len(valid_dataset)))),
            batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
        probe_batch = next(iter(probe_loader))
        probe_ids   = probe_batch['input_ids'].to(f'cuda:{gpu_id}')
        probe_mask  = probe_batch['attention_mask'].to(f'cuda:{gpu_id}')
        expert_weights = predict_weights(preference, probe_ids, probe_mask)
    else:
        expert_weights = manual_weights

    print(f'  expert_weights={[round(w,3) for w in expert_weights]}')

    temp_path = os.path.join(output_dir,
        f'temp_model_pref_{"_".join([str(round(p,2)) for p in preference])}_{k}')

    if process_id == 0:
        merge_and_save_weights(script_args.expert_model_paths, expert_weights, temp_path)

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()

    all_rewards, all_prompts, all_responses = evaluate_model(temp_path)
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        result = {
            'prompt': all_prompts,
            'response': all_responses,
            'preference': [preference] * len(all_prompts),
            'expert_weights': [expert_weights] * len(all_prompts),
        }
        for i, name in enumerate(reward_names):
            result[f'reward_{name}'] = all_rewards[i]
            print(f'  mean {name} reward: {np.mean(all_rewards[i]):.4f}')

        pd.DataFrame(result).to_csv(
            os.path.join(output_dir,
                f'eval_pref_{"_".join([str(round(p,2)) for p in preference])}.csv'),
            escapechar='\\')

        all_results.append({
            'pref_idx': k,
            **{f'pref_{reward_names[j]}': preference[j] for j in range(num_experts)},
            **{f'expert_w{j}': expert_weights[j] for j in range(num_experts)},
            **{f'mean_reward_{reward_names[i]}': float(np.mean(all_rewards[i]))
               for i in range(len(reward_names))},
        })

        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)

if process_id == 0:
    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(output_dir, 'eval_summary.csv'), index=False)
    print('\nEvaluation complete. Summary:')
    print(summary[[c for c in summary.columns if 'mean_reward' in c or 'pref_' in c]].to_string())
