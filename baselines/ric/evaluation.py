"""RiC evaluation — preference-conditioned generation across the Pareto front.

Adapted from RiC/ric/evaluation.py.

Changes vs original (RiC/ric/evaluation.py):
  1. exp_type='beaver': uses build_beaver_dataset_with_preference_n (new function
     in local utils.py) and Instructions_n (same format as assistant).
  2. reward_path_tokenizer_dict: extended with beaver_reward / beaver_cost entries.
  3. beaver_dataset_path: added as a module-level constant.
  4. score column removal in evaluate_model: changed from hard-coded ['score1','score2']
     to dynamic [c for c in columns if c.startswith('score')] so it works with any
     number of reward objectives (2 or 3).
  5. Multi-GPU: module-level Accelerator() created once; process_id uses
     accelerator.local_process_index. Inside evaluate_model a local reference
     is created (standard pattern for functions that call accelerator.prepare).
     The redundant second accelerator = Accelerator() before wait_for_everyone()
     was removed.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, DataCollatorWithPadding
from peft import PeftModel
from trl import set_seed
import numpy as np
import pandas as pd

from utils import get_clean_data, Instructions_n, build_dataset_with_preference_n, \
                  load_main_tokenizer, Instructions_summary_n
from multi_reward_models import RewardModels
tqdm.pandas()
from utils import save_configs, map_rewards_from_preference, build_summary_dataset_with_preference_n, \
                  clean_gpu_memory, build_beaver_dataset_with_preference_n

script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from baselines.utils.utils import sample_preferences_uniform

hhrlhf_dataset_path  = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
beaver_dataset_path  = 'PKU-Alignment/PKU-SafeRLHF-10K'


@dataclass
class ScriptArguments:
    peft_name:          Optional[str]   = field(default='')
    num_prefer_points:  Optional[int]   = field(default=10)
    log_with:           Optional[str]   = field(default='wandb')
    save_directory:     Optional[str]   = field(default='./results/ric/')
    wandb_name:         Optional[str]   = field(default='ric_eval')
    reward_names:       Optional[str]   = field(default='harmless,helpful')
    base_model_name:    Optional[str]   = field(default='meta-llama/Llama-2-7b-hf')
    reward_stats_path:  Optional[str]   = field(default='')
    exp_type:           Optional[str]   = field(default='assistant', metadata={"help": "assistant | summary | beaver"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type        = script_args.exp_type
base_model_name = script_args.base_model_name
tokenier_name   = script_args.base_model_name
reward_stats_path = script_args.reward_stats_path if len(script_args.reward_stats_path) else None
print('base model: ', base_model_name)

peft_name    = script_args.peft_name
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)

reward_path_tokenizer_dict = {
    'harmless':     ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful':      ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta':      ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary':      ['Tristan/gpt2_reward_summarization'],
    'faithful':     ['CogComp/bart-faithful-summary-detector'],
    'humor':        ['mohameddhiab/humor-no-humor'],
    'beaver_reward':['PKU-Alignment/beaver-7b-v1.0-reward'],
    'beaver_cost':  ['PKU-Alignment/beaver-7b-v1.0-cost'],
}

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f'Unknown reward: {name}')
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

save_info = {
    'base_model_name': base_model_name,
    'peft_name':       peft_name,
    'tokenier_name':   tokenier_name,
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))

accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

set_seed(8888)
tokenizer = load_main_tokenizer(tokenier_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=gpu_id,
)
model.resize_token_embeddings(len(tokenizer))
if len(peft_name) > 0:
    model = PeftModel.from_pretrained(model, peft_name)
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()

# do not normalize for evaluation
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id, reward_stats_path)
num_rewards   = len(reward_model_path_list)
if exp_type in ('assistant', 'beaver'):
    instructions = Instructions_n(num_rewards)
else:
    instructions = Instructions_summary_n(num_rewards)

generation_kwargs = {
    "max_new_tokens": 128 if exp_type in ('assistant', 'beaver') else 48,
    "min_length": -1,
    "do_sample": False,
}

print('evaluation........')
tokenizer.padding_side = "left"

# prepare model once — not inside evaluate_model per preference round
model = accelerator.prepare(model)

# preference grid — use same logic as eval_ppo_rs.py
preferences = sample_preferences_uniform(reward_models.num_rewards, num_samples=script_args.num_prefer_points)

if process_id == 0:
    print(f"\nAll {len(preferences)} preference samples:")
    for i, pref in enumerate(preferences):
        print(f"  [{i}] {pref}")

# Gaussian reference for target score mapping
rewards_reference_list = [np.random.randn(50000) for _ in range(len(preferences[0]))]

rm_tokenizers = []
for i in range(num_rewards):
    rm_tokenizers.append(AutoTokenizer.from_pretrained(rm_tokenizer_path_list[i]))


def evaluate_model(model, reward_models, tokenizer, target_rewards, instructions, gpu_id):
    if exp_type == 'assistant':
        valid_dataset = build_dataset_with_preference_n(hhrlhf_dataset_path, tokenizer, rm_tokenizers, target_rewards, split='test')
    elif exp_type == 'summary':
        valid_dataset = build_summary_dataset_with_preference_n(summary_dataset_path, tokenizer, rm_tokenizers, target_rewards, split='test')
    else:  # beaver
        valid_dataset = build_beaver_dataset_with_preference_n(beaver_dataset_path, tokenizer, rm_tokenizers, target_rewards, split='test')
    print(f"Size of the validation set: {len(valid_dataset)}")

    valid_dataset = valid_dataset.remove_columns('input_ids')
    valid_dataset = valid_dataset.rename_column('prompt_with_score_ids', 'input_ids')
    score_cols = [c for c in valid_dataset.column_names if c.startswith('score')]
    valid_dataset = valid_dataset.remove_columns(['prompt', 'response', 'query', 'prompt_with_score'] + score_cols)
    for key in ['key', 'text']:
        if key in valid_dataset.column_names:
            valid_dataset = valid_dataset.remove_columns(key)
    data_collator    = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=32, drop_last=True, collate_fn=data_collator)
    valid_data_loader = accelerator.prepare(valid_data_loader)

    full_response_tensors = []
    full_prompts          = []
    pbar = tqdm(total=len(valid_dataset) // accelerator.num_processes // 32)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            response_tensors = accelerator.unwrap_model(model).generate(
                batch['input_ids'], attention_mask=batch['attention_mask'], **generation_kwargs)
            full_response_tensors.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)

    full_prompts   = tokenizer.batch_decode(full_prompts)
    full_responses = tokenizer.batch_decode(full_response_tensors)
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [(instructions.get_input(text), instructions.get_response(text)) for text in full_responses]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)

    desired_rewards_list = [[] for _ in range(reward_models.num_rewards)]
    for text in full_responses:
        desired_scores = instructions.get_scores(text)
        for i in range(reward_models.num_rewards):
            desired_rewards_list[i].append(float(desired_scores[i]))

    accelerator.wait_for_everyone()
    all_rewards         = []
    all_desired_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
        all_desired_rewards.append(accelerator.gather_for_metrics(desired_rewards_list[i]))
    all_full_prompts   = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_desired_rewards, all_full_prompts, all_full_responses


for k in range(len(preferences)):
    preference    = preferences[k]
    target_rewards = map_rewards_from_preference(rewards_reference_list, preference, method='l2').reshape(-1)
    all_rewards, all_desired_rewards, all_full_prompts, all_full_responses = evaluate_model(
        model, reward_models, tokenizer, target_rewards, instructions, gpu_id,
    )
    if process_id == 0:
        evaluation_result = {
            'prompt':   all_full_prompts,
            'response': all_full_responses,
        }
        obtained_means = []
        desired_means  = []
        for i in range(num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            evaluation_result['desired_score{}'.format(i+1)]  = all_desired_rewards[i]
            obtained_means.append(np.mean(all_rewards[i]))
            desired_means.append(np.mean(all_desired_rewards[i]))
        obtained_str = '  '.join(f'r{i+1}={obtained_means[i]:.4f}' for i in range(num_rewards))
        desired_str  = '  '.join(f'r{i+1}={desired_means[i]:.4f}'  for i in range(num_rewards))
        print(f'[pref {k}] preference={preference}')
        print(f'         obtained: {obtained_str}')
        print(f'         desired:  {desired_str}')

        dataframe = pd.DataFrame(evaluation_result)
        if len(preference) == 3:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,
                                          'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))
        else:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,
                                          'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))
