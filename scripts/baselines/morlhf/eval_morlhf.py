"""MORLHF evaluation — generate responses and measure per-objective rewards.

Adapted from RiC/ppo/eval_ppo_single_model.py.

Changes vs original (RiC/ppo/eval_ppo_single_model.py):
  1. Imports: sys.path → MOMoE/scripts/utils; uses scripts.utils helpers.
  2. Beaver support: exp_type='beaver' added using build_dataset_beaver_eval;
     uses Instructions() (same Human/Assistant format as assistant).
  3. Reward dict: extended with beaver_reward / beaver_cost entries.
  4. Model loading split into two args:
       base_model_name  = bare LLaMA weights
       checkpoint_path  = trained LoRA adapter saved by ppo_trainer.save_pretrained()
     Loads base, applies LoRA via PeftModel.from_pretrained, then merges.
     The original loaded a single merged checkpoint path; that fails when
     ppo_trainer saves only an adapter directory.
  5. Multi-GPU: single Accelerator() instance; process_id from accelerator attribute.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

script_dir   = Path(__file__).resolve().parent        # baselines/morlhf/
project_root = script_dir.parent.parent.parent         # MOMoE/
sys.path.insert(0, str(project_root))

from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, DataCollatorWithPadding
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from peft import PeftModel
from scripts.utils.utils import (
    load_main_tokenizer,
    Instructions, Instructions_summary,
    build_dataset_eval,
    build_dataset_summary_eval,
    build_dataset_beaver_eval,
    get_clean_data,
)
from scripts.utils.multi_reward_models import RewardModels

tqdm.pandas()

hhrlhf_dataset_path  = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
beaver_dataset_path  = 'PKU-Alignment/PKU-SafeRLHF-10K'


@dataclass
class ScriptArguments:
    save_directory:  Optional[str] = field(default='./results/morlhf/')
    batch_size:      Optional[int] = field(default=1)
    wandb_name:      Optional[str] = field(default='eval_morlhf')
    reward_names:    Optional[str] = field(default='harmless,helpful,humor')
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={'help': 'base LLaMA model (local path or HF id)'})
    checkpoint_path: Optional[str] = field(default='', metadata={'help': 'trained MORLHF LoRA adapter (output of ppo_trainer.save_pretrained)'})
    exp_type:        Optional[str] = field(default='assistant', metadata={"help": "assistant | summary | beaver"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type        = script_args.exp_type
base_model_name = script_args.base_model_name
checkpoint_path = script_args.checkpoint_path
tokenier_name   = base_model_name
print('base model:  ', base_model_name)
print('checkpoint:  ', checkpoint_path)

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards  = len(reward_names)
print('number of rewards: {}'.format(num_rewards))

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

os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

set_seed(8888)
tokenizer = load_main_tokenizer(tokenier_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=gpu_id,
)
model.resize_token_embeddings(len(tokenizer))
# Apply the trained PPO LoRA adapter and merge for generation
if checkpoint_path:
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()

generation_kwargs = {
    "max_new_tokens": 128 if exp_type in ('assistant', 'beaver') else 48,
    "do_sample": False,
}

print('evaluation........')
tokenizer.padding_side = "left"

if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions  = Instructions()
elif exp_type == 'summary':
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions  = Instructions_summary()
else:  # beaver
    valid_dataset = build_dataset_beaver_eval(beaver_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions  = Instructions()

print(f"Size of the validation set: {len(valid_dataset)}")

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=script_args.batch_size, drop_last=False, collate_fn=data_collator)
model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

full_response_tensors = []
full_prompts          = []

pbar = tqdm(total=len(valid_dataset) // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        response_tensors = accelerator.unwrap_model(model).generate(
            batch['input_ids'], attention_mask=batch['attention_mask'], **generation_kwargs,
        )
        full_response_tensors.extend(response_tensors)
        full_prompts.extend(batch['input_ids'])
        pbar.update(1)

full_prompts  = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

queries_responses = [
    (instructions.get_input(text), instructions.get_response(text))
    for text in full_responses
]

if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses)

all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))

all_full_prompts   = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)

if process_id == 0:
    evaluation_result = {
        'prompt':   all_full_prompts,
        'response': all_full_responses,
    }
    for i in range(reward_models.num_rewards):
        evaluation_result['obtained_score{}'.format(i + 1)] = all_rewards[i]
        print('total average obtained score {}: {}'.format(
            i + 1, np.mean(evaluation_result['obtained_score{}'.format(i + 1)])))

    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name, 'eval_data.csv'))
