"""MORLHF evaluation — generate responses and measure per-objective rewards.

Aligned with eval_ppo_single.py so that results are directly comparable with
RS, HoE, and ES evaluations.  Key invariants shared across all eval scripts:
  - normalize_rewards=False  (raw reward scores, no scaling)
  - mini_batch_size=8
  - min_length=-1, do_sample=False
  - valid_dataset.with_format("numpy")
  - drop_last=False
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
    build_dataset_steer_eval,
    build_dataset_ultrafeedback_eval,
    get_clean_data,
)
from scripts.utils.multi_reward_models import RewardModels

tqdm.pandas()

_ARMORM = 'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1'
_URM    = 'urm://LxzGordon/URM-LLaMa-3.1-8B'

reward_path_tokenizer_dict = {
    'harmless':                 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':                  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':                  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':                  'Tristan/gpt2_reward_summarization',
    'faithful':                 'CogComp/bart-faithful-summary-detector',
    'humor':                    'mohameddhiab/humor-no-humor',
    'beaver_reward':            'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':              'PKU-Alignment/beaver-7b-v1.0-cost',
    'steer_helpfulness':        f'{_URM}#0',
    'steer_correctness':        f'{_URM}#1',
    'steer_coherence':          f'{_URM}#2',
    'steer_complexity':         f'{_URM}#3',
    'steer_verbosity':          f'{_URM}#4',
    'uf_instruction_following': f'{_ARMORM}#6',
    'uf_truthfulness':          f'{_ARMORM}#7',
    'uf_honesty':               f'{_ARMORM}#8',
    'uf_helpfulness':           f'{_ARMORM}#9',
}

_DEFAULT_REWARD_NAMES = {
    'Anthropic/hh-rlhf':                'harmless,helpful',
    'openai/summarize_from_feedback':    'summary,faithful',
    'argilla/news-summary':              'summary,faithful',
    'PKU-Alignment/PKU-SafeRLHF-10K':   'beaver_reward,beaver_cost',
    'nvidia/HelpSteer':                  'steer_helpfulness,steer_correctness',
    'nvidia/HelpSteer2':                 'steer_helpfulness,steer_correctness',
    'openbmb/UltraFeedback':             'uf_instruction_following,uf_truthfulness,uf_honesty,uf_helpfulness',
}


@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={'help': 'base LLaMA model (local path or HF id)'})
    checkpoint_path: Optional[str] = field(default='', metadata={'help': 'trained MORLHF LoRA adapter directory'})
    dataset_name:    Optional[str] = field(default='Anthropic/hh-rlhf', metadata={'help': 'dataset name'})
    reward_names:    Optional[str] = field(default='', metadata={'help': 'comma-separated reward names; auto-selected from dataset_name if empty'})
    save_directory:  Optional[str] = field(default='./results/morlhf/')
    wandb_name:      Optional[str] = field(default='eval_morlhf')


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if not script_args.reward_names:
    script_args.reward_names = _DEFAULT_REWARD_NAMES.get(script_args.dataset_name, 'harmless,helpful')
reward_names = [x.strip() for x in script_args.reward_names.split(',')]

print('base model:  ', script_args.base_model_name)
print('checkpoint:  ', script_args.checkpoint_path)
print('reward names:', reward_names)

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f'Unknown reward name: {name!r}')
    reward_model_path_list.append(reward_path_tokenizer_dict[name])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name])

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
process_id = Accelerator().local_process_index
gpu_id     = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

# ── Load model ────────────────────────────────────────────────────────────────
tokenizer = load_main_tokenizer(script_args.base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name,
    torch_dtype=torch.bfloat16,
    device_map=gpu_id,
)
model.resize_token_embeddings(len(tokenizer))
if script_args.checkpoint_path:
    model = PeftModel.from_pretrained(model, script_args.checkpoint_path)
    model = model.merge_and_unload()

# ── Generation kwargs (identical to eval_ppo_single.py) ──────────────────────
generation_kwargs = {
    "max_new_tokens": 128 if script_args.dataset_name in {
        'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K',
        'nvidia/HelpSteer', 'nvidia/HelpSteer2'} else 48,
    "min_length": -1,
    "do_sample": False,
}
tokenizer.padding_side = "left"

# ── Dataset ───────────────────────────────────────────────────────────────────
if script_args.dataset_name == 'Anthropic/hh-rlhf':
    valid_dataset = build_dataset_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    valid_dataset = build_dataset_summary_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    valid_dataset = build_dataset_beaver_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    valid_dataset = build_dataset_steer_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions()
elif script_args.dataset_name == 'openbmb/UltraFeedback':
    valid_dataset = build_dataset_ultrafeedback_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers)
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}')

print(f"Size of the validation set: {len(valid_dataset)}")

mini_batch_size = 8
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

valid_dataset = valid_dataset.with_format("numpy")
data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=mini_batch_size,
                               drop_last=False, collate_fn=data_collator)
accelerator = Accelerator()
model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

# ── Generation ────────────────────────────────────────────────────────────────
print('evaluation........')
full_response_tensors = []
full_prompts          = []

pbar = tqdm(total=len(valid_dataset) // mini_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        response_tensors = accelerator.unwrap_model(model).generate(
            batch['input_ids'], attention_mask=batch['attention_mask'], **generation_kwargs,
        )
        full_response_tensors.extend(response_tensors)
        full_prompts.extend(batch['input_ids'])
        pbar.update(1)

full_prompts   = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

queries_responses = [
    (instructions.get_input(text), instructions.get_response(text))
    for text in full_responses
]

# normalize_rewards=False: report raw scores, same as all other eval scripts
if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(
        queries_responses, instructions.get_post, normalize_rewards=False)
else:
    rewards_list = reward_models.get_reward_model_scores(
        queries_responses, normalize_rewards=False)

all_rewards        = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
all_full_prompts   = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)

if process_id == 0:
    print('Saving evaluation results')
    evaluation_result = {
        'prompt':   all_full_prompts,
        'response': all_full_responses,
    }
    for i in range(reward_models.num_rewards):
        evaluation_result['obtained_score{}'.format(i + 1)] = all_rewards[i]
        print('total average obtained score {}: {}'.format(
            i + 1, np.mean(evaluation_result['obtained_score{}'.format(i + 1)])))

    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(os.path.join(output_dir, 'eval_data.csv'))
