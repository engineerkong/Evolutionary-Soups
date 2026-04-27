"""RiC — main training entry point.

Adapted from RiC/ric/main.py.

Changes vs original (RiC/ric/main.py):
  1. exp_type='beaver': added as a valid choice (uses Instructions_n / assistant
     format since PKU-SafeRLHF-10K uses the same \n\nHuman:/\n\nAssistant: template).
  2. reward_path_tokenizer_dict: extended with beaver_reward / beaver_cost entries.
  3. beaver_dataset_path: added as a module-level constant.
  4. sft_model_name arg (new): optional SFT LoRA adapter path passed as peft_name
     for the first offline training call so RiC starts from the same SFT initialisation
     as MORLHF, HoE, and NSGA-II. If empty, training starts from base_model_name.
"""

import os
import copy
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from utils import clean_gpu_memory, merge_dataset, save_configs
from training import train_model
from generation import generate_data
from transformers import HfArgumentParser

hhrlhf_dataset_path  = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
beaver_dataset_path  = 'PKU-Alignment/PKU-SafeRLHF-10K'


@dataclass
class ScriptArguments:
    log_with:                   Optional[str]   = field(default=None)
    disable_wandb:              Optional[str]   = field(default=False)
    save_directory:             Optional[str]   = field(default='./results/ric/', metadata={'help': 'path'})
    learning_rate:              Optional[float] = field(default=1e-5)
    batch_size:                 Optional[int]   = field(default=1)
    training_steps:             Optional[int]   = field(default=20000)
    online_training_steps:      Optional[int]   = field(default=4000)
    gradient_accumulation_steps:Optional[int]   = field(default=1)
    num_online_iterations:      Optional[int]   = field(default=1)
    num_generation_samples:     Optional[int]   = field(default=20000)
    max_grad_norm:              Optional[float] = field(default=1)
    quantile_threshold:         Optional[float] = field(default=0.7)
    num_origin_samples:         Optional[int]   = field(default=10000)
    load_in_8bit:               Optional[bool]  = field(default=True)
    bf16:                       Optional[bool]  = field(default=False)
    wandb_name:                 Optional[str]   = field(default='ric_assistant_harmlesshelpful_offline20000_lr1e-4')
    base_model_name:            Optional[str]   = field(default='meta-llama/Llama-2-7b-hf', metadata={'help': 'base LLaMA model (local path or HF id)'})
    sft_model_name:             Optional[str]   = field(default='', metadata={'help': 'SFT LoRA adapter path; used as peft_name for the first offline training iteration so RiC starts from the same SFT init as other methods'})
    peft_name:                  Optional[str]   = field(default='')
    reward_names:               Optional[str]   = field(default='harmless,helpful')
    train_dataset_path:         Optional[str]   = field(default='./datasets/all_full_train_harmhelp.hf')
    train_reward_stats_path:    Optional[str]   = field(default='')
    exp_type:                   Optional[str]   = field(default='assistant', metadata={"help": "assistant | summary | beaver"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type        = script_args.exp_type
base_model_name = script_args.base_model_name
tokenizer_name  = script_args.base_model_name
print('base model: ', base_model_name)

if script_args.disable_wandb:
    os.environ['WANDB_DISABLED'] = 'true'

reward_names = [x.strip() for x in script_args.reward_names.split(',')]

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

train_dataset_path = script_args.train_dataset_path
reward_stats_path  = (script_args.train_reward_stats_path
                      if len(script_args.train_reward_stats_path)
                      else script_args.train_dataset_path + '/all_reward_stat.npy')

save_info = {
    'train_dataset_path': train_dataset_path,
    'base_model_name':    base_model_name,
    'tokenizer_name':     tokenizer_name,
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))

save_path = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(save_path, exist_ok=True)

# offline training — start from SFT LoRA adapter if provided
initial_peft = script_args.sft_model_name if script_args.sft_model_name else None
dataset = train_model(
    base_model_name=base_model_name,
    reward_model_path_list=reward_model_path_list,
    train_dataset=train_dataset_path,
    save_path=save_path + '/model_iter0',
    tokenizer_name=tokenizer_name,
    rm_tokenizer_path_list=rm_tokenizer_path_list,
    peft_name=initial_peft,
    training_steps=script_args.training_steps,
    learning_rate=1.414e-4,
    args=script_args,
    exp_type=exp_type,
)
clean_gpu_memory()

online_dataset = None
model_path     = base_model_name
for i in range(script_args.num_online_iterations):
    print('iteration {} ...'.format(i))
    checkpoint_path = os.path.join(save_path, 'model_iter{}'.format(i))
    if i == 0 and script_args.training_steps == 0:
        peft_name = script_args.peft_name
    else:
        peft_name = checkpoint_path

    if script_args.num_generation_samples > 0 and not os.path.exists(os.path.join(checkpoint_path, 'data.csv')):
        generate_data(
            model_path,
            reward_model_path_list=reward_model_path_list,
            tokenizer_name=tokenizer_name,
            rm_tokenizer_path_list=rm_tokenizer_path_list,
            dataset=dataset,
            save_path=os.path.join(save_path, 'model_iter{}'.format(i)),
            peft_name=peft_name,
            reward_stats_path=reward_stats_path,
            iter=i,
            args=script_args,
            exp_type=exp_type,
        )

    clean_gpu_memory()
    info_path    = os.path.join(script_args.save_directory, script_args.wandb_name, 'reward_info.npy')
    merged_data, online_dataset = merge_dataset(
        dataset, online_dataset, checkpoint_path, tokenizer_name,
        info_path=info_path, exp_type=exp_type,
        quantile_threshold=script_args.quantile_threshold,
        sample_origin=script_args.num_origin_samples,
    )

    train_model(
        base_model_name=model_path,
        peft_name=peft_name,
        reward_model_path_list=reward_model_path_list,
        train_dataset=merged_data,
        save_path=save_path + '/model_iter{}'.format(i+1),
        tokenizer_name=tokenizer_name,
        rm_tokenizer_path_list=rm_tokenizer_path_list,
        training_steps=script_args.online_training_steps,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type='constant',
        iter=i+1,
        args=script_args,
        exp_type=exp_type,
    )
    clean_gpu_memory()
