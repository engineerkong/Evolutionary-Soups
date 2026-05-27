import shutil
import sys
from pathlib import Path
import os
import gc
from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM, DataCollatorWithPadding
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent  # project/scripts/fine-tuning
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.baselines.utils.multi_reward_models import RewardModels
from scripts.baselines.utils.utils import get_clean_data, load_main_tokenizer, save_configs, sample_preferences_uniform, \
    Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval, \
    build_dataset_summary_ppo, \
    build_dataset_news_summary_ppo, build_dataset_beaver_eval, build_dataset_steer_eval, \
    build_dataset_ultrafeedback_eval
tqdm.pandas()


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


def _merge_and_save_rewarded_soup(base_model_name, ppo_model_list, preference, save_path):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map='cpu')

    expert_sds  = [_load_adapter_sd(p) for p in ppo_model_list]
    peft_config = LoraConfig.from_pretrained(ppo_model_list[0])
    blended_sd  = {
        k: sum(preference[i] * expert_sds[i][k].to(torch.bfloat16) for i in range(len(preference)))
        for k in expert_sds[0]
    }
    model = get_peft_model(base, peft_config)
    set_peft_model_state_dict(model, blended_sd)
    model = model.merge_and_unload()

    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    model.save_pretrained(save_path)

    del model, base, expert_sds, blended_sd
    gc.collect()
    torch.cuda.empty_cache()

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "base LLaMA model path or HF id"})
    expert_model_paths: List[str] = field(default_factory=list, metadata={"help": "PPO adapter paths (space-separated on CLI)"})
    num_pref_samples: int = field(default=10, metadata={"help": "number of preference samples per input"})
    reward_names: Optional[str] = field(default='', metadata={"help": "comma-separated reward names; auto-selected from dataset_name if empty"})
    dataset_name: Optional[str] = field(default='Anthropic/hh-rlhf', metadata={"help": "dataset: 'Anthropic/hh-rlhf', 'openai/summarize_from_feedback', or 'argilla/news-summary'"})
    save_directory: Optional[str] = field(default='./results/ppo_rs/', metadata={"help": "directory to save results"})
    wandb_name: Optional[str] = field(default='assistant_ppo_rs_eval', metadata={"help": "name for this experiment"})
    split: Optional[str] = field(default='test', metadata={"help": "dataset split: 'train' or 'test'"})
    size: Optional[int] = field(default=0, metadata={"help": "number of prompts to use (0 = all)"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ppo_model_list = script_args.expert_model_paths
print(script_args)

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ========== load reward models ==========
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print('reward names:', reward_names)
_ARMORM = 'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1'
_URM = 'urm://LxzGordon/URM-LLaMa-3.1-8B'
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
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f'Unknown reward name: {name!r}')
    reward_model_path_list.append(reward_path_tokenizer_dict[name])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name])

save_info = {'tokenizer_name': ppo_model_list[0]}
for i, p in enumerate(ppo_model_list):
    save_info[f'ppo_model_name{i+1}'] = p
for i, p in enumerate(reward_model_path_list):
    save_info[f'reward_peft_path{i+1}'] = p
save_configs(save_info, output_dir)
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 

# ========== prepare evaluation dataset and dataloader ==========
tokenizer = load_main_tokenizer(ppo_model_list[0])
if script_args.dataset_name == 'Anthropic/hh-rlhf':
    valid_dataset = build_dataset_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    valid_dataset = build_dataset_summary_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions_summary()
elif script_args.dataset_name == 'argilla/news-summary':
    valid_dataset = build_dataset_news_summary_ppo(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    valid_dataset = build_dataset_beaver_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    valid_dataset = build_dataset_steer_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions()
elif script_args.dataset_name == 'openbmb/UltraFeedback':
    valid_dataset = build_dataset_ultrafeedback_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers)
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}')
if script_args.size > 0:
    valid_dataset = valid_dataset.select(range(min(script_args.size, len(valid_dataset))))
print(f"Size of the validation set: {len(valid_dataset)}")

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

# ========== evaluation function ==========
def evaluate_model(temp_save_path, tokenizer, valid_dataset):
    mini_batch_size = 32
    valid_dataset = valid_dataset.with_format("numpy")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, mini_batch_size, drop_last=False, collate_fn=data_collator)
    # load_merged_model
    model = AutoModelForCausalLM.from_pretrained(
        temp_save_path,
        torch_dtype=torch.bfloat16,
        device_map=gpu_id
        )
    model.resize_token_embeddings(len(tokenizer))

    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    # define generation kwargs
    generation_kwargs = {
        "max_new_tokens": 128 if script_args.dataset_name in {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K'} else 48,
        "min_length": -1,
        "do_sample": False,
    }
    tokenizer.padding_side = "left"
    
    # obtain model responses
    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            response_tensors = accelerator.unwrap_model(model).generate(batch["input_ids"], attention_mask=batch['attention_mask'], **generation_kwargs) #length_sampler=output_length_sampler, 
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    # clean data
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post, normalize_rewards=False)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses, normalize_rewards=False)
    
    # error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_full_prompts, all_full_responses


# ========== start evaluation ==========
print("evaluating........")
# preference list
sampled_preferences = sample_preferences_uniform(reward_models.num_rewards, num_samples=script_args.num_pref_samples)
print(f"\nSampled {len(sampled_preferences)} preferences:")
for i, pref in enumerate(sampled_preferences):
    pref_str = ", ".join([f"{reward_names[k]}={pref[k]:.2f}" for k in range(len(reward_names))])
    print(f"  Pref {i+1}: [{pref_str}]")

for k in range(0, len(sampled_preferences)):
    preference = sampled_preferences[k]
    temp_save_path = output_dir + '/temp_merged_model_pref_{}_{}'.format('_'.join([str(p) for p in preference]), k)
    if process_id == 0:
        _merge_and_save_rewarded_soup(
            script_args.base_model_name,
            ppo_model_list, preference, temp_save_path)
        print("merged model saved to {}".format(temp_save_path))

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    print(k, preference)
    all_rewards, all_full_prompts, all_full_responses = evaluate_model(temp_save_path, tokenizer, valid_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)
        pref_str = '_'.join([str(p) for p in preference])
        dataframe.to_csv(os.path.join(output_dir, f'eval_data_pref{pref_str}.csv'), escapechar='\\')




