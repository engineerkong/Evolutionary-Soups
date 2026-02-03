import sys
from pathlib import Path
import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM, DataCollatorWithPadding
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent  # project/scripts/fine-tuning
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import get_clean_data_ppo, load_main_tokenizer, save_configs, merge_weights_with_preference, \
                                Instructions, Instructions_summary, build_dataset_eval_ppo, build_dataset_summary_eval_ppo
tqdm.pandas()

# ========== define paths for two datasets ==========
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "local path to the base model or the huggingface id"})
    ppo_model_path1: Optional[str]=field(default='./ppo/assistant_ppo_harmless/batch_832')
    ppo_model_path2: Optional[str]=field(default='./ppo/assistant_ppo_helpful/batch_832')
    ppo_model_path3: Optional[str]=field(default='')
    reward_names:Optional[str] = field(default='harmless,helpful', metadata={"help": "comma separated list of reward names"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    save_directory: Optional[str] = field(default='./results/ppo_rs/', metadata={"help": "directory to save results"})
    wandb_name: Optional[str] = field(default='assistant_ppo_rs_eval', metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
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
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

save_info = {
    'ppo_model_path1': script_args.ppo_model_path1,
    'ppo_model_path2': script_args.ppo_model_path2,
    'ppo_model_path3': script_args.ppo_model_path3,
    'reward_peft_path1': reward_model_path_list[0],
    'reward_peft_path2': reward_model_path_list[1],
    'tokenizer_name': script_args.base_model_name,
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, output_dir)
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 

# ========== prepare evaluation dataset and dataloader ==========
tokenizer = load_main_tokenizer(script_args.base_model_name)
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_ppo(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test') 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_ppo(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test')
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

# ========== evaluation function ==========
def evaluate_model(temp_save_path, tokenizer, valid_dataset):
    mini_batch_size = 1
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, mini_batch_size, drop_last=True, collate_fn=data_collator)
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
        "max_new_tokens": 128 if script_args.exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }
    tokenizer.padding_side = "left"
    
    # obtain model responses
    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            response_tensors = accelerator.unwrap_model(model).generate(batch["input_ids"], **generation_kwargs) #length_sampler=output_length_sampler, 
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    # clean data
    full_prompts, full_responses = get_clean_data_ppo(full_responses, full_prompts)

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
if reward_models.num_rewards == 3:
    preferences = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.4, 0.4],
        [0.2, 0.6, 0.2],
        [0.33, 0.33, 0.33],
        [0.4, 0.4, 0.2],
        [0.4, 0.2, 0.4], 
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1], 
        [1.0, 0.0, 0.0], 
        ])
elif reward_models.num_rewards == 2:
    # # orginal code for two rewards
    # M = 10
    # preferences = np.zeros((M+1, 2))
    # preferences[:, 0] = np.arange(0,1+ 1/M,1 / M)
    # preferences[:, 1] = 1 - preferences[:, 0]
    # preferences = np.round(preferences, 1)
    preferences = np.array([
        [0.5, 0.5]
        ])
else:
    raise NotImplementedError

for k in range(0, len(preferences)):
    preference = preferences[k]
    temp_save_path = output_dir + '/temp_merged_model_pref_{}_{}'.format('_'.join([str(p) for p in preference]), k)
    if process_id == 0:
        if len(preference) == 3:
            ppo_model_list = [script_args.ppo_model_path1, script_args.ppo_model_path2, script_args.ppo_model_path3]
        else:
            ppo_model_list = [script_args.ppo_model_path1, script_args.ppo_model_path2]
        merge_weights_with_preference(ppo_model_list, preference, temp_save_path)
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
        if len(preference) == 2:
            dataframe.to_csv(os.path.join(output_dir,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))
        else:
            dataframe.to_csv(os.path.join(output_dir,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))




