import sys
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, DataCollatorWithPadding
from trl import set_seed
from peft import PeftModel
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent  # project/scripts/fine-tuning
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import load_main_tokenizer, check_lora_in_model_path, Instructions, Instructions_summary, \
                    build_dataset_eval_sft, build_dataset_summary_eval_sft, get_clean_data_sft
tqdm.pandas()

# ========== define paths for two datasets ==========
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    sft_model_name: Optional[str] = field(default='./models/sft/', metadata={'help':"the path to the sft model; need to merge if using lora"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    reward_names:Optional[str] = field(default='harmless,helpful', metadata={"help": "the reward model name: 'summary', 'faithful', 'helpful', 'harmless', 'deberta', 'humor'"}) 
    save_directory: Optional[str] = field(default='./results/sft/', metadata={"help": "directory to save the results"})
    wandb_name: Optional[str] = field(default='assistant_sft_eval', metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print('sft model: ', script_args.sft_model_name)
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
process_id = Accelerator().local_process_index 
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
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

# ========== load sft model and tokenizer ==========
tokenizer = load_main_tokenizer(script_args.sft_model_name)
model = AutoModelForCausalLM.from_pretrained(
    script_args.sft_model_name, 
    torch_dtype=torch.bfloat16,
    device_map=gpu_id, 
)
model.resize_token_embeddings(len(tokenizer))
if check_lora_in_model_path(model, script_args.sft_model_name):
    model = PeftModel.from_pretrained(model, script_args.sft_model_name)
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()

# ========== define generation kwargs ==========
generation_kwargs = {
    "max_new_tokens": 128 if script_args.exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9, 
    "do_sample": True,
}
tokenizer.padding_side = "left"

# ========== prepare evaluation dataset and dataloader ==========
if script_args.exp_type == 'assistant':
    valid_dataset = build_dataset_eval_sft(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test') 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval_sft(summary_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test') 
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

valid_batch_size = 1
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, collate_fn=data_collator)
accelerator = Accelerator()
model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

# ========== start evaluation ==========
print('evaluation........')
full_response_tensors = []
full_prompts = []

pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        response_tensors = accelerator.unwrap_model(model).generate(batch['input_ids'], attention_mask=batch['attention_mask'], **generation_kwargs)
        full_response_tensors.extend(response_tensors)
        full_prompts.extend(batch['input_ids'])
        pbar.update(1)

full_prompts = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_responses = get_clean_data_sft(full_responses, full_prompts)

queries_responses = [
    (instructions.get_input(text),  instructions.get_response(text))
    for text in full_responses
]

if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post, normalize_rewards=False) # no normalization during eval
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses, normalize_rewards=False)

all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
all_full_prompts = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)

if process_id == 0:
    print('Saving evaluation results')
    evaluation_result = {
        'prompt': all_full_prompts,
        'response': all_full_responses,
    }
    for i in range(reward_models.num_rewards):
        evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
        print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(os.path.join(output_dir,'eval_data.csv'))

