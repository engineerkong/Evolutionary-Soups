import sys
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, set_seed
from peft import LoraConfig

script_dir = Path(__file__).resolve().parent  # project/scripts/fine-tuning
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_config, Instructions, Instructions_summary, \
    build_dataset_sft, build_dataset_summary_sft, build_dataset_news_summary_sft, \
    build_dataset_beaver_sft, build_dataset_steer_sft, load_main_tokenizer
tqdm.pandas()

SUMMARIZATION_DATASETS = {'openai/summarize_from_feedback', 'argilla/news-summary'}

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "local path to the base model or the huggingface id"})
    dataset_name: Optional[str] = field(default='Anthropic/hh-rlhf', metadata={"help": "dataset: 'Anthropic/hh-rlhf', 'openai/summarize_from_feedback', or 'argilla/news-summary'"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "loading model in 8 bit or bfloat16"})
    log_with: Optional[str] = field(default='none', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./models/sft/', metadata={"help": "directory to save the model"})
    wandb_name: Optional[str] = field(default='assistant_sft', metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config(script_dir / 'config.yaml')['sft']
print(f"Script arguments: {script_args}")
print(f"Training config: {cfg}")

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ========== define training and lora configurations ==========
training_args = TrainingArguments(
        **cfg,
        output_dir=output_dir,
        run_name=script_args.wandb_name,
        report_to=script_args.log_with,
    )

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    target_modules=["gate_proj", "up_proj", "down_proj"], # FFN layers in Llama2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ========== load model and tokenizer ==========
tokenizer = load_main_tokenizer(script_args.base_model_name)
if script_args.load_in_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, 
        load_in_8bit=True, 
        device_map=gpu_id, 
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id, 
    )
model.resize_token_embeddings(len(tokenizer))

# ========== prepare dataset and data collator ==========
if script_args.dataset_name == 'Anthropic/hh-rlhf':
    dataset = build_dataset_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    dataset = build_dataset_summary_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
elif script_args.dataset_name == 'argilla/news-summary':
    # argilla/news-summary: train split has only ~1000 samples; test split has ~20k — use test for training
    dataset = build_dataset_news_summary_sft(script_args.dataset_name, tokenizer, split='test')
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    dataset = build_dataset_beaver_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    dataset = build_dataset_steer_sft(script_args.dataset_name, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}. '
                     f'Choose from: Anthropic/hh-rlhf, openai/summarize_from_feedback, argilla/news-summary, '
                     f'PKU-Alignment/PKU-SafeRLHF-10K, nvidia/HelpSteer, nvidia/HelpSteer2')
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}")

# ========== define trainer and verify trainable parameters ==========
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
    packing=False,
    dataset_text_field="query",
    data_collator=collator,
)

if process_id == 0:
    trainer.model.print_trainable_parameters()

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")
    print(f"Trainable %: {100 * trainable / total:.6f}%")
    
# ========== start training ==========
print("Training........")
trainer.train()

if process_id == 0:
    print("Saving last checkpoint of the model")
    save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'model')
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    