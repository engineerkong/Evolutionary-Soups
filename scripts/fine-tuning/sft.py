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
from scripts.utils.utils import Instructions, Instructions_summary, build_dataset_sft, build_dataset_summary_sft, load_main_tokenizer
tqdm.pandas()

# ========== define paths for two datasets ==========
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "local path to the base model or the huggingface id"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant'"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "loading model in 8 bit or bfloat16"})
    log_with: Optional[str] = field(default='none', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./models/sft/', metadata={"help": "directory to save the model"})
    wandb_name: Optional[str] = field(default='assistant_sft', metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print('base model: ', script_args.base_model_name)
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ========== define training and lora configurations ==========
training_args = TrainingArguments(
        max_steps=20000,  
        output_dir=output_dir,
        dataloader_drop_last=True,
        eval_steps=30000,
        save_steps=10000,
        logging_steps=10,
        save_strategy='steps',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1.4e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        weight_decay=0.01,
        run_name=script_args.wandb_name,
        report_to=script_args.log_with,
        ddp_find_unused_parameters=False,
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
if script_args.exp_type == 'assistant':
    dataset = build_dataset_sft(hhrlhf_dataset_path, tokenizer, split='train') 
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
else:
    dataset = build_dataset_summary_sft(summary_dataset_path, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
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
    

