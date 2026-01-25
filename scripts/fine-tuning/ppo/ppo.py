import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np
import pandas as pd
from utils import print_trainable_parameters, load_main_tokenizer, Instructions, Instructions_summary, \
                  build_dataset, build_dataset_summary                  
from multi_reward_models import RewardModels
tqdm.pandas()
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
pt_model_name = 'meta-llama/Llama-2-7b-hf'

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=False, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_ppo_summary/')
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1.4e-4, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target: Optional[float] = field(default=3, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.2,metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},)
    max_grad_norm: Optional[float] = field(default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"})
    wandb_name: Optional[str] = field(default='ppo_llamma2_klreg0.2_summary_faithfulrm', metadata={"help": "Name for this experiment"})
    exp_type: Optional[str] = field(default='summary', metadata={"help": "exp type: 'assistant" or 'summary'}) 
    base_model_name: Optional[str] = field(default='./merged_sft_summary', metadata={'help':"the path to the sft model; need to merge if using lora"})
    reward_name: Optional[str] = field(default='faithful')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
# Remember to use a merged sft model if using lora 
base_model_name = script_args.base_model_name
tokenier_name = script_args.base_model_name
print('base model: ', base_model_name)

from peft import PeftConfig
sft_config = PeftConfig.from_pretrained(base_model_name)
print("=" * 70)
print("SFT LoRA Configuration:")
print(f"  r: {sft_config.r}")
print(f"  lora_alpha: {sft_config.lora_alpha}")
print(f"  target_modules: {sft_config.target_modules}")
print(f"  lora_dropout: {sft_config.lora_dropout}")
print("=" * 70)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

reward_name = script_args.reward_name
if reward_name == 'summary':
    reward_peft_path = 'Tristan/gpt2_reward_summarization'
elif reward_name == 'faithful':
    reward_peft_path = 'CogComp/bart-faithful-summary-detector'
elif reward_name == 'helpful':
    reward_peft_path = 'Ray2333/gpt2-large-helpful-reward_model'
elif reward_name == 'harmless':
    reward_peft_path = 'Ray2333/gpt2-large-harmless-reward_model'
elif reward_name == 'deberta':
    reward_peft_path = 'OpenAssistant/reward-model-deberta-v3-large-v2'
elif reward_name == 'humor':
    reward_peft_path = 'mohameddhiab/humor-no-humor'
else:
    raise NotImplementedError
rm_tokenizer_path = reward_peft_path
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)


config = PPOConfig(
    model_name=base_model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target=script_args.target,
    max_grad_norm=script_args.max_grad_norm,
    optimize_cuda_cache=True,
    init_kl_coef=script_args.init_kl_coef,
    tracker_project_name='ppo',
    tracker_kwargs={"wandb":{"name":script_args.wandb_name}},
)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}'.format(process_id))
reward_model = RewardModels([reward_peft_path], [rm_tokenizer_path], gpu_id)
rm_tokenizer = reward_model.rm_tokenizers[0] 


# set seed before initializing value head for deterministic eval
set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=sft_config.r,  # 使用 SFT 的配置
    lora_alpha=sft_config.lora_alpha,
    lora_dropout=sft_config.lora_dropout,
    target_modules=sft_config.target_modules,  # 必须一致！
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    dataset = build_dataset(hhrlhf_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
else:
    dataset = build_dataset_summary(summary_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}.")

if script_args.load_in_8bit:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pt_model_name,
        load_in_8bit=True,
        device_map=gpu_id,
    )
    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model)
    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model,
        base_model_name,
        is_trainable=True
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pt_model_name,
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model,
        base_model_name,
        is_trainable=True
    )

print("After loading SFT LoRA:")
print_trainable_parameters(model)


model.pretrained_model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48,
    'min_length': -1, 
    "top_k": 0.0,
    "top_p": 1.0, 
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id],
}

print("Training........")
model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True

epochs = script_args.epochs
mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    'kl_std': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
}

for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)

    # 训练前保存初始参数
    print("Saving initial parameters for comparison...")
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.clone().detach()

    # 在第一个 batch 后检查
    first_batch_checked = False

    for i, batch in enumerate(ppo_trainer.dataloader):
        print('epoch {}, batch {}'.format(epoch, i))
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        with torch.no_grad():
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs
            )

        decoded_responses = tokenizer.batch_decode(
            response_tensors,
            skip_special_tokens=True
        )
        decoded_responses = [r.strip() for r in decoded_responses]
        batch["response"] = decoded_responses

        # Compute rewards
        texts_merge = [q + r for q, r in zip(batch['query'], batch['response'])]
        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in texts_merge
        ]
        if hasattr(instructions, 'get_post'):
            rewards = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)[0]
        else:
            rewards = reward_model.get_reward_model_scores(queries_responses)[0]

        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("iter {}, batch {}: mean score: {}".format(epoch, i, torch.mean(torch.tensor(rewards)).item()))

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        # ========== 关键修改：Hook 梯度检查 ==========
        if i == 0:
            # 注册 hook 来捕获梯度计算
            gradient_info = {}
            def grad_hook(name):
                def hook(grad):
                    gradient_info[name] = grad.norm().item()
                    return grad
                return hook
            
            # 为所有可训练参数注册 hook
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.register_hook(grad_hook(name))

        # PPO step - 这里会计算梯度并更新参数
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        
        # ========== 在 step 之后立即检查（通过 hook 捕获的信息）==========
        if i == 0 and gradient_info:
            print("\n" + "=" * 70)
            print("Gradient Check (captured via hooks):")
            has_grad = False
            for name, grad_norm in gradient_info.items():
                if grad_norm > 1e-8:
                    print(f"  ✓ {name}: grad_norm={grad_norm:.6e}")
                    has_grad = True
            
            if not has_grad:
                print("  ❌ NO GRADIENTS CAPTURED!")
            else:
                print(f"  ✓ Total parameters with gradients: {len(gradient_info)}")
            print("=" * 70 + "\n")
        
        ppo_trainer.log_stats(stats, batch, rewards)
        policy_kl = [stats["objective/kl"]]

        # 参数更新检查
        if not first_batch_checked:
            print("\n" + "=" * 70)
            print("Checking if parameters actually updated after first batch:")
            params_changed = False
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_params:
                    diff = (param - initial_params[name]).abs().max().item()
                    if diff > 1e-8:
                        print(f"✓ {name}: changed by {diff:.6e}")
                        params_changed = True

            if not params_changed:
                print("❌ WARNING: NO PARAMETERS CHANGED!")
            else:
                print("✓ Parameters are updating correctly")
            print("=" * 70 + "\n")
            first_batch_checked = True

        all_rewards = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)

        if ppo_trainer.accelerator.is_main_process:
            mean_scores.append(torch.mean(torch.tensor(rewards)).item())
            std_scores.append(torch.std(torch.tensor(rewards)).item())

            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png')
            plt.plot(mean_scores)
            plt.fill_between(
                np.arange(len(mean_scores)),
                np.array(mean_scores) - np.array(std_scores),
                np.array(mean_scores) + np.array(std_scores),
                alpha=0.5
            )
            plt.savefig(save_path)

            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['kl_std'].append(np.std(all_policy_kl))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append(texts_merge[0])

            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name, 'data.csv'))
            print("iter {}, batch {}: log finish".format(epoch, i))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)

        # save model
        if ppo_trainer.accelerator.is_main_process and i % 100 == 0 and i != 0:
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'batch_{}'.format(i))
            ppo_trainer.save_pretrained(save_path)
            print("iter {}, batch {}: model saved".format(epoch, i))

    # save model
    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'batch_{}'.format(i))
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))

            