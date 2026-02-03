import sys
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, set_seed
from peft import LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent  # project/scripts/fine-tuning
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_config, Instructions, Instructions_summary, build_dataset_ppo, build_dataset_summary_ppo, load_main_tokenizer, print_trainable_parameters                
from scripts.utils.multi_reward_models import RewardModels
tqdm.pandas()

# ========== define paths for two datasets ==========
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "local path to the base model or the huggingface id"})
    sft_model_name: Optional[str] = field(default='./models/sft/', metadata={'help':"the path to the sft model; need to merge if using lora"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type: 'summary' or 'assistant'"}) 
    reward_name: Optional[str] = field(default='harmless', metadata={"help": "the reward model name: 'summary', 'faithful', 'helpful', 'harmless', 'deberta', 'humor'"})
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "loading model in 8 bit or bfloat16"})
    log_with: Optional[str] = field(default='none', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./models/ppo/', metadata={"help": "directory to save the model"})
    wandb_name: Optional[str] = field(default='assistant_ppo', metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config('config.yaml')['ppo_{}'.format(script_args.exp_type)]
print(f"Script arguments: {script_args}")
print(f"Training config: {cfg}")

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
print('output dir: ', output_dir)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id = accelerator.local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ========== load reward model ==========
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
reward_model = RewardModels([reward_peft_path], [rm_tokenizer_path], gpu_id)
rm_tokenizer = reward_model.rm_tokenizers[0] 

# ========== define training and lora configurations ==========
config = PPOConfig(
    model_name=script_args.sft_model_name,
    **cfg,
    log_with=script_args.log_with if script_args.log_with != 'none' else None,
    tracker_kwargs={"wandb":{"name":script_args.wandb_name if script_args.log_with != 'none' else None}},
)

sft_config = PeftConfig.from_pretrained(script_args.sft_model_name)
print("=" * 70)
print("SFT LoRA Configuration:")
print(f"  r: {sft_config.r}")
print(f"  lora_alpha: {sft_config.lora_alpha}")
print(f"  target_modules: {sft_config.target_modules}")
print("=" * 70)

lora_config = LoraConfig(
    r=sft_config.r,  
    lora_alpha=sft_config.lora_alpha,
    lora_dropout=sft_config.lora_dropout,
    target_modules=sft_config.target_modules,
    bias=sft_config.bias,
    task_type=sft_config.task_type,
)

# ========== load model and tokenizer ==========
tokenizer = load_main_tokenizer(script_args.sft_model_name)
if script_args.load_in_8bit:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.base_model_name,
        load_in_8bit=True,
        device_map=gpu_id,
    )
    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model)
    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model,
        script_args.sft_model_name,
        is_trainable=True
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model,
        script_args.sft_model_name,
        is_trainable=True
    )
print_trainable_parameters(model)
model.pretrained_model.resize_token_embeddings(len(tokenizer))

# ========== prepare dataset and dataloader ==========
if script_args.exp_type == 'assistant':
    dataset = build_dataset_ppo(hhrlhf_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
else:
    dataset = build_dataset_summary_ppo(summary_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}.")
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# ========== define ppo trainer and generation kwargs ==========
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

generation_kwargs = {
    "max_new_tokens": 128 if script_args.exp_type == 'assistant' else 48,
    'min_length': -1, 
    "top_k": 0.0,
    "top_p": 1.0, 
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id],
}

# ========== start training ==========
print("Training........")
model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True

epochs = cfg['epochs']
mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    'kl_std': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
}

global_step = 0
for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // ppo_trainer.config.batch_size // accelerator.num_processes)

    # save initial parameters for comparison
    print("Saving initial parameters for comparison...")
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.clone().detach()

    # check after the first batch
    first_batch_checked = False

    for i, batch in enumerate(ppo_trainer.dataloader):
        print('epoch {}, batch {}, global_step {}'.format(epoch, i, global_step))
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        
        full_responses = tokenizer.batch_decode(response_tensors)
        full_responses_clean = []
        for _, response in enumerate(full_responses):
            response = response.strip('[PAD] ')
            response = response.strip('<unk>')
            temp_resp = response.strip('<s>').strip('</s>')
            temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
            temp_resp = temp_resp.split('\nHuman:')[0].strip()
            temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
            temp_resp = temp_resp.split('\nAssistant:')[0].strip()
            temp_resp = temp_resp.split('###')[0].strip()
            temp_resp = temp_resp.split('\n\n\n')[0].strip()
            full_responses_clean.append(temp_resp)

        clean_texts = full_responses_clean
        clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]
        
        lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        response_tensors = [response_tensors[j][:np.max([lengths[j], 2])] for j in range(len(response_tensors))]
        batch['response'] = clean_texts

        # compute rewards
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
        print("epoch {}, batch {}, global_step {}: mean score: {:.4f}".format(epoch, i, global_step, torch.mean(torch.tensor(rewards)).item()))

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        # first batch diagnostics (only on first batch of first epoch)
        if global_step == 0:
            # register gradient hooks
            gradient_info = {}
            def grad_hook(name):
                def hook(grad):
                    gradient_info[name] = grad.norm().item()
                    return grad
                return hook
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.register_hook(grad_hook(name))

        # PPO step - this will compute gradients and update parameters
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)

        # print stats after PPO step
        print(f"  Raw reward: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
        print(f"  KL: {stats['objective/kl']:.4f}")
        print(f"  Policy loss: {stats.get('ppo/loss/policy', 'N/A')}")
        print(f"  Value loss: {stats.get('ppo/loss/value', 'N/A')}")
        print(f"  Entropy: {stats.get('ppo/policy/entropy', 'N/A')}")

        # warn if KL is too high
        if stats['objective/kl'] > 1.0:
            print("⚠️ WARNING: KL divergence is high!")

        # log stats
        ppo_trainer.log_stats(stats, batch, rewards)
        policy_kl = [stats["objective/kl"]]

        # first batch diagnostics (only on first batch of first epoch)
        if global_step == 0:
            print("\n" + "=" * 70)
            print("First Batch Diagnostics:")
            
            # check gradients
            if gradient_info:
                nonzero_grads = sum(1 for v in gradient_info.values() if v > 1e-8)
                if nonzero_grads:
                    print(f"  ✓ Gradients: {nonzero_grads}/{len(gradient_info)} parameters")
                else:
                    print("  ❌ NO GRADIENTS CAPTURED!")
            
            # check parameter updates
            params_changed = any(
                (param - initial_params[name]).abs().max().item() > 1e-8
                for name, param in model.named_parameters()
                if param.requires_grad and name in initial_params
            )
            print(f"  {'✓' if params_changed else '❌'} Parameters {'updated' if params_changed else 'NOT updated'}")
            print("=" * 70 + "\n")

        all_rewards = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)

        if ppo_trainer.accelerator.is_main_process:
            mean_scores.append(torch.mean(torch.tensor(rewards)).item())
            std_scores.append(torch.std(torch.tensor(rewards)).item())

            # save plot
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png')
            plt.clf()  # Clear figure to avoid overlapping plots
            plt.plot(mean_scores)
            plt.fill_between(
                np.arange(len(mean_scores)),
                np.array(mean_scores) - np.array(std_scores),
                np.array(mean_scores) + np.array(std_scores),
                alpha=0.5
            )
            plt.xlabel('Global Step')
            plt.ylabel('Reward')
            plt.title('Training Progress')
            plt.savefig(save_path)

            # save data
            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['kl_std'].append(np.std(all_policy_kl))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append(texts_merge[0])

            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(
                os.path.join(script_args.save_directory, script_args.wandb_name, 'data.csv'), 
                escapechar='\\'
            )
            print("epoch {}, batch {}, global_step {}: log saved".format(epoch, i, global_step))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)

        # save model checkpoint every 100 steps
        if ppo_trainer.accelerator.is_main_process and global_step % 100 == 0 and global_step != 0:
            save_path = os.path.join(
                script_args.save_directory, 
                script_args.wandb_name, 
                'step_{}'.format(global_step)
            )
            ppo_trainer.save_pretrained(save_path)
            print("epoch {}, batch {}, global_step {}: checkpoint saved".format(epoch, i, global_step))

        global_step += 1

    # save model at end of each epoch
    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(
            script_args.save_directory, 
            script_args.wandb_name, 
            'epoch_{}_final'.format(epoch)
        )
        ppo_trainer.save_pretrained(save_path)
        print("epoch {} complete, global_step {}: epoch checkpoint saved".format(epoch, global_step))

# save final model
if ppo_trainer.accelerator.is_main_process:
    save_path = os.path.join(
        script_args.save_directory, 
        script_args.wandb_name, 
        'final'
    )
    ppo_trainer.save_pretrained(save_path)
    print("Training complete! Final model saved at global_step {}".format(global_step))

            