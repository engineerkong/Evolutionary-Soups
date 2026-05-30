"""MORLHF — Multi-Objective RLHF via preference-weighted PPO.

Adapted from RiC/ppo/morlhf.py.

Changes vs original (RiC/ppo/morlhf.py):
  1. Imports: sys.path → MOMoE/baselines/utils instead of local RiC/ppo/utils.
     Uses baselines.utils.{utils, multi_reward_models} for all helpers.
  2. Dataset builders: build_dataset/build_dataset_summary replaced with
     build_dataset_ppo/build_dataset_summary_ppo from MOMoE utils.
  3. Beaver support: exp_type='beaver' added (PKU-Alignment/PKU-SafeRLHF-10K)
     using build_dataset_beaver_ppo; uses Instructions() (same format as assistant).
  4. Reward dict: extended with beaver_reward / beaver_cost entries.
  5. preference arg: changed from a single float to a comma-separated string so
     the full N-dimensional weight vector is specified directly (e.g. '0.2,0.4,0.4').
     The original hard-coded equal 1/N weights for N==3 is removed.
  6. Model loading: mirrors fine-tuning/ppo.py exactly —
       base_model_name = bare LLaMA weights (no LoRA)
       sft_model_name  = SFT LoRA adapter (output of fine-tuning/sft.py)
     Loads base model, then applies SFT LoRA via PeftModel.from_pretrained(...,
     is_trainable=True). No separate PPO LoRA is added; the SFT LoRA is trained
     directly by PPO. Previously, morlhf.py expected a *merged* SFT model and
     added a fresh LoRA on top — that breaks when sft.py saves an adapter only.
  7. PPO hyper-parameters: mini_batch_size and gradient_accumulation_steps now
     default to 8 (matching ppo.py). The original defaults of 1/1 caused 64×
     more optimizer steps per rollout (~3-10× wall-clock overhead).
  8. Best-model tracking: rolling-mean window of 10 batches; saves best_model/
     whenever a new high is reached (same logic as fine-tuning/ppo.py).
  9. Checkpoint naming: step_N (every 100 steps), epoch_N_final (end of epoch),
     final (end of training). Previously used batch_N which was ambiguous.
  10. Multi-GPU: single Accelerator() instance created once; process_id derived
      from accelerator.local_process_index (no redundant Accelerator() calls).
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

script_dir   = Path(__file__).resolve().parent        # baselines/morlhf/
project_root = script_dir.parent.parent.parent         # MOMoE/
sys.path.insert(0, str(project_root))

from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np
import pandas as pd
from peft import PeftModel, PeftConfig, prepare_model_for_kbit_training
import matplotlib.pyplot as plt

from baselines.utils.utils import (
    print_trainable_parameters, load_main_tokenizer,
    Instructions, Instructions_summary,
    build_dataset_ppo         as build_dataset,
    build_dataset_summary_ppo as build_dataset_summary,
    build_dataset_beaver_ppo  as build_dataset_beaver,
)
from baselines.utils.multi_reward_models import RewardModels

tqdm.pandas()

hhrlhf_dataset_path  = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
beaver_dataset_path  = 'PKU-Alignment/PKU-SafeRLHF-10K'

_ROLLING_WINDOW = 10   # rolling-mean window for best-model tracking


@dataclass
class ScriptArguments:
    log_with:                    Optional[str]   = field(default='wandb')
    disable_wandb:               Optional[str]   = field(default=True)
    save_directory:              Optional[str]   = field(default='./results/morlhf/')
    epochs:                      Optional[int]   = field(default=1)
    learning_rate:               Optional[float] = field(default=1e-5)
    mini_batch_size:             Optional[int]   = field(default=8)
    batch_size:                  Optional[int]   = field(default=64)
    gradient_accumulation_steps: Optional[int]   = field(default=8)
    early_stopping:              Optional[bool]  = field(default=True)
    target:                      Optional[float] = field(default=3)
    init_kl_coef:                Optional[float] = field(default=0.2)
    max_grad_norm:               Optional[float] = field(default=0.5)
    load_in_8bit:                Optional[bool]  = field(default=False)
    preference:                  Optional[str]   = field(default='0.5,0.5', metadata={"help": "comma-separated preference weights, one per reward (must sum to 1)"})
    wandb_name:                  Optional[str]   = field(default='morlhf')
    base_model_name:             Optional[str]   = field(default='meta-llama/Llama-2-7b-hf',                   metadata={'help': 'base LLaMA model (local path or HF id)'})
    sft_model_name:              Optional[str]   = field(default='./models/sft/sft_assistant_2701/model/',     metadata={'help': 'SFT LoRA adapter path (output of fine-tuning/sft.py)'})
    reward_names:                Optional[str]   = field(default='harmless,helpful,humor')
    exp_type:                    Optional[str]   = field(default='assistant',                                  metadata={"help": "assistant | summary | beaver"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type   = script_args.exp_type
preference = [round(float(p), 2) for p in script_args.preference.split(',')]
pref_str   = '_'.join(str(p) for p in preference)
script_args.wandb_name = script_args.wandb_name + '_pref' + pref_str

base_model_name = script_args.base_model_name
sft_model_name  = script_args.sft_model_name
tokenier_name   = sft_model_name
print('base model: ', base_model_name)
print('sft model:  ', sft_model_name)

if script_args.disable_wandb:
    os.environ['WANDB_DISABLED'] = 'true'

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards  = len(reward_names)
print('number of rewards: {}'.format(num_rewards))
print('preference: {}'.format(preference))

assert len(preference) == num_rewards, \
    f'--preference has {len(preference)} values but {num_rewards} rewards were specified'

reward_path_tokenizer_dict = {
    'harmless':      ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful':       ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta':       ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary':       ['Tristan/gpt2_reward_summarization'],
    'faithful':      ['CogComp/bart-faithful-summary-detector'],
    'humor':         ['mohameddhiab/humor-no-humor'],
    'beaver_reward': ['PKU-Alignment/beaver-7b-v1.0-reward'],
    'beaver_cost':   ['PKU-Alignment/beaver-7b-v1.0-cost'],
}

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f'Unknown reward: {name}')
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

config = PPOConfig(
    model_name=sft_model_name,
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
    tracker_project_name='morlhf',
    tracker_kwargs={"wandb": {"name": script_args.wandb_name}},
)

accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


set_seed(8888)

sft_peft_config = PeftConfig.from_pretrained(sft_model_name)
print(f"SFT LoRA  r={sft_peft_config.r}  alpha={sft_peft_config.lora_alpha}"
      f"  modules={sft_peft_config.target_modules}")

tokenizer = load_main_tokenizer(tokenier_name)

if exp_type == 'assistant':
    dataset      = build_dataset(hhrlhf_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif exp_type == 'summary':
    dataset      = build_dataset_summary(summary_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
else:  # beaver
    dataset      = build_dataset_beaver(beaver_dataset_path, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()

train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}")

if script_args.load_in_8bit:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name, load_in_8bit=True, device_map=gpu_id,
    )
    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model)
    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model, sft_model_name, is_trainable=True,
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map=gpu_id,
    )
    model.pretrained_model = PeftModel.from_pretrained(
        model.pretrained_model, sft_model_name, is_trainable=True,
    )

print_trainable_parameters(model)
model.pretrained_model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.learning_rate,
)

ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer,
    dataset=dataset, data_collator=collator, optimizer=optimizer,
)

generation_kwargs = {
    "max_new_tokens": 128 if exp_type in ('assistant', 'beaver') else 48,
    "min_length": -1,
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

epochs      = script_args.epochs
mean_scores = []
std_scores  = []
save_data   = {
    'kl_mean': [], 'reward_mean': [], 'reward_std': [],
    'text_sample': [], 'batch_time': [], 'total_time': [],
}
best_score       = float('-inf')
best_score_label = None
recent_scores    = []
t_start      = time.time()
global_step  = 0

for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)
    for i, batch in enumerate(ppo_trainer.dataloader):
        t_batch_start = time.time()
        print('epoch {}, global_step {}'.format(epoch, global_step))
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True

        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)

        full_responses = tokenizer.batch_decode(response_tensors)
        full_responses_clean = []
        for _, response in enumerate(full_responses):
            response  = response.strip('[PAD] ').strip('<unk>')
            temp_resp = response.strip('<s>').strip('</s>')
            temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
            temp_resp = temp_resp.split('\nHuman:')[0].strip()
            temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
            temp_resp = temp_resp.split('\nAssistant:')[0].strip()
            temp_resp = temp_resp.split('\n\n\n')[0].strip()
            temp_resp = temp_resp.split('###')[0].strip()
            full_responses_clean.append(temp_resp)

        clean_texts            = full_responses_clean
        clean_response_tensors = [tokenizer.encode(t) for t in clean_texts]
        lengths                = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        response_tensors       = [response_tensors[j][:np.max([lengths[j], 2])] for j in range(len(response_tensors))]
        batch['response']      = clean_texts

        texts_merge       = [q + r for q, r in zip(batch['query'], batch['response'])]
        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in texts_merge
        ]
        if hasattr(instructions, 'get_post'):
            rewards_list = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)
        else:
            rewards_list = reward_model.get_reward_model_scores(queries_responses)

        rewards = [
            round(sum(preference[k] * rewards_list[k][j] for k in range(num_rewards)), 2)
            for j in range(len(queries_responses))
        ]
        batch_mean     = float(np.mean(rewards))
        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("epoch {}, global_step {}, mean score: {:.4f}".format(epoch, global_step, batch_mean))

        recent_scores.append(batch_mean)
        if len(recent_scores) > _ROLLING_WINDOW:
            recent_scores.pop(0)

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats     = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        policy_kl = [stats["objective/kl"]]
        ppo_trainer.log_stats(stats, batch, rewards)

        all_rewards   = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)

        if ppo_trainer.accelerator.is_main_process:
            mean_scores.append(float(np.mean(all_rewards)))
            std_scores.append(float(np.std(all_rewards)))
            t_now = time.time()
            save_data['batch_time'].append(t_now - t_batch_start)
            save_data['total_time'].append(t_now - t_start)
            save_data['kl_mean'].append(float(np.mean(all_policy_kl)))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std']  = std_scores
            save_data['text_sample'].append(texts_merge[0])

            plt.clf()
            plt.plot(mean_scores)
            plt.fill_between(
                np.arange(len(mean_scores)),
                np.array(mean_scores) - np.array(std_scores),
                np.array(mean_scores) + np.array(std_scores),
                alpha=0.5,
            )
            plt.xlabel('Global Step')
            plt.ylabel('Weighted Reward')
            plt.savefig(os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png'))

            pd.DataFrame(save_data).to_csv(
                os.path.join(script_args.save_directory, script_args.wandb_name, 'data.csv'),
                escapechar='\\',
            )

        accelerator.wait_for_everyone()
        pbar.update(1)

        if ppo_trainer.accelerator.is_main_process and global_step % 100 == 0 and global_step != 0:
            ckpt_path = os.path.join(script_args.save_directory, script_args.wandb_name,
                                     'step_{}'.format(global_step))
            ppo_trainer.save_pretrained(ckpt_path)
            print("  checkpoint saved: step_{}".format(global_step))
            rolling_mean = sum(recent_scores) / len(recent_scores)
            if rolling_mean > best_score:
                best_score       = rolling_mean
                best_score_label = 'step_{}'.format(global_step)
                ppo_trainer.save_pretrained(
                    os.path.join(script_args.save_directory, script_args.wandb_name, 'best_model'))
                print("  → new best model (rolling_mean={:.4f}) saved as best_model/".format(best_score))

        global_step += 1

    if ppo_trainer.accelerator.is_main_process:
        ckpt_path = os.path.join(script_args.save_directory, script_args.wandb_name,
                                 'epoch_{}_final'.format(epoch))
        ppo_trainer.save_pretrained(ckpt_path)
        print("epoch {} complete: checkpoint saved".format(epoch))
        rolling_mean = sum(recent_scores) / len(recent_scores)
        if rolling_mean > best_score:
            best_score       = rolling_mean
            best_score_label = 'epoch_{}_final'.format(epoch)
            ppo_trainer.save_pretrained(
                os.path.join(script_args.save_directory, script_args.wandb_name, 'best_model'))
            print("  → new best model (rolling_mean={:.4f}) saved as best_model/".format(best_score))

if ppo_trainer.accelerator.is_main_process:
    ppo_trainer.save_pretrained(
        os.path.join(script_args.save_directory, script_args.wandb_name, 'final'))
    print("Training complete! Final model saved at global_step {}".format(global_step))
    print("Best model: '{}' (rolling_mean={:.4f}) → saved as best_model/".format(
        best_score_label, best_score))
