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
from scripts.utils.utils import load_config, Instructions, Instructions_steer, Instructions_summary, \
    build_dataset_ppo, build_dataset_summary_ppo, build_dataset_news_summary_ppo, \
    build_dataset_beaver_ppo, build_dataset_steer_ppo, \
    load_main_tokenizer, print_trainable_parameters
from scripts.utils.multi_reward_models import RewardModels
tqdm.pandas()

SUMMARIZATION_DATASETS = {'openai/summarize_from_feedback', 'argilla/news-summary'}

_PPO_CONFIG_KEY = {
    'Anthropic/hh-rlhf':                    'ppo_assistant',
    'openai/summarize_from_feedback':        'ppo_summary',
    'argilla/news-summary':                  'ppo_news_summary',
    'PKU-Alignment/PKU-SafeRLHF-10K':        'ppo_beaver',
    'nvidia/HelpSteer':                      'ppo_steer',
    'nvidia/HelpSteer2':                     'ppo_steer2',
}

# ========== define script arguments ==========
@dataclass
class ScriptArguments:
    base_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "local path to the base model or the huggingface id"})
    sft_model_name: Optional[str] = field(default='./models/sft/', metadata={'help':"the path to the sft model; need to merge if using lora"})
    dataset_name: Optional[str] = field(default='Anthropic/hh-rlhf', metadata={"help": "dataset: 'Anthropic/hh-rlhf', 'openai/summarize_from_feedback', or 'argilla/news-summary'"})
    reward_name: Optional[str] = field(default='harmless', metadata={"help": "the reward model name: 'summary', 'faithful', 'helpful', 'harmless', 'deberta', 'humor', 'beaver_reward', 'beaver_cost', 'steer_helpfulness', 'steer_correctness', 'steer_coherence', 'steer_complexity', 'steer_verbosity'"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "loading model in 8 bit or bfloat16"})
    log_with: Optional[str] = field(default='none', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./models/ppo/', metadata={"help": "directory to save the model"})
    wandb_name: Optional[str] = field(default='assistant_ppo', metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg_key = _PPO_CONFIG_KEY.get(script_args.dataset_name)
if cfg_key is None:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}. '
                     f'Choose from: {list(_PPO_CONFIG_KEY.keys())}')
cfg = load_config(script_dir / 'config.yaml')[cfg_key]
# 'epochs' drives the outer training loop; not a valid PPOConfig kwarg in TRL 0.8+
epochs = cfg.pop('epochs', 1)
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
_URM = 'urm://LxzGordon/URM-LLaMa-3.1-8B'
_NEMOTRON = 'nemotron://nvidia/llama-3.1-nemotron-70b-reward'
_STEERLM = 'steerlm://nvidia/Llama2-13B-SteerLM-RM'
_REWARD_PATH_MAP = {
    'summary':                  'Tristan/gpt2_reward_summarization',
    'faithful':                 'CogComp/bart-faithful-summary-detector',
    'helpful':                  'Ray2333/gpt2-large-helpful-reward_model',
    'harmless':                 'Ray2333/gpt2-large-harmless-reward_model',
    'deberta':                  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'humor':                    'mohameddhiab/humor-no-humor',
    'beaver_reward':            'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':              'PKU-Alignment/beaver-7b-v1.0-cost',
    'steer_helpfulness':        f'{_URM}#0',
    'steer_correctness':        f'{_URM}#1',
    'steer_coherence':          f'{_URM}#2',
    'steer_complexity':         f'{_URM}#3',
    'steer_verbosity':          f'{_URM}#4',
    'nemotron':                 _NEMOTRON,
    'steerlm_helpfulness':      f'{_STEERLM}#4',
    'steerlm_correctness':      f'{_STEERLM}#5',
    'steerlm_coherence':        f'{_STEERLM}#6',
    'steerlm_complexity':       f'{_STEERLM}#7',
    'steerlm_verbosity':        f'{_STEERLM}#8',
}
if reward_name not in _REWARD_PATH_MAP:
    raise NotImplementedError(f'Unknown reward name: {reward_name!r}')
reward_peft_path = _REWARD_PATH_MAP[reward_name]
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
_normalise_query_for_rm = False  # overridden below for HelpSteer2 + gpt2 rewards
if script_args.dataset_name == 'Anthropic/hh-rlhf':
    dataset = build_dataset_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    dataset = build_dataset_summary_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'argilla/news-summary':
    # argilla/news-summary: train split has only ~1000 samples; test split has ~20k — use test for training
    dataset = build_dataset_news_summary_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='test')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    dataset = build_dataset_beaver_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    dataset = build_dataset_steer_ppo(script_args.dataset_name, tokenizer, rm_tokenizer, split='train')
    _steer_reward = reward_name.startswith('steer') or reward_name.startswith('steerlm')
    if _steer_reward:
        # URM/SteerLM apply their own chat template and expect raw (q, r).
        instructions = Instructions_steer(script_args.dataset_name)
        _normalise_query_for_rm = False
    else:
        # gpt2-based rewards were trained on "\n\nHuman:…\n\nAssistant:" format.
        # HelpSteer2 prompts use "\n\nuser:" — flag for substitution at reward time.
        instructions = Instructions()
        _normalise_query_for_rm = True
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}')
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}.")
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# ========== define ppo trainer and generation kwargs ==========
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config.learning_rate))
ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

generation_kwargs = {
    "max_new_tokens": 128 if script_args.dataset_name in {'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K', 'nvidia/HelpSteer', 'nvidia/HelpSteer2'} else 48,
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

mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    'kl_std': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
}

best_score = float('-inf')
best_score_label = None
_ROLLING_WINDOW = 10  # rolling mean window for best-model tracking
recent_scores = []


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
            temp_resp = temp_resp.split('\n\nuser:')[0].strip()
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
        _rm_texts = [t.replace('\n\nuser: ', '\n\nHuman: ') for t in texts_merge] \
                    if _normalise_query_for_rm else texts_merge
        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in _rm_texts
        ]
        if global_step == 0 and accelerator.is_main_process:
            q0, r0 = queries_responses[0]
            print("\n" + "=" * 70)
            print("Reward model input sample (global_step 0):")
            print(f"  [QUERY]    {repr(q0)}")
            print(f"  [RESPONSE] {repr(r0)}")
            print("=" * 70 + "\n")

        if hasattr(instructions, 'get_post'):
            rewards = reward_model.get_reward_model_scores(queries_responses, instructions.get_post, normalize_rewards=False)[0]
        else:
            rewards = reward_model.get_reward_model_scores(queries_responses, normalize_rewards=False)[0]
        batch_mean = float(np.mean(rewards))
        batch_std  = float(np.std(rewards))
        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("epoch {}, batch {}, global_step {}: mean score: {:.4f}".format(epoch, i, global_step, batch_mean))
        recent_scores.append(batch_mean)
        if len(recent_scores) > _ROLLING_WINDOW:
            recent_scores.pop(0)

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
        print(f"  Raw reward: mean={batch_mean:.3f}, std={batch_std:.3f}")
        print(f"  KL: {stats['objective/kl']:.4f}")
        print(f"  Policy loss: {stats.get('ppo/loss/policy', 'N/A')}")
        print(f"  Value loss: {stats.get('ppo/loss/value', 'N/A')}")
        print(f"  Entropy: {stats.get('ppo/policy/entropy', 'N/A')}")

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
            mean_scores.append(batch_mean)
            std_scores.append(batch_std)

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
            rolling_mean = sum(recent_scores) / len(recent_scores)
            if rolling_mean > best_score:
                best_score = rolling_mean
                best_score_label = 'step_{}'.format(global_step)
                best_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'best_model')
                ppo_trainer.save_pretrained(best_path)
                print("  → new best model (rolling_mean={:.4f}) saved as best_model".format(best_score))

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
        rolling_mean = sum(recent_scores) / len(recent_scores)
        if rolling_mean > best_score:
            best_score = rolling_mean
            best_score_label = 'epoch_{}_final'.format(epoch)
            best_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'best_model')
            ppo_trainer.save_pretrained(best_path)
            print("  → new best model (rolling_mean={:.4f}) saved as best_model".format(best_score))

# save final model
if ppo_trainer.accelerator.is_main_process:
    save_path = os.path.join(
        script_args.save_directory,
        script_args.wandb_name,
        'final'
    )
    ppo_trainer.save_pretrained(save_path)
    print("Training complete! Final model saved at global_step {}".format(global_step))
    print("Best model: '{}' (rolling_mean={:.4f}) → saved as best_model/".format(best_score_label, best_score))

            