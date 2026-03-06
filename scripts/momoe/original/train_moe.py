import sys
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional, List
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import set_seed
import pandas as pd
# adapt versions
from moe_architecture import MoEGatingTrainer
from moe_utils import (
    convert_to_moe_model,
    save_moe_gating_weights
)

script_dir = Path(__file__).resolve().parent  # project/scripts/momoe
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.utils import (
    load_config,
    print_trainable_parameters, 
    load_main_tokenizer, 
    build_dataset_ppo, 
    build_dataset_summary_ppo,
    Instructions,
    Instructions_summary
)
from scripts.utils.multi_reward_models import RewardModels

# ========== define paths for two datasets ==========
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ==================== define script arguments ====================
@dataclass
class ScriptArguments:
    base_model_name: str = field(default="./models/sft/model/", metadata={"help": "local path to the sft model"})
    lora_expert_paths: List[str] = field(default_factory=lambda: [], metadata={"help": "list of paths to LoRA expert models"})
    num_pref_samples: int = field(default=10, metadata={"help": "number of preference samples per input during training"})
    reward_names: Optional[str] = field(default='harmless,helpful', metadata={"help": "comma separated list of reward names"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    log_with: Optional[str] = field(default='none', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: str = field(default="./models/momoe/", metadata={"help": "directory to save the model"})
    wandb_name: str = field(default="assistant_moemoe", metadata={"help": "name for this experiment"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config(script_dir / 'config.yaml')
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
rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])

# ========== convert to MoE model ==========
moe_model = convert_to_moe_model(
    base_model_name=script_args.base_model_name,
    lora_expert_paths=script_args.lora_expert_paths,
    subspace_rank=cfg['subspace_rank'],
    d_model=cfg['d_model'],
    num_rewards=len(reward_names),
    target_device=f"cuda:{gpu_id}"
)
moe_model = accelerator.prepare(moe_model)
tokenizer = load_main_tokenizer(script_args.base_model_name)

# ========== prepare dataset and dataloader ==========
if script_args.exp_type == 'assistant':
    dataset = build_dataset_ppo(
        hhrlhf_dataset_path,
        tokenizer,
        rm_tokenizer,
        split='train'
    )
    instructions = Instructions()
else:
    dataset = build_dataset_summary_ppo(
        summary_dataset_path,
        tokenizer,
        rm_tokenizer,
        split='train'
    )
    instructions = Instructions_summary()
print(f"Dataset size: {len(dataset)}")
if accelerator.num_processes > 1:
    # Deterministic shard sizing (no collective needed): each shard is either
    # floor(N/world_size) or floor(N/world_size)+1. We trim to floor(N/world_size)
    # to guarantee identical number of optimizer steps across ranks.
    min_len = len(dataset) // accelerator.num_processes
    dataset = dataset.shard(
        num_shards=accelerator.num_processes,
        index=accelerator.process_index,
        contiguous=True
    )
    print(f"Rank {accelerator.process_index}: local shard size {len(dataset)}")
    if len(dataset) > min_len:
        dataset = dataset.select(range(min_len))
    print(f"Rank {accelerator.process_index}: synchronized shard size {len(dataset)}")

# ========== start training ==========
trainer = MoEGatingTrainer(
    moe_model=moe_model,
    reward_models=reward_models,
    instructions=instructions,
    learning_rate=float(cfg['learning_rate']),
    num_rewards=len(reward_names),
    num_pref_samples=script_args.num_pref_samples
)
print_trainable_parameters(moe_model)

stats = {
    'rewards': [],
    'balance_losses': [],
    'policy_losses': [],
    'entropy_losses': [],
    'hypervolume_losses': [],
    'hypervolume_values': [],
    'total_losses': [],
    'reward_mean': [],
    'reward_std': [],
}

epochs = cfg['epochs']
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    dataset_shuffled = dataset.shuffle(seed=epoch)
    pbar = tqdm(total=len(dataset_shuffled) // cfg['batch_size'])
    
    for i in range(0, len(dataset_shuffled), cfg['batch_size']):
        batch_data = dataset_shuffled[i:i+cfg['batch_size']]
        
        losses = trainer.train_step_reinforce(
            batch_data,
            tokenizer,
            alpha_balance=cfg['alpha_balance'],
            alpha_entropy=cfg['alpha_entropy'],
            alpha_hypervolume=cfg['alpha_hypervolume']
        )
        
        # record stats
        stats['rewards'].append(losses['mean_reward'])
        stats['balance_losses'].append(losses['balance_loss'])
        stats['policy_losses'].append(losses.get('policy_loss', 0.0))
        stats['entropy_losses'].append(losses.get('entropy_loss', 0.0))
        stats['hypervolume_losses'].append(losses.get('hypervolume_loss', 0.0))
        stats['hypervolume_values'].append(losses.get('hypervolume_value', 0.0)) # ?
        stats['total_losses'].append(losses['total_loss'])
        stats['reward_mean'].append(losses['mean_reward'])
        stats['reward_std'].append(losses.get('std_reward', 0.0))
        
        # print
        if i % 100 == 0:
            print(f"\nBatch {i // cfg['batch_size']}: "
                    f"Reward: {losses['mean_reward']:.4f}, "
                    f"Balance: {losses['balance_loss']:.4f}, "
                    f"Policy: {losses.get('policy_loss', 0.0):.4f}, "
                    f"Entropy: {losses.get('entropy_loss', 0.0):.4f}, "
                    f"HV Loss: {losses.get('hypervolume_loss', 0.0):.4f}, "
                    f"HV Value: {losses.get('hypervolume_value', 0.0):.4f}, "
                    f"Total: {losses['total_loss']:.4f}")
        
        pbar.update(1)
        
        # Regular checkpoint
        if i > 0 and i % 1000 == 0:
            save_path = os.path.join(
                script_args.save_directory,
                script_args.wandb_name,
                f'batch_{i}'
            )
            os.makedirs(save_path, exist_ok=True)
            
            # Save only gating weights and tokenizer
            if accelerator.is_main_process:
                save_moe_gating_weights(moe_model, save_path)
                tokenizer.save_pretrained(save_path)
                
                # Save training stats
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'baseline': trainer.reward_baseline,
                    'stats': stats
                }, os.path.join(save_path, 'training_state.pt'))
                
                print(f"\nCheckpoint saved to {save_path}")
    
    pbar.close()
    
    # Epoch end save
    save_path = os.path.join(
        script_args.save_directory,
        script_args.wandb_name,
        f'epoch_{epoch}_final'
    )
    os.makedirs(save_path, exist_ok=True)
    if accelerator.is_main_process:
        save_moe_gating_weights(moe_model, save_path)
        tokenizer.save_pretrained(save_path)
        
        # save training stats
        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(output_dir, 'data.csv'))

if accelerator.is_main_process:
    print("Training complete!")
    print(f"Final model saved to: {save_path}")
