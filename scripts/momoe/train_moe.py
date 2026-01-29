import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from dataclasses import dataclass, field
from typing import Optional, List
from accelerate import Accelerator
from tqdm import tqdm
from transformers import HfArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from moe_architecture import (
    LoRAExpertFFNComplete,
    MoEFFNLayer,
    RewardModels,
    AttentionGatingNetwork,
    MoEGatingTrainer
)
from utils import (
    print_trainable_parameters, 
    load_main_tokenizer, 
    build_dataset, 
    build_dataset_summary,
    Instructions,
    Instructions_summary
)

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

# ==================== 模型转换 ====================

def convert_to_moe_model(base_model_name, lora_expert_paths, subspace_rank=8, d_model=512, num_rewards=2, target_device=None):
    """
    将 LLaMA base model 转换为 MoE 模型 (Preference-Conditioned)
    
    Args:
        base_model_name: base model 路径
        lora_expert_paths: LoRA expert 路径列表
        subspace_rank: 子空间秩
        d_model: 注意力机制的隐藏维度
        num_rewards: number of reward objectives for preference conditioning
        target_device: target device for the model (for distributed training)
    
    Returns:
        moe_model: 转换后的 MoE 模型
    """
    print(f"Loading base model: {base_model_name}")
    if target_device is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=target_device
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype 
    print(f"Base model is on device: {model_device}, dtype: {model_dtype}")

    # 加载 LoRA experts - load state dicts only for efficiency
    print(f"Loading {len(lora_expert_paths)} LoRA experts...")
    lora_state_dicts = []
    for expert_path in lora_expert_paths:
        print(f"  Loading: {expert_path}")
        safetensor_path = os.path.join(expert_path, "adapter_model.safetensors")
        pytorch_path = os.path.join(expert_path, "adapter_model.bin")
        if os.path.exists(safetensor_path):
            from safetensors import safe_open
            state_dict = {}
            with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            lora_state_dicts.append(state_dict)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
            lora_state_dicts.append(state_dict)
        else:
            lora_model = PeftModel.from_pretrained(base_model, expert_path)
            lora_state_dicts.append(lora_model.state_dict())
            del lora_model
    
    num_lora_experts = len(lora_state_dicts)
    hidden_dim = base_model.config.hidden_size
    num_layers = base_model.config.num_hidden_layers
    
    print(f"Converting {num_layers} layers to MoE (Preference-Conditioned)...")
    
    # Helper to create LoRA layer from state dict
    def create_lora_layer(state_dict, layer_idx, proj_name, device, dtype):
        lora_A_key = None
        lora_B_key = None
        for key in state_dict.keys():
            if f"layers.{layer_idx}.mlp.{proj_name}.lora_A" in key:
                lora_A_key = key
            elif f"layers.{layer_idx}.mlp.{proj_name}.lora_B" in key:
                lora_B_key = key
        if lora_A_key is None or lora_B_key is None:
            raise KeyError(f"Could not find LoRA weights for layer {layer_idx} {proj_name}")
        
        lora_A_weight = state_dict[lora_A_key].to(device=device, dtype=dtype)
        lora_B_weight = state_dict[lora_B_key].to(device=device, dtype=dtype)
        
        class LoraWrapper(nn.Module):
            def __init__(self, lora_A_w, lora_B_w):
                super().__init__()
                self.lora_A = nn.ModuleDict({'default': nn.Module()})
                self.lora_B = nn.ModuleDict({'default': nn.Module()})
                self.lora_A['default'].weight = nn.Parameter(lora_A_w)
                self.lora_B['default'].weight = nn.Parameter(lora_B_w)
            def forward(self, x):
                return (x @ self.lora_A['default'].weight.t()) @ self.lora_B['default'].weight.t()
        
        return LoraWrapper(lora_A_weight, lora_B_weight)
    
    # 遍历每一层，替换 MLP
    for layer_idx in range(num_layers):
        if layer_idx % 5 == 0:
            print(f"Processing layer {layer_idx}/{num_layers}...")
        
        base_mlp = base_model.model.layers[layer_idx].mlp
        layer_device = next(base_mlp.parameters()).device
        
        lora_experts = []
        for state_dict in lora_state_dicts:
            gate_lora = create_lora_layer(state_dict, layer_idx, 'gate_proj', layer_device, model_dtype)
            up_lora = create_lora_layer(state_dict, layer_idx, 'up_proj', layer_device, model_dtype)
            down_lora = create_lora_layer(state_dict, layer_idx, 'down_proj', layer_device, model_dtype)
            
            expert = LoRAExpertFFNComplete(
                base_gate_proj=base_mlp.gate_proj,
                base_up_proj=base_mlp.up_proj,
                base_down_proj=base_mlp.down_proj,
                gate_proj_lora=gate_lora,
                up_proj_lora=up_lora,
                down_proj_lora=down_lora,
                act_fn=nn.SiLU()
            )
            lora_experts.append(expert)
        
        gate_network = AttentionGatingNetwork(
            hidden_dim=hidden_dim,
            num_lora_experts=num_lora_experts,
            subspace_rank=subspace_rank,
            d_model=d_model,
            num_rewards=num_rewards
        )
        gate_network = gate_network.to(layer_device, dtype=model_dtype)

        for expert_idx, state_dict in enumerate(lora_state_dicts):
            gate_lora = create_lora_layer(state_dict, layer_idx, 'gate_proj', layer_device, model_dtype)
            up_lora = create_lora_layer(state_dict, layer_idx, 'up_proj', layer_device, model_dtype)
            down_lora = create_lora_layer(state_dict, layer_idx, 'down_proj', layer_device, model_dtype)
            gate_network.load_expert_embedding(
                expert_idx,
                gate_lora=gate_lora,
                up_lora=up_lora,
                down_lora=down_lora
            )
        
        moe_ffn = MoEFFNLayer(
            base_mlp=base_mlp,
            lora_experts=lora_experts,
            gate_network=gate_network
        )
        
        base_model.model.layers[layer_idx].mlp = moe_ffn
    
    print("MoE conversion complete!")
    return base_model


# ==================== MoE Gating Weights Save/Load ====================

def save_moe_gating_weights(model, save_path):
    """Save only the trainable gating network weights"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    gating_state_dict = {}
    for name, param in model.named_parameters():
        if 'gate' in name and param.requires_grad:
            gating_state_dict[name] = param.cpu()
    
    torch.save(gating_state_dict, os.path.join(save_path, 'gating_weights.pt'))
    print(f"Saved gating weights to {save_path}/gating_weights.pt")

def load_moe_gating_weights(model, save_path):
    """Load gating network weights"""
    import os
    weights_path = os.path.join(save_path, 'gating_weights.pt')
    
    if os.path.exists(weights_path):
        gating_state_dict = torch.load(weights_path, map_location=model.device)
        model.load_state_dict(gating_state_dict, strict=False)
        print(f"Loaded gating weights from {weights_path}")
    else:
        print(f"No gating weights found at {weights_path}")


# ==================== MoE Gating Trainer ====================

@dataclass
class ScriptArguments:
    base_model_name: str = field(default="meta-llama/Llama-2-7b-hf")
    lora_expert_paths: List[str] = field(default_factory=lambda: [])
    dataset_name: str = field(default="Anthropic/hh-rlhf")
    save_directory: str = field(default="./moe_models/")
    wandb_name: str = field(default="moe_gating_training")
    epochs: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    batch_size: int = field(default=64)
    subspace_rank: int = field(default=8)
    d_model: int = field(default=512)
    alpha_balance: float = field(default=0.01)
    alpha_entropy: float = field(default=0.01)
    alpha_hypervolume: float = field(default=0.1, metadata={"help": "coefficient for hypervolume loss"})
    disable_wandb: bool = field(default=True)
    reward_names: str = field(default='harmless,helpful')
    # preference: Optional[float] = field(default=0.5, metadata={"help": "the weight for reward 1"})
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    num_pref_samples: int = field(default=10, metadata={"help": "number of preference samples per input during training"})


# ==================== 主训练流程 ====================

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Parse reward names and preferences (align with reference)
    reward_names = [x.strip() for x in args.reward_names.split(',')]
    num_rewards = len(reward_names)
    print('number of rewards: {}'.format(num_rewards))
    
    # # Calculate preferences
    # if num_rewards == 2:
    #     preference = [round(args.preference, 1), round(1 - args.preference, 1)]
    # else:
    #     preference = [round(1 / num_rewards, 2) for _ in range(num_rewards)]
    
    # print('preference: {}'.format(preference))
    
    # # Update wandb name with preference
    # args.wandb_name = args.wandb_name + '_pref{}_{}_hv'.format(preference[0], preference[1] if num_rewards == 2 else 'multi')
    
    if args.disable_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    
    # 创建保存目录
    os.makedirs(os.path.join(args.save_directory, args.wandb_name), exist_ok=True)
    
    # Initialize Accelerator first to get correct device
    accelerator = Accelerator()
    gpu_id = accelerator.local_process_index
    target_device = f"cuda:{gpu_id}"
    print(f'Process {gpu_id} using device: {target_device}')
    
    # 1. 转换为 MoE 模型
    print("\n" + "=" * 70)
    print("Step 1: Converting to MoE model (Preference-Conditioned)")
    print("=" * 70)
    
    moe_model = convert_to_moe_model(
        base_model_name=args.base_model_name,
        lora_expert_paths=args.lora_expert_paths,
        subspace_rank=args.subspace_rank,
        d_model=args.d_model,
        num_rewards=num_rewards,
        target_device=target_device
    )

    tokenizer = load_main_tokenizer(args.base_model_name)
    
    # 2. 加载 Reward Model (aligned with reference)
    print("\n" + "=" * 70)
    print("Step 2: Loading reward model")
    print("=" * 70)
    
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

    reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])
    
    # 3. 加载数据集 (aligned with reference)
    print("\n" + "=" * 70)
    print("Step 3: Loading dataset")
    print("=" * 70)
    
    if args.exp_type == 'assistant':
        dataset = build_dataset(
            hhrlhf_dataset_path,
            tokenizer,
            rm_tokenizer,
            split='train'
        )
        instructions = Instructions()
    else:
        dataset = build_dataset_summary(
            summary_dataset_path,
            tokenizer,
            rm_tokenizer,
            split='train'
        )
        instructions = Instructions_summary()
    
    print(f"Dataset size: {len(dataset)}")
    
    # 4. 创建 Trainer (Preference-Conditioned)
    print("\n" + "=" * 70)
    print("Step 4: Creating trainer (Preference-Conditioned with Hypervolume)")
    print("=" * 70)
    
    trainer = MoEGatingTrainer(
        moe_model=moe_model,
        reward_model=reward_model,
        instructions=instructions,
        learning_rate=args.learning_rate,
        num_rewards=num_rewards,
        # preference=preference,
        num_pref_samples=args.num_pref_samples
    )
    
    # 打印可训练参数
    print_trainable_parameters(moe_model)
    
    # 5. 训练循环
    print("\n" + "=" * 70)
    print("Step 5: Training with Preference Conditioning and Hypervolume Loss")
    print("=" * 70)
    
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
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 创建 dataloader
        dataset_shuffled = dataset.shuffle(seed=epoch)
        
        pbar = tqdm(total=len(dataset_shuffled) // args.batch_size)
        
        for i in range(0, len(dataset_shuffled), args.batch_size):
            batch_data = dataset_shuffled[i:i+args.batch_size]
            
            losses = trainer.train_step_reinforce(
                batch_data,
                tokenizer,
                alpha_balance=args.alpha_balance,
                alpha_entropy=args.alpha_entropy,
                alpha_hypervolume=args.alpha_hypervolume
            )
            
            # 记录统计
            stats['rewards'].append(losses['mean_reward'])
            stats['balance_losses'].append(losses['balance_loss'])
            stats['policy_losses'].append(losses.get('policy_loss', 0.0))
            stats['entropy_losses'].append(losses.get('entropy_loss', 0.0))
            stats['hypervolume_losses'].append(losses.get('hypervolume_loss', 0.0))
            stats['hypervolume_values'].append(losses.get('hypervolume_value', 0.0)) # ?
            stats['total_losses'].append(losses['total_loss'])
            stats['reward_mean'].append(losses['mean_reward'])
            stats['reward_std'].append(losses.get('std_reward', 0.0))
            
            # 打印
            if i % 10 == 0:
                print(f"\nBatch {i // args.batch_size}: "
                      f"Reward: {losses['mean_reward']:.4f}, "
                      f"Balance: {losses['balance_loss']:.4f}, "
                      f"Policy: {losses.get('policy_loss', 0.0):.4f}, "
                      f"Entropy: {losses.get('entropy_loss', 0.0):.4f}, "
                      f"HV Loss: {losses.get('hypervolume_loss', 0.0):.4f}, "
                      f"HV Value: {losses.get('hypervolume_value', 0.0):.4f}, "
                      f"Total: {losses['total_loss']:.4f}")
            
            pbar.update(1)
            
            # Regular checkpoint
            if i > 0 and i % 100 == 0:
                save_path = os.path.join(
                    args.save_directory,
                    args.wandb_name,
                    f'batch_{i}'
                )
                os.makedirs(save_path, exist_ok=True)
                
                # Save only gating weights
                save_moe_gating_weights(moe_model, save_path)
                
                # Save tokenizer
                tokenizer.save_pretrained(save_path)
                
                # Save training stats
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'baseline': trainer.reward_baseline,
                    'stats': stats
                }, os.path.join(save_path, 'training_state.pt'))
                
                print(f"\nCheckpoint saved to {save_path}")
                
                # Save plots
                save_plot_path = os.path.join(save_path, 'scores.png')
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Reward plot
                axes[0, 0].plot(stats['reward_mean'])
                axes[0, 0].fill_between(
                    np.arange(len(stats['reward_mean'])), 
                    np.array(stats['reward_mean']) - np.array(stats['reward_std']), 
                    np.array(stats['reward_mean']) + np.array(stats['reward_std']), 
                    alpha=0.5
                )
                axes[0, 0].set_title('Reward Mean ± Std')
                axes[0, 0].set_xlabel('Batch')
                axes[0, 0].set_ylabel('Reward')
                
                # Hypervolume plot
                axes[0, 1].plot(stats['hypervolume_values'])
                axes[0, 1].set_title('Hypervolume Value')
                axes[0, 1].set_xlabel('Batch')
                axes[0, 1].set_ylabel('HV')
                
                # Loss components plot
                axes[1, 0].plot(stats['policy_losses'], label='Policy Loss')
                axes[1, 0].plot(stats['balance_losses'], label='Balance Loss')
                axes[1, 0].plot(stats['entropy_losses'], label='Entropy Loss')
                axes[1, 0].plot(stats['hypervolume_losses'], label='HV Loss')
                axes[1, 0].set_title('Loss Components')
                axes[1, 0].set_xlabel('Batch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                
                # Total loss plot
                axes[1, 1].plot(stats['total_losses'])
                axes[1, 1].set_title('Total Loss')
                axes[1, 1].set_xlabel('Batch')
                axes[1, 1].set_ylabel('Loss')
                
                plt.tight_layout()
                plt.savefig(save_plot_path)
                plt.close()
        
        pbar.close()
        
        # Epoch end save
        save_path = os.path.join(
            args.save_directory,
            args.wandb_name,
            f'epoch_{epoch}_final'
        )
        os.makedirs(save_path, exist_ok=True)
        save_moe_gating_weights(moe_model, save_path)
        tokenizer.save_pretrained(save_path)
        
        # 保存统计数据
        df = pd.DataFrame(stats)
        df.to_csv(os.path.join(args.save_directory, args.wandb_name, 'data.csv'))
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Final model saved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()