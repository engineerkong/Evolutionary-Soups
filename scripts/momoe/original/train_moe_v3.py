import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import set_seed

from moe_architecture_v3 import MoEGatingTrainer
from moe_utils_v3 import convert_to_moe_model, save_moe_gating_weights

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions,
    Instructions_summary,
    build_dataset_ppo,
    build_dataset_summary_ppo,
    load_config,
    load_main_tokenizer,
    print_trainable_parameters,
)

import wandb
wandb.login(key="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")

REWARD_PATHS = {
    "harmless": "Ray2333/gpt2-large-harmless-reward_model",
    "helpful": "Ray2333/gpt2-large-helpful-reward_model",
    "deberta": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "summary": "Tristan/gpt2_reward_summarization",
    "faithful": "CogComp/bart-faithful-summary-detector",
    "humor": "mohameddhiab/humor-no-humor",
}


@dataclass
class ScriptArguments:
    base_model_name: str = "./models/sft/model/"
    lora_expert_paths: List[str] = field(default_factory=list)
    num_pref_samples: int = 10
    grad_accumulation_steps: int = field(default=8, metadata={"help": "number of dataset batches to accumulate before optimizer step"})
    random_exploration_steps: int = field(default=10, metadata={"help": "number of initial train steps that use random routing weights for exploration"})
    ppo_clip_range: float = field(default=0.2, metadata={"help": "PPO clip range for gate policy updates"})
    gating_type: str = field(default="linear", metadata={"help": "one of: simplified, linear, qk_attention, film"})
    log_with: str = field(default="none", metadata={"help": "use 'wandb' to log with Weights & Biases"})
    reward_names: str = "harmless,helpful"
    exp_type: str = "assistant"
    save_directory: str = "./models/momoe/"
    wandb_name: str = "assistant_momoe_v3"


def build_reward_models(reward_names, gpu_id):
    paths = [REWARD_PATHS[name] for name in reward_names]
    return RewardModels(paths, paths, gpu_id), AutoTokenizer.from_pretrained(paths[0])


def build_training_dataset(script_args, tokenizer, rm_tokenizer):
    if script_args.exp_type == "assistant":
        return (
            build_dataset_ppo("Anthropic/hh-rlhf", tokenizer, rm_tokenizer, split="train"),
            Instructions(),
        )
    return (
        build_dataset_summary_ppo("openai/summarize_from_feedback", tokenizer, rm_tokenizer, split="train"),
        Instructions_summary(),
    )


def shard_dataset(dataset, accelerator):
    if accelerator.num_processes == 1:
        return dataset
    min_len = len(dataset) // accelerator.num_processes
    dataset = dataset.shard(
        num_shards=accelerator.num_processes,
        index=accelerator.process_index,
        contiguous=True,
    )
    return dataset.select(range(min_len)) if len(dataset) > min_len else dataset


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
cfg = load_config(script_dir / "config.yaml")
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
gpu_id = accelerator.local_process_index
reward_names = [name.strip() for name in script_args.reward_names.split(",")]
reward_models, rm_tokenizer = build_reward_models(reward_names, gpu_id)

wandb_run = None
if accelerator.is_main_process and script_args.log_with == "wandb":
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("log_with='wandb' requires the 'wandb' package to be installed.") from exc
    wandb_run = wandb.init(
        project="momoe",
        name=script_args.wandb_name,
        config={
            "base_model_name": script_args.base_model_name,
            "gating_type": script_args.gating_type,
            "reward_names": reward_names,
            "num_pref_samples": script_args.num_pref_samples,
            "learning_rate": float(cfg["learning_rate"]),
            "batch_size": cfg["batch_size"],
        },
    )

moe_model = convert_to_moe_model(
    base_model_name=script_args.base_model_name,
    lora_expert_paths=script_args.lora_expert_paths,
    num_rewards=len(reward_names),
    gating_type=script_args.gating_type,
    target_device=f"cuda:{gpu_id}",
)
moe_model = accelerator.prepare(moe_model)
tokenizer = load_main_tokenizer(script_args.base_model_name)
dataset, instructions = build_training_dataset(script_args, tokenizer, rm_tokenizer)
dataset = dataset.select(range(min(64, len(dataset))))
# dataset = shard_dataset(dataset, accelerator)

trainer = MoEGatingTrainer(
    moe_model=moe_model,
    reward_models=reward_models,
    instructions=instructions,
    learning_rate=float(cfg["learning_rate"]),
    num_rewards=len(reward_names),
    num_pref_samples=script_args.num_pref_samples,
    random_exploration_steps=script_args.random_exploration_steps,
    ppo_clip_range=script_args.ppo_clip_range,
)
print_trainable_parameters(moe_model)
trainer.optimizer.zero_grad()

stats = []
global_step = 0
for epoch in range(cfg["epochs"]):
    progress = tqdm(total=len(dataset) // cfg["batch_size"], desc=f"Epoch {epoch + 1}")
    shuffled = dataset.shuffle(seed=epoch)
    batch_starts = list(range(0, len(shuffled), cfg["batch_size"]))
    for batch_idx, start in enumerate(batch_starts):
        step_optimizer = (
            (batch_idx + 1) % max(1, script_args.grad_accumulation_steps) == 0
            or batch_idx == len(batch_starts) - 1
        )
        losses = trainer.train_step_reinforce(
            shuffled[start:start + cfg["batch_size"]],
            tokenizer,
            grad_scale=script_args.grad_accumulation_steps,
            step_optimizer=step_optimizer,
        )
        stats.append(losses)
        global_step += 1
        progress.update(1)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/reward": losses["mean_reward"],
                    "train/policy_loss": losses["policy_loss"],
                    "train/total_loss": losses["total_loss"],
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                },
                step=global_step,
            )
    progress.close()

    save_path = os.path.join(output_dir, f"epoch_{epoch + 1}_step_{global_step}")
    if accelerator.is_main_process:
        save_moe_gating_weights(moe_model, save_path)
        tokenizer.save_pretrained(save_path)
        pd.DataFrame(stats).to_csv(os.path.join(output_dir, "data.csv"), index=False)
        torch.save({"epoch": epoch, "global_step": global_step, "baseline": trainer.reward_baseline}, os.path.join(save_path, "training_state.pt"))

if wandb_run is not None:
    wandb_run.finish()
