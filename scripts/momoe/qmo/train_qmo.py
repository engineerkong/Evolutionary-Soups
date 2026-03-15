import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import set_seed

from qmo_architecture import MoEGatingTrainer
from qmo_utils import load_base_model, load_expert_state_dicts, save_moe_qtable

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

REWARD_PATHS = {
    "harmless": "Ray2333/gpt2-large-harmless-reward_model",
    "helpful":  "Ray2333/gpt2-large-helpful-reward_model",
    "deberta":  "OpenAssistant/reward-model-deberta-v3-large-v2",
    "summary":  "Tristan/gpt2_reward_summarization",
    "faithful": "CogComp/bart-faithful-summary-detector",
    "humor":    "mohameddhiab/humor-no-humor",
}


@dataclass
class ScriptArguments:
    sft_model_name: str = "./models/sft/model/"
    expert_model_paths: List[str] = field(default_factory=list)
    num_pref_samples: int = 10
    log_with: str = field(default="none", metadata={"help": "use 'wandb' to log with W&B"})
    reward_names: str = "harmless,helpful"
    exp_type: str = "assistant"
    save_directory: str = "./models/qmo/"
    wandb_name: str = "qmo_assistant"


def build_reward_models(reward_names, gpu_id):
    from transformers import AutoTokenizer
    paths = [REWARD_PATHS[name] for name in reward_names]
    return RewardModels(paths, paths, gpu_id), AutoTokenizer.from_pretrained(paths[0])


def build_training_dataset(script_args, tokenizer, rm_tokenizer):
    if script_args.exp_type == "assistant":
        return (
            build_dataset_ppo("Anthropic/hh-rlhf", tokenizer, rm_tokenizer, split="train", size=64), # hardcoded small size for quick testing; remove size arg for full dataset
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
    dataset = dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index, contiguous=True)
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
        wandb.login(key="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")
    except ImportError as exc:
        raise ImportError("log_with='wandb' requires the 'wandb' package to be installed.") from exc
    wandb_run = wandb.init(
        project="momoe",
        name=script_args.wandb_name,
        config={
            "sft_model_name": script_args.sft_model_name,
            "reward_names": reward_names,
            "num_pref_samples": script_args.num_pref_samples,
            "batch_size": cfg["batch_size"],
        },
    )

# Load one expert model as the mutable base; its weights are overwritten each step
model = load_base_model(script_args.expert_model_paths[0], target_device=f"cuda:{gpu_id}")
# Load all expert full state dicts on CPU in float32
expert_state_dicts = load_expert_state_dicts(script_args.expert_model_paths)
model = accelerator.prepare(model)
tokenizer = load_main_tokenizer(script_args.sft_model_name)
dataset, instructions = build_training_dataset(script_args, tokenizer, rm_tokenizer)
dataset = shard_dataset(dataset, accelerator)

trainer = MoEGatingTrainer(
    moe_model=model,
    expert_state_dicts=expert_state_dicts,
    reward_models=reward_models,
    instructions=instructions,
    num_rewards=len(reward_names),
    num_pref_samples=script_args.num_pref_samples,
    num_pref_bins=cfg.get("num_pref_bins", 11),
    num_action_bins=cfg.get("num_action_bins", 11),
    alpha=float(cfg.get("alpha", 0.1)),
    epsilon=float(cfg.get("epsilon", 0.3)),
)
print_trainable_parameters(model)

stats = []
global_step = 0
progress = tqdm(total=cfg["epochs"])
for epoch in range(cfg["epochs"]):
    # progress = tqdm(total=len(dataset) // cfg["batch_size"], desc=f"Epoch {epoch + 1}")
    shuffled = dataset.shuffle(seed=epoch)
    for start in range(0, len(shuffled), cfg["batch_size"]):
        losses = trainer.train_step_qlearning(shuffled[start:start + cfg["batch_size"]], tokenizer)
        stats.append(losses)
        global_step += 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/reward": losses["mean_reward"],
                    "train/epsilon": losses["epsilon"],
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                },
                step=global_step,
            )
        if global_step % 100 == 0:
            mean_expert_weights = ", ".join(f"{w:.4f}" for w in losses["mean_expert_weights"])
            print(
                f"step={global_step} reward={losses['mean_reward']:.4f} "
                f"epsilon={losses['epsilon']:.4f} "
                f"mean_expert_weights=[{mean_expert_weights}]"
            )
    if global_step % 100 == 0:
        save_path = os.path.join(output_dir, f"epoch_{epoch + 1}_step_{global_step}")
        if accelerator.is_main_process:
            save_moe_qtable(trainer.q_table, save_path)
            tokenizer.save_pretrained(save_path)
            pd.DataFrame(stats).to_csv(os.path.join(output_dir, "data.csv"), index=False)
    progress.update(1)
progress.close()

if wandb_run is not None:
    wandb_run.finish()