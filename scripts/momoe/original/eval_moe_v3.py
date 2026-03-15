import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

from moe_utils_v3 import compute_hypervolume, convert_to_moe_model, load_moe_gating_weights, resolve_gating_checkpoint_path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions,
    Instructions_summary,
    build_dataset_eval_ppo,
    build_dataset_summary_eval_ppo,
    get_clean_data,
    load_main_tokenizer,
    sample_preferences_uniform,
)

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
    checkpoint_path: str = ""
    manual_lora_weights: str = field(
        default="1.0,0.0",
        metadata={"help": "fallback manual expert weights when checkpoint_path is missing, e.g. '1.0,0.0'"},
    )
    num_pref_samples: int = 10
    num_eval_samples: int = 0
    gating_type: str = field(default="linear", metadata={"help": "one of: linear, qk_attention, film"})
    reward_names: str = "harmless,helpful"
    exp_type: str = "assistant"
    save_directory: str = "./results/momoe/"
    wandb_name: str = "assistant_momoe_v3"


def build_reward_models(reward_names, gpu_id):
    paths = [REWARD_PATHS[name] for name in reward_names]
    return RewardModels(paths, paths, gpu_id), AutoTokenizer.from_pretrained(paths[0])


def build_eval_dataset(script_args, tokenizer, rm_tokenizers):
    if script_args.exp_type == "assistant":
        return build_dataset_eval_ppo("Anthropic/hh-rlhf", tokenizer, rm_tokenizers, split="test"), Instructions()
    return build_dataset_summary_eval_ppo("openai/summarize_from_feedback", tokenizer, rm_tokenizers, split="test"), Instructions_summary()


def shard_dataset(dataset, accelerator):
    if accelerator.num_processes == 1:
        return dataset, 0
    size = len(dataset)
    rank = accelerator.process_index
    world = accelerator.num_processes
    base = size // world
    remainder = size % world
    shard_start = rank * base + min(rank, remainder)
    return dataset.shard(num_shards=world, index=rank, contiguous=True), shard_start


def score_outputs(reward_models, instructions, responses):
    pairs = [(instructions.get_input(text), instructions.get_response(text)) for text in responses]
    if hasattr(instructions, "get_post"):
        return reward_models.get_reward_model_scores(pairs, instructions.get_post, normalize_rewards=False)
    return reward_models.get_reward_model_scores(pairs, normalize_rewards=False)


def parse_manual_lora_weights(spec):
    weights = [float(value.strip()) for value in spec.split(",") if value.strip()]
    if not weights:
        raise ValueError("manual_lora_weights must contain at least one value")
    total = sum(weights)
    if total <= 0:
        raise ValueError("manual_lora_weights must sum to a positive value")
    return [value / total for value in weights]


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
gpu_id = accelerator.local_process_index
reward_names = [name.strip() for name in script_args.reward_names.split(",")]
reward_models, _ = build_reward_models(reward_names, gpu_id)

model = convert_to_moe_model(
    base_model_name=script_args.base_model_name,
    lora_expert_paths=script_args.lora_expert_paths,
    num_rewards=len(reward_names),
    gating_type=script_args.gating_type,
    target_device=f"cuda:{gpu_id}",
)
resolved_checkpoint = resolve_gating_checkpoint_path(script_args.checkpoint_path)
weights_loaded = bool(script_args.checkpoint_path) and load_moe_gating_weights(model, resolved_checkpoint)
if not weights_loaded:
    manual_weights = parse_manual_lora_weights(script_args.manual_lora_weights)
    if len(manual_weights) != len(script_args.lora_expert_paths):
        raise ValueError(
            f"manual_lora_weights size mismatch: got {len(manual_weights)}, expected {len(script_args.lora_expert_paths)}"
        )
    model.set_manual_lora_weights(manual_weights)
    print(f"No valid checkpoint found at '{script_args.checkpoint_path}'. Using manual_lora_weights={manual_weights}")

tokenizer = load_main_tokenizer(script_args.base_model_name)
dataset, instructions = build_eval_dataset(script_args, tokenizer, reward_models.rm_tokenizers)
if 0 < script_args.num_eval_samples < len(dataset):
    dataset = dataset.select(range(script_args.num_eval_samples))
dataset, shard_start = shard_dataset(dataset, accelerator)
for key in ["key", "text", "prompt", "response", "query"]:
    if key in dataset.column_names:
        dataset = dataset.remove_columns(key)

dataloader = DataLoader(dataset, batch_size=8, drop_last=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer))
preferences = sample_preferences_uniform(len(reward_names), script_args.num_pref_samples)

model.eval()
tokenizer.padding_side = "left"
results = []
per_input_rewards = []
progress = tqdm(total=len(dataset), desc="Evaluating")

with torch.no_grad():
    input_offset = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        batch_size = input_ids.shape[0]
        for pref_idx, preference in enumerate(preferences):
            model.set_preference(preference)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128 if script_args.exp_type == "assistant" else 48,
                do_sample=False,
                top_k=0.0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            prompts, decoded = get_clean_data(decoded, prompts, remove_bad=False)
            rewards = score_outputs(reward_models, instructions, decoded)

            lora_weights = model.shared_gate._last_routing_weights.detach().cpu()
            for idx in range(batch_size):
                reward_vector = [rewards[k][idx] for k in range(len(reward_names))]
                scalarized = sum(preference[k] * reward_vector[k] for k in range(len(reward_names)))
                if pref_idx == 0:
                    per_input_rewards.append([])
                per_input_rewards[input_offset + idx].append(reward_vector)
                row = {
                    "input_idx": shard_start + input_offset + idx,
                    "pref_idx": pref_idx,
                    "prompt": prompts[idx],
                    "response": decoded[idx].replace(prompts[idx], "").strip(),
                    "scalarized_reward": scalarized,
                    "rank": accelerator.process_index,
                    "lora_weights": lora_weights[idx].tolist(),
                }
                for k, name in enumerate(reward_names):
                    row[f"pref_{name}"] = preference[k]
                    row[f"reward_{name}"] = reward_vector[k]
                results.append(row)
        input_offset += batch_size
        progress.update(batch_size)
progress.close()

hv_values = [compute_hypervolume(reward_vectors) for reward_vectors in per_input_rewards]
pd.DataFrame(results).to_csv(os.path.join(output_dir, f"eval_results_rank{accelerator.process_index}.csv"), index=False)
pd.DataFrame({"rank": accelerator.process_index, "hypervolume": hv_values}).to_csv(
    os.path.join(output_dir, f"per_input_hv_rank{accelerator.process_index}.csv"), index=False
)

if not accelerator.is_main_process:
    sys.exit(0)

if accelerator.num_processes > 1:
    start = time.time()
    result_paths = [os.path.join(output_dir, f"eval_results_rank{rank}.csv") for rank in range(accelerator.num_processes)]
    hv_paths = [os.path.join(output_dir, f"per_input_hv_rank{rank}.csv") for rank in range(accelerator.num_processes)]
    while not (all(os.path.exists(path) for path in result_paths) and all(os.path.exists(path) for path in hv_paths)):
        if time.time() - start > 1800:
            raise TimeoutError("Timed out waiting for evaluation shards")
        time.sleep(2)
    results_df = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    hv_df = pd.concat([pd.read_csv(path) for path in hv_paths], ignore_index=True)
else:
    results_df = pd.DataFrame(results)
    hv_df = pd.DataFrame({"hypervolume": hv_values})

summary_rows = [{
    "type": "overall",
    "mean_scalarized_reward": results_df["scalarized_reward"].mean(),
    "std_scalarized_reward": results_df["scalarized_reward"].std(),
    "hypervolume": compute_hypervolume([[row[f"reward_{name}"] for name in reward_names] for row in results_df.to_dict("records")]),
    "mean_input_hv": hv_df["hypervolume"].mean(),
}]
for pref_idx, preference in enumerate(preferences):
    pref_df = results_df[results_df["pref_idx"] == pref_idx]
    row = {
        "type": "preference",
        "pref_idx": pref_idx,
        "mean_scalarized_reward": pref_df["scalarized_reward"].mean(),
        "std_scalarized_reward": pref_df["scalarized_reward"].std(),
    }
    for k, name in enumerate(reward_names):
        row[f"pref_{name}"] = preference[k]
        row[f"mean_reward_{name}"] = pref_df[f"reward_{name}"].mean()
    summary_rows.append(row)

results_df.to_csv(os.path.join(output_dir, "eval_results.csv"), index=False)
pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "eval_summary.csv"), index=False)
