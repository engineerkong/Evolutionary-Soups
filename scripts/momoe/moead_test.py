"""test_moe_sanity.py — Compare reward scores of MoE (fixed gating) vs standalone experts.

Evaluates four configurations and prints mean rewards for each:
  expert[0] standalone
  expert[1] standalone
  MoE with gating [1, 0]   → should match expert[0]
  MoE with gating [0, 1]   → should match expert[1]

Usage:
    python test_moe_sanity.py \
        --expert_model_paths ./models/ppo/assistant_ppo_harmless_2701/batch_832/ \
                             ./models/ppo/assistant_ppo_helpful_2701/batch_832/ \
        --sft_model_name ./models/sft/assistant_sft/model/ \
        --reward_names harmless,helpful \
        --gpu_id 0
"""

import sys
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, HfArgumentParser, DataCollatorWithPadding
from torch.utils.data import DataLoader

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import Instructions, get_clean_data, load_main_tokenizer
from moead_architecture import GatingNetwork, MoEForCausalLM
from moead_utils import REWARD_PATHS


@dataclass
class Args:
    sft_model_name:     str       = './models/sft/assistant_sft/model/'
    expert_model_paths: List[str] = field(default_factory=list)
    reward_names:       str       = 'harmless,helpful'
    batch_size:         int       = 64
    max_new_tokens:     int       = 128
    do_sample:          bool      = False
    num_continuations:  int       = 1
    gpu_id:             int       = 0
    seed:               int       = 42


class FixedGating(GatingNetwork):
    """Always returns a fixed coefficient vector regardless of input."""
    def __init__(self, coeffs: List[float]):
        super().__init__(lm_hidden_size=1, num_experts=len(coeffs))
        self._c = torch.tensor(coeffs, dtype=torch.float32)

    def forward(self, hidden_states):
        return self._c.to(hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)


def generate_and_score(model, input_ids, attention_mask, sft_tokenizer,
                       reward_models, instructions, generation_kwargs,
                       gpu_id, num_continuations):
    """Generate and return mean reward vector. Works for any model with .generate()."""
    device = f'cuda:{gpu_id}'
    accumulated = None

    for _ in range(num_continuations):
        outputs = model.generate(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            **generation_kwargs,
        )
        responses       = sft_tokenizer.batch_decode(outputs.cpu())
        prompts_decoded = sft_tokenizer.batch_decode(input_ids.cpu())
        del outputs

        prompts_clean, responses_clean = get_clean_data(responses, prompts_decoded)
        pairs = [(instructions.get_input(r), instructions.get_response(r))
                 for r in responses_clean]
        if hasattr(instructions, 'get_post'):
            scores = reward_models.get_reward_model_scores(
                pairs, instructions.get_post, normalize_rewards=False, round_digits=None)
        else:
            scores = reward_models.get_reward_model_scores(
                pairs, normalize_rewards=False, round_digits=None)

        n_prompts, n_rewards = len(prompts_clean), len(scores)
        if accumulated is None:
            accumulated = [[[] for _ in range(n_rewards)] for _ in range(n_prompts)]
        for p in range(n_prompts):
            for k in range(n_rewards):
                accumulated[p][k].append(scores[k][p])
        torch.cuda.empty_cache()

    per_prompt = np.array([[np.mean(accumulated[p][k]) for k in range(n_rewards)]
                           for p in range(n_prompts)])
    return per_prompt.mean(axis=0)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

args: Args     = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device         = f'cuda:{args.gpu_id}'
reward_names   = [x.strip() for x in args.reward_names.split(',')]
instructions   = Instructions()

tokenizer              = load_main_tokenizer(args.sft_model_name)
tokenizer.padding_side = 'left'

reward_model_paths = [REWARD_PATHS[n] for n in reward_names]
reward_models      = RewardModels(reward_model_paths, reward_model_paths, args.gpu_id)

generation_kwargs = dict(
    max_new_tokens=args.max_new_tokens,
    do_sample=args.do_sample,
    top_k=0, top_p=0.9, temperature=1.0,
)

# Build test dataset — same logic as build_dataset_eval_ppo (test split, stride 4)
print('Loading Anthropic/hh-rlhf test split ...')
ds = load_dataset('Anthropic/hh-rlhf', split='test[:128]')

def tokenize(sample):
    split_text          = sample['chosen'].split('\n\nAssistant:')
    sample['prompt']    = '\n\nAssistant:'.join(split_text[:-1]) + ' \n\nAssistant:'
    sample['input_ids'] = tokenizer.encode(sample['prompt'])
    return sample

ds = ds.map(tokenize, batched=False, num_proc=20)
ds = ds.filter(lambda x: 8 <= len(x['input_ids']) <= 512)
ds = ds.remove_columns([c for c in ds.column_names if c != 'input_ids'])
ds.set_format(type='torch')
print(f'  {len(ds)} prompts after filtering')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
loader        = DataLoader(ds, batch_size=args.batch_size,
                           collate_fn=data_collator, drop_last=False)

# Load experts
print(f'\nLoading {len(args.expert_model_paths)} experts ...')
experts = []
for i, path in enumerate(args.expert_model_paths):
    m = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device)
    m.resize_token_embeddings(len(tokenizer))
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    experts.append(m)
    print(f'  [{i}] {path}')

# ---------------------------------------------------------------------------
# Evaluate all configurations
# ---------------------------------------------------------------------------

configs = (
    [(f'expert[{i}] standalone', experts[i]) for i in range(len(experts))]
    + [(f'MoE gating={[1.0 if j==i else 0.0 for j in range(len(experts))]}',
        MoEForCausalLM(experts, FixedGating([1.0 if j==i else 0.0
                                             for j in range(len(experts))])).to(device))
       for i in range(len(experts))]
)

print(f'\nEvaluating {len(configs)} configurations over {len(ds)} prompts '
      f'({len(loader)} batches) ...\n')
print(f'{"Config":<40}  ' + '  '.join(f'{n:>10}' for n in reward_names))
print('-' * (40 + 14 * len(reward_names)))

for label, model in configs:
    if hasattr(model, 'eval'):
        model.eval()
    batch_rewards = []
    for batch in loader:
        r = generate_and_score(
            model, batch['input_ids'], batch['attention_mask'],
            tokenizer, reward_models, instructions,
            generation_kwargs, args.gpu_id, args.num_continuations)
        batch_rewards.append(r)
    mean_r = np.mean(batch_rewards, axis=0)
    print(f'{label:<40}  ' + '  '.join(f'{v:>10.4f}' for v in mean_r))
