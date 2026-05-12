"""train_hoe_router.py — Stage 1: Supervised router pre-training for HoE.

Trains only the MoLA router parameters with NLLLoss so each router learns
to route tokens to the appropriate expert based on content type.

Run this BEFORE train_hoe.py (Stage 2).  The saved router_weights.pt is
loaded in Stage 2 via --pretrained_moe_path, and a deepcopy of the loaded
model is used as the KL reference — exactly reproducing the original HoE
(Mymorlhf_ref.py) design.

Labeling strategy per task
--------------------------
Anthropic/hh-rlhf              : harmless split → idx(harmless expert)
                                  helpful split  → idx(helpful expert)
PKU-Alignment/PKU-SafeRLHF-10K : safe responses   → idx(beaver_reward)
                                  unsafe responses → idx(beaver_cost)
openai/summarize_from_feedback  : round-robin by example index (0,1,2,…)

In all cases the preference vector λ fed to the router equals one_hot(expert_idx)
so that the router simultaneously learns content-based routing AND λ-passthrough.
"""

import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import HfArgumentParser

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(script_dir))

HoE_ROOT = project_root.parent / 'anonymous-repo-for-HoE' / 'Code' / 'HoE'
for _p in [str(HoE_ROOT / 'src'), str(HoE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hoe_utils import build_hoe_model, make_number_experts_str, parse_comma_int_list, parse_comma_str_list, save_router_weights
from scripts.utils.utils import load_main_tokenizer, Instructions_summary


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    base_model_name:    str = 'meta-llama/Llama-2-7b-hf'
    expert_model_paths: str = ''           # comma-separated LoRA adapter paths
    dataset_name:       str = 'Anthropic/hh-rlhf'
    reward_names:       str = ''           # auto-selected from dataset_name if empty
    save_directory:     str = './models/hoe_router/'
    run_name:           str = 'hoe_router'
    max_seq_length:     int = 512
    learning_rate:      float = 1e-5
    num_train_epochs:   float = 3.0
    per_device_batch:   int = 2
    grad_accum_steps:   int = 2
    seed:               int = 8888

_DEFAULT_REWARD_NAMES = {
    'Anthropic/hh-rlhf':                  'harmless,helpful',
    'openai/summarize_from_feedback':      'summary,faithful,deberta',
    'PKU-Alignment/PKU-SafeRLHF-10K':     'beaver_reward,beaver_cost',
}

_RESPONSE_SPLIT = {
    'Anthropic/hh-rlhf':                  '\n\nAssistant:',
    'openai/summarize_from_feedback':      '### Response:',
    'PKU-Alignment/PKU-SafeRLHF-10K':     '\n\nAssistant:',
}

# Match max_new_tokens used in collect_rewards.py / eval scripts
_MAX_NEW_TOKENS = {
    'openai/summarize_from_feedback': 48,
}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _label_tokens(text: str, expert_idx: int, tokenizer, max_seq_length: int,
                  response_split: str, max_new_tokens: int = 128) -> dict | None:
    """Tokenize full text and assign router_labels: expert_idx for response
    tokens, -100 for prompt tokens.  Returns None if text is too short.

    Response tokens are capped at max_new_tokens to match the generation
    length used at inference time (48 for summary, 128 for others).
    """
    full_ids = tokenizer.encode(text, max_length=max_seq_length, truncation=True)
    if len(full_ids) < 8:
        return None

    # Find the end of the prompt (last occurrence of response_split)
    prompt_text = text[:text.rfind(response_split) + len(response_split)]
    prompt_len  = min(len(tokenizer.encode(prompt_text)), len(full_ids))

    # Cap sequence at prompt + max_new_tokens to match inference-time distribution
    total_len = min(len(full_ids), prompt_len + max_new_tokens)
    full_ids  = full_ids[:total_len]

    labels = [-100] * len(full_ids)
    for j in range(prompt_len, len(full_ids)):
        labels[j] = expert_idx

    return {
        'input_ids':      full_ids,
        'attention_mask': [1] * len(full_ids),
        'router_labels':  labels,
    }


def build_router_dataset(args: ScriptArguments, tokenizer) -> Dataset:
    reward_names   = [r.strip() for r in args.reward_names.split(',')]
    n_experts      = len(reward_names)
    resp_split     = _RESPONSE_SPLIT.get(args.dataset_name, '\n\nAssistant:')
    max_new_tokens = _MAX_NEW_TOKENS.get(args.dataset_name, 128)
    samples        = []

    # ---- Anthropic/hh-rlhf ------------------------------------------------
    if args.dataset_name == 'Anthropic/hh-rlhf':
        harmless_idx = reward_names.index('harmless') if 'harmless' in reward_names else 0
        helpful_idx  = reward_names.index('helpful')  if 'helpful'  in reward_names else 1

        for split_name, expert_idx in [('harmless-base', harmless_idx),
                                        ('helpful-base',  helpful_idx)]:
            ds = load_dataset(args.dataset_name, data_dir=split_name, split='train')
            for item in ds:
                for text in [item['chosen'], item['rejected']]:
                    rec = _label_tokens(text, expert_idx, tokenizer,
                                        args.max_seq_length, resp_split, max_new_tokens)
                    if rec:
                        samples.append(rec)

    # ---- PKU-SafeRLHF-10K -------------------------------------------------
    elif args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
        reward_idx = reward_names.index('beaver_reward') if 'beaver_reward' in reward_names else 0
        cost_idx   = reward_names.index('beaver_cost')   if 'beaver_cost'   in reward_names else 1

        ds = load_dataset(args.dataset_name, split='train')
        for item in ds:
            for resp_key, safe_key in [('response_0', 'is_response_0_safe'),
                                        ('response_1', 'is_response_1_safe')]:
                text = '\n\nHuman:' + item['prompt'] + ' \n\nAssistant:' + item[resp_key]
                expert_idx = reward_idx if item[safe_key] else cost_idx
                rec = _label_tokens(text, expert_idx, tokenizer,
                                    args.max_seq_length, resp_split, max_new_tokens)
                if rec:
                    samples.append(rec)

    # ---- openai/summarize_from_feedback (round-robin) ----------------------
    elif args.dataset_name == 'openai/summarize_from_feedback':
        ds = load_dataset(args.dataset_name, 'comparisons', split='train')
        ds = ds.filter(lambda x: x['info']['post'] is not None
                                  and 100 < len(x['info']['post']) < 1200)
        for i, item in enumerate(ds):
            expert_idx = i % n_experts
            post       = item['info']['post'].replace('\n', ' ')
            prompt     = Instructions_summary.prompt_input(post)
            # Use both summaries for this post, both labeled with same expert_idx
            for summary_item in item['summaries']:
                text = prompt + summary_item['text']
                rec  = _label_tokens(text, expert_idx, tokenizer,
                                     args.max_seq_length, resp_split)
                if rec:
                    samples.append(rec)
    else:
        raise ValueError(f'Unsupported dataset_name: {args.dataset_name!r}')

    print(f'[router] Built {len(samples)} training samples for {n_experts} experts.')
    return Dataset.from_list(samples)


# ---------------------------------------------------------------------------
# Collator: pad to same length within batch
# ---------------------------------------------------------------------------

def collate_fn(batch, pad_id: int):
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids, attn_masks, router_labels = [], [], []
    for x in batch:
        pad = max_len - len(x['input_ids'])
        input_ids.append(x['input_ids']      + [pad_id] * pad)
        attn_masks.append(x['attention_mask'] + [0]      * pad)
        router_labels.append(x['router_labels'] + [-100] * pad)
    return {
        'input_ids':      torch.tensor(input_ids,     dtype=torch.long),
        'attention_mask': torch.tensor(attn_masks,    dtype=torch.long),
        'router_labels':  torch.tensor(router_labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Custom Trainer with NLLLoss on router logits
# ---------------------------------------------------------------------------

def _generate_random_weights(n_samples: int, n_experts: int) -> torch.Tensor:
    """Sample random preference vectors from the probability simplex.
    Matches train4or.py's generate_weights() for 2 experts, generalised to N.
    Using random λ (not label-aligned) forces the router to learn content-based
    routing purely from hidden states, independent of λ.
    """
    lam = np.random.dirichlet(np.ones(n_experts), size=n_samples)
    return torch.tensor(lam, dtype=torch.bfloat16)


class RouterTrainer:
    """Minimal training loop that reproduces train4or.py's compute_loss exactly.

    Key alignment with train4or.py:
      - dynamic_weights set with batch_size = bs * seqlen (per token, not per sample)
      - λ is random uniform from simplex (NOT one-hot), so router must learn
        content routing from hidden states independently of λ
      - loss = NLLLoss(router_logits.log(), labels)  — router outputs are probabilities
      - forward via model.model.model(**inputs) matching the commented-out path
        model.base_model.model.model(**inputs) in train4or.py
    """

    def __init__(self, model, dataset, args: ScriptArguments, n_experts: int,
                 lora_target_modules: list, num_layers: int, pad_id: int):
        self.model               = model
        self.dataset             = dataset
        self.args                = args
        self.n_experts           = n_experts
        self.lora_target_modules = lora_target_modules
        self.num_layers          = num_layers
        self.num_modules         = len(lora_target_modules)
        self.pad_id              = pad_id
        self.loss_fn             = nn.NLLLoss(ignore_index=-100)

    def compute_loss(self, model, batch):
        """Mirrors train4or.py MyTrainer.compute_loss exactly."""
        router_labels = batch.pop('router_labels')          # (bs, seq)
        bs, seqlen    = router_labels.shape

        # Random λ per token — same as train4or.py generate_weights(bs*seqlen0)
        lam = _generate_random_weights(bs * seqlen, self.n_experts)
        model.dynamic_weights.set_dynamic_weights(lam, batch_size=bs * seqlen)

        # Forward through LlamaModel_d — mirrors train4or.py line:
        # output = accelerator.unwrap_model(model).model.model(**inputs)
        # model.model  → LlamaForCausalLM_d (via PeftModel.__getattr__ → base_model.model)
        # model.model.model → LlamaModel_d inside LlamaForCausalLM_d
        output = model.model.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )

        # output[1]: tuple of per-layer router logit tensors
        # each: (modules*bs*seq, n_experts) — router outputs are already probabilities
        router_logits = torch.stack(output[1])          # (layers, modules*bs*seq, n_experts)

        layers, modules_bs_seq, n_exp = router_logits.shape
        assert layers == self.num_layers
        modules = modules_bs_seq // (bs * seqlen)
        assert modules == self.num_modules

        # Repeat labels for all layers and modules — mirrors train4or.py exactly
        labels = router_labels.view(-1)                  # (bs*seq,)
        target = labels.repeat(layers * modules)         # (layers*modules*bs*seq,)
        router_logits = router_logits.view(-1, n_exp)    # (layers*modules*bs*seq, n_experts)

        # NLLLoss on log-probabilities — matches train4or.py: loss_fn(router_logits.log(), target)
        return self.loss_fn(router_logits.log(), target)

    def train(self, accelerator: Accelerator):
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.per_device_batch,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.pad_id),
        )

        # Only router parameters trainable — matches train4or.py:
        # for name, param in model.named_parameters():
        #     if "router" in name: param.requires_grad = True
        self.model.train()
        for name, param in self.model.named_parameters():
            param.requires_grad = 'router' in name

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate,
        )


        self.model, optimizer, dataloader = accelerator.prepare(
            self.model, optimizer, dataloader)
        n_epochs = int(self.args.num_train_epochs)
        for epoch in range(n_epochs):
            total_loss, n_batches = 0.0, 0
            pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not accelerator.is_main_process)
            for step, batch in enumerate(pbar):
                unwrapped = accelerator.unwrap_model(self.model)
                loss = self.compute_loss(unwrapped, batch)
                accelerator.backward(loss)

                if (step + 1) % self.args.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                n_batches  += 1

                if step % 100 == 0 and accelerator.is_main_process:
                    print(f'Epoch {epoch} | step {step} | loss {loss.item():.4f}')

            if accelerator.is_main_process:
                avg = total_loss / max(n_batches, 1)
                print(f'Epoch {epoch} done | avg loss {avg:.4f}')

        # Flush any remaining accumulated gradients
        optimizer.step()
        optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = HfArgumentParser(ScriptArguments)
    args   = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()

    # Auto-fill reward_names
    if not args.reward_names.strip():
        args.reward_names = _DEFAULT_REWARD_NAMES.get(args.dataset_name, '')
        if not args.reward_names:
            raise ValueError(f'Cannot infer reward_names for {args.dataset_name}')

    reward_names = [r.strip() for r in args.reward_names.split(',')]
    n_experts    = len(reward_names)

    # ---- Load tokenizer ----
    expert_paths = parse_comma_str_list(args.expert_model_paths)
    tokenizer    = load_main_tokenizer(expert_paths[0])
    pad_id       = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ---- Build MoLA model ----
    experts_str     = make_number_experts_str(n_experts, 32)
    number_experts  = parse_comma_int_list(experts_str)
    top_k           = number_experts[:]

    # Read lora_target_modules for shape calculation
    with open(Path(expert_paths[0]) / 'adapter_config.json') as f:
        lora_target_modules = json.load(f)['target_modules']

    hoe_model = build_hoe_model(
        base_model_name=args.base_model_name,
        expert_model_paths=expert_paths,
        number_experts=number_experts,
        top_k=top_k,
        router_type='v1',
        num_rewards=n_experts,
    )

    # Cast entire model to bfloat16 so lora_A/lora_B match the base model dtype
    hoe_model = hoe_model.to(torch.bfloat16)

    # Enable router logit collection — matches train4or.py:
    # model.base_model.model.model.config.output_router_logits = True
    # (model.model.model is equivalent via PeftModel.__getattr__ → base_model.model)
    hoe_model.base_model.model.model.config.output_router_logits = True

    # ---- Build dataset ----
    dataset = build_router_dataset(args, tokenizer)

    # ---- Train ----
    trainer = RouterTrainer(
        model=hoe_model,
        dataset=dataset,
        args=args,
        n_experts=n_experts,
        lora_target_modules=lora_target_modules,
        num_layers=32,
        pad_id=pad_id,
    )
    trainer.train(accelerator)

    # ---- Save router weights ----
    if accelerator.is_main_process:
        save_path = os.path.join(args.save_directory, args.run_name)
        os.makedirs(save_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(hoe_model)
        save_router_weights(unwrapped, save_path)
        print(f'Router weights saved to {save_path}/router_weights.pt')


if __name__ == '__main__':
    main()
