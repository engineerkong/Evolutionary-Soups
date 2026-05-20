"""MOD (Mixture-of-Distributions) evaluation.

Mirrors eval_ppo_rs.py exactly in interface (same args, same dataset handling,
same preference sampling) but fuses logits at decoding time instead of
merging LoRA weights before generation.

MOD inference: all N per-objective PPO LoRA adapters are loaded into one base
model and run in parallel at each decoding step; their log-probs are combined
as  Σ λᵢ · log pᵢ(token)  (product-of-experts / reverse-KL fusion).
"""

import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

script_dir   = Path(__file__).resolve().parent        # baselines/mod/
project_root = script_dir.parent.parent.parent         # MOMoE/
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from trl import set_seed

from scripts.utils.multi_reward_models import RewardModels
from scripts.utils.utils import (
    Instructions, Instructions_summary,
    build_dataset_eval, build_dataset_summary_eval,
    build_dataset_news_summary_ppo,
    build_dataset_beaver_eval, build_dataset_steer_eval,
    build_dataset_ultrafeedback_eval,
    get_clean_data, load_main_tokenizer, save_configs,
    sample_preferences_uniform,
)

tqdm.pandas()


# ---------------------------------------------------------------------------
# Logit-fusion model
# ---------------------------------------------------------------------------

class LogitsFusionModel(torch.nn.Module):
    """Runs N LoRA adapters in parallel and fuses log-probs at each step.

    Adapters must already be registered in `peft_model` under the names
    'model_0', 'model_1', ... via PeftModel / load_adapter().
    """

    def __init__(self, peft_model, weights):
        super().__init__()
        self.model   = peft_model
        self.weights = list(weights)
        self.n       = len(weights)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None,
                 max_new_tokens=128, min_length=-1,
                 do_sample=True, temperature=0.7, top_p=0.9, **kwargs):
        batch_size = input_ids.shape[0]
        cfg = self.model.config

        eos_id = cfg.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        pad_id = cfg.pad_token_id if cfg.pad_token_id is not None else eos_id

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        unfinished      = input_ids.new_ones(batch_size)
        logp_accum      = None       # accumulated log-prob per adapter (batch,)
        past_key_values = None       # list[n] of per-adapter kv-cache
        cur_ids         = input_ids
        cur_mask        = attention_mask

        for _ in range(max_new_tokens):
            all_logits  = []
            all_past_kv = []
            for idx in range(self.n):
                self.model.set_adapter(f'model_{idx}')
                out = self.model(
                    input_ids      = cur_ids,
                    attention_mask = cur_mask,
                    past_key_values= past_key_values[idx] if past_key_values else None,
                    use_cache      = True,
                )
                all_logits.append(out.logits[:, -1, :])   # (batch, vocab)
                all_past_kv.append(out.past_key_values)
            past_key_values = all_past_kv

            # accumulate log-probs across steps (product-of-experts)
            if logp_accum is not None:
                all_logits = [all_logits[i] + logp_accum[i].unsqueeze(1)
                              for i in range(self.n)]

            fused = sum(self.weights[i] * all_logits[i] for i in range(self.n))

            if do_sample:
                if temperature != 1.0:
                    fused = fused / temperature
                # top-p (nucleus) sampling on fused logits
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(fused, dim=-1, descending=True)
                    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # remove tokens with cumulative prob above top_p (shift right to keep first token)
                    sorted_logits[cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p] = -float('inf')
                    fused = torch.zeros_like(fused).scatter_(-1, sorted_idx, sorted_logits)
                probs       = torch.softmax(fused, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = fused.argmax(dim=-1)

            logp_accum = [
                torch.gather(all_logits[i], 1, next_tokens.unsqueeze(1)).squeeze(1)
                for i in range(self.n)
            ]

            if eos_id is not None:
                next_tokens = next_tokens * unfinished + pad_id * (1 - unfinished)
                unfinished  = unfinished * (next_tokens != eos_id).long()

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            cur_ids   = next_tokens.unsqueeze(1)
            cur_mask  = torch.cat([cur_mask, cur_mask.new_ones(batch_size, 1)], dim=1)

            if unfinished.max() == 0:
                break

        return input_ids


# ---------------------------------------------------------------------------
# Script arguments  (identical to eval_ppo_rs.py)
# ---------------------------------------------------------------------------

_ARMORM = 'armorm://RLHFlow/ArmoRM-Llama3-8B-v0.1'
_URM    = 'urm://LxzGordon/URM-LLaMa-3.1-8B'

reward_path_tokenizer_dict = {
    'harmless':                 'Ray2333/gpt2-large-harmless-reward_model',
    'helpful':                  'Ray2333/gpt2-large-helpful-reward_model',
    'deberta':                  'OpenAssistant/reward-model-deberta-v3-large-v2',
    'summary':                  'Tristan/gpt2_reward_summarization',
    'faithful':                 'CogComp/bart-faithful-summary-detector',
    'humor':                    'mohameddhiab/humor-no-humor',
    'beaver_reward':            'PKU-Alignment/beaver-7b-v1.0-reward',
    'beaver_cost':              'PKU-Alignment/beaver-7b-v1.0-cost',
    'steer_helpfulness':        f'{_URM}#0',
    'steer_correctness':        f'{_URM}#1',
    'steer_coherence':          f'{_URM}#2',
    'steer_complexity':         f'{_URM}#3',
    'steer_verbosity':          f'{_URM}#4',
    'uf_instruction_following': f'{_ARMORM}#6',
    'uf_truthfulness':          f'{_ARMORM}#7',
    'uf_honesty':               f'{_ARMORM}#8',
    'uf_helpfulness':           f'{_ARMORM}#9',
}

_DEFAULT_REWARD_NAMES = {
    'Anthropic/hh-rlhf':               'harmless,helpful',
    'openai/summarize_from_feedback':   'summary,faithful',
    'argilla/news-summary':             'summary,faithful',
    'PKU-Alignment/PKU-SafeRLHF-10K':  'beaver_reward,beaver_cost',
    'nvidia/HelpSteer':                 'steer_helpfulness,steer_correctness',
    'nvidia/HelpSteer2':                'steer_helpfulness,steer_correctness',
    'openbmb/UltraFeedback':            'uf_instruction_following,uf_truthfulness,uf_honesty,uf_helpfulness',
}


@dataclass
class ScriptArguments:
    base_model_name:   str       = field(default='meta-llama/Llama-2-7b-hf')
    expert_model_paths: List[str] = field(default_factory=list,
                                          metadata={'help': 'PPO adapter paths (space-separated on CLI)'})
    num_pref_samples:  int       = field(default=10)
    reward_names:      str       = field(default='',
                                         metadata={'help': 'comma-separated; auto-selected from dataset_name if empty'})
    dataset_name:      str       = field(default='Anthropic/hh-rlhf')
    save_directory:    str       = field(default='./results/mod/')
    wandb_name:        str       = field(default='eval_mod')
    split:             str       = field(default='test')
    size:              int       = field(default=0, metadata={'help': 'prompts to use (0 = all)'})
    num_continuations: int       = field(default=3, metadata={'help': 'generations per prompt; rewards are averaged'})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ppo_model_list = script_args.expert_model_paths
print(script_args)

if not script_args.reward_names:
    script_args.reward_names = _DEFAULT_REWARD_NAMES.get(script_args.dataset_name, 'harmless,helpful')
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
n_objectives = len(reward_names)

assert len(ppo_model_list) == n_objectives, (
    f'Need one PPO model path per reward ({n_objectives}), got {len(ppo_model_list)}')

output_dir = os.path.join(script_args.save_directory, script_args.wandb_name)
os.makedirs(output_dir, exist_ok=True)

set_seed(8888)
accelerator = Accelerator()
process_id  = accelerator.local_process_index
gpu_id      = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# ── Reward models ─────────────────────────────────────────────────────────────
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict:
        raise NotImplementedError(f'Unknown reward name: {name!r}')
    reward_model_path_list.append(reward_path_tokenizer_dict[name])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name])

save_info = {'tokenizer_name': ppo_model_list[0]}
for i, p in enumerate(ppo_model_list):
    save_info[f'ppo_model_name{i+1}'] = p
save_configs(save_info, output_dir)
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

# ── Tokenizer + base model + adapters ────────────────────────────────────────
# tokenizer from first PPO model dir (same as eval_ppo_rs.py)
tokenizer = load_main_tokenizer(ppo_model_list[0])
tokenizer.padding_side = 'left'

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name, torch_dtype=torch.bfloat16, device_map=gpu_id)
base_model.resize_token_embeddings(len(tokenizer))

# Load first adapter via PeftModel, rest via load_adapter
peft_model = PeftModel.from_pretrained(
    base_model, ppo_model_list[0], adapter_name='model_0')
for idx in range(1, n_objectives):
    peft_model.load_adapter(ppo_model_list[idx], adapter_name=f'model_{idx}')
    print(f'  Loaded adapter model_{idx} from {ppo_model_list[idx]}')
peft_model.eval()
for p in peft_model.parameters():
    p.requires_grad = False

# ── Dataset ───────────────────────────────────────────────────────────────────
if script_args.dataset_name == 'Anthropic/hh-rlhf':
    valid_dataset = build_dataset_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions()
elif script_args.dataset_name == 'openai/summarize_from_feedback':
    valid_dataset = build_dataset_summary_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions_summary()
elif script_args.dataset_name == 'argilla/news-summary':
    valid_dataset = build_dataset_news_summary_ppo(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers[0], split='train')
    instructions = Instructions_summary()
elif script_args.dataset_name == 'PKU-Alignment/PKU-SafeRLHF-10K':
    valid_dataset = build_dataset_beaver_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions()
elif script_args.dataset_name in {'nvidia/HelpSteer', 'nvidia/HelpSteer2'}:
    valid_dataset = build_dataset_steer_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers, split=script_args.split)
    instructions = Instructions()
elif script_args.dataset_name == 'openbmb/UltraFeedback':
    valid_dataset = build_dataset_ultrafeedback_eval(
        script_args.dataset_name, tokenizer, reward_models.rm_tokenizers)
    instructions = Instructions()
else:
    raise ValueError(f'Unsupported dataset_name: {script_args.dataset_name!r}')

if script_args.size > 0:
    valid_dataset = valid_dataset.select(range(min(script_args.size, len(valid_dataset))))
print(f'Size of the validation set: {len(valid_dataset)}')

for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)
valid_dataset = valid_dataset.with_format('numpy')

generation_kwargs = {
    'max_new_tokens': 128 if script_args.dataset_name in {
        'Anthropic/hh-rlhf', 'PKU-Alignment/PKU-SafeRLHF-10K',
        'nvidia/HelpSteer', 'nvidia/HelpSteer2'} else 48,
    'min_length':  -1,
    'do_sample':   True,
    'temperature': 0.7,
    'top_p':       0.9,
}

# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate_preference(preference):
    model = LogitsFusionModel(peft_model=peft_model, weights=preference)
    data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=8,
                                   drop_last=False, collate_fn=data_collator)
    acc = Accelerator()
    model, valid_data_loader = acc.prepare(model, valid_data_loader)

    num_cont = script_args.num_continuations

    # For each prompt we run num_cont generations and average rewards
    # full_prompts holds one copy per prompt; rewards are averaged over continuations
    full_prompts = []
    # rewards_accum[i] = list of per-prompt reward arrays for objective i
    rewards_accum = [[] for _ in range(reward_models.num_rewards)]
    # keep one set of responses (last continuation) for CSV logging
    full_responses_last = []

    pbar = tqdm(total=len(valid_dataset) // 8 // acc.num_processes)
    with torch.no_grad():
        for batch in valid_data_loader:
            batch_prompts = tokenizer.batch_decode(batch['input_ids'])
            cont_rewards  = [[] for _ in range(reward_models.num_rewards)]

            for _ in range(num_cont):
                response_tensors = acc.unwrap_model(model).generate(
                    batch['input_ids'], attention_mask=batch['attention_mask'],
                    **generation_kwargs)
                decoded_full = tokenizer.batch_decode(response_tensors)
                decoded_prompts, decoded_responses = get_clean_data(decoded_full, batch_prompts)

                pairs = [(instructions.get_input(t), instructions.get_response(t))
                         for t in decoded_responses]
                if hasattr(instructions, 'get_post'):
                    rlist = reward_models.get_reward_model_scores(
                        pairs, instructions.get_post, normalize_rewards=False)
                else:
                    rlist = reward_models.get_reward_model_scores(pairs, normalize_rewards=False)

                for i in range(reward_models.num_rewards):
                    cont_rewards[i].append(np.array(rlist[i]))

            # average over continuations
            for i in range(reward_models.num_rewards):
                rewards_accum[i].extend(np.mean(np.stack(cont_rewards[i], axis=0), axis=0).tolist())

            full_prompts.extend(decoded_prompts)
            full_responses_last.extend(decoded_responses)
            pbar.update(1)

    all_rewards        = [acc.gather_for_metrics(rewards_accum[i]) for i in range(reward_models.num_rewards)]
    all_full_prompts   = acc.gather_for_metrics(full_prompts)
    all_full_responses = acc.gather_for_metrics(full_responses_last)
    return all_rewards, all_full_prompts, all_full_responses


# ── Preference loop ───────────────────────────────────────────────────────────
print('evaluating........')
sampled_preferences = sample_preferences_uniform(
    reward_models.num_rewards, num_samples=script_args.num_pref_samples)
print(f'\nSampled {len(sampled_preferences)} preferences:')
for i, pref in enumerate(sampled_preferences):
    pref_str = ', '.join(f'{reward_names[k]}={pref[k]:.2f}' for k in range(len(reward_names)))
    print(f'  Pref {i+1}: [{pref_str}]')

for k, preference in enumerate(sampled_preferences):
    print(f'\n[{k+1}/{len(sampled_preferences)}] preference={preference}')
    all_rewards, all_prompts, all_responses = evaluate_preference(preference)
    gc.collect(); torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {'prompt': all_prompts, 'response': all_responses}
        for i in range(reward_models.num_rewards):
            evaluation_result[f'obtained_score{i+1}'] = all_rewards[i]
            print(f'  total average score {i+1}: {np.mean(all_rewards[i]):.4f}')

        dataframe = pd.DataFrame(evaluation_result)
        pref_str  = '_'.join(str(p) for p in preference)
        dataframe.to_csv(os.path.join(output_dir, f'eval_data_pref{pref_str}.csv'),
                         escapechar='\\')
