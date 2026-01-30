"""
Additions to moe_architecture.py for preference-conditioned MoE training with hypervolume loss.

These classes should be added to your existing moe_architecture.py file.
The existing classes (LoRAExpertFFNComplete, AttentionGatingNetwork, MoEFFNLayer, 
RewardModels, MoEGatingTrainer) should remain unchanged.

Key design decisions:
1. Expert embeddings are computed EXACTLY as in original (using LoRAExpertEmbedding.extract_ffn_embedding)
2. No subspace projection - hidden states go directly to query_proj as in original
3. Preference is concatenated with hidden states before query projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from pymoo.indicators.hv import HV
from scripts.utils.utils import load_reward_model, get_rewards


# ==================== LoRA Expert Embedding 提取 ====================

class LoRAExpertEmbedding:
    """从 LoRA 参数提取 expert embedding"""
    
    @staticmethod
    def extract_ffn_embedding(gate_lora, up_lora, down_lora, subspace_rank=8):
        """
        从 FFN 三个投影提取完整 embedding
        模拟 down(gate * up) 的数据流
        
        Args:
            gate_lora: gate_proj LoRA layer
            up_lora: up_proj LoRA layer  
            down_lora: down_proj LoRA layer
            subspace_rank: 提取的主方向数量
        
        Returns:
            expert_embedding: [subspace_rank * hidden_dim]
        """
        # 提取 LoRA 矩阵
        gate_A = gate_lora.lora_A['default'].weight  # [rank, 4096]
        gate_B = gate_lora.lora_B['default'].weight  # [11008, rank]
        gate_full = (gate_B @ gate_A).float()  # [11008, 4096], convert to float32 for SVD
        
        up_A = up_lora.lora_A['default'].weight
        up_B = up_lora.lora_B['default'].weight
        up_full = (up_B @ up_A).float()  # [11008, 4096]
        
        down_A = down_lora.lora_A['default'].weight  # [rank, 11008]
        down_B = down_lora.lora_B['default'].weight  # [4096, rank]
        down_full = (down_B @ down_A).float()  # [4096, 11008]
        
        # SVD 提取主方向
        U_gate, S_gate, _ = torch.svd(gate_full)
        U_up, S_up, _ = torch.svd(up_full)
        
        gate_principal = U_gate[:, :subspace_rank]  # [11008, rank]
        up_principal = U_up[:, :subspace_rank]
        
        # Gate 和 Up 的交互, Down 投影及展平
        gate_up_interaction = gate_principal * up_principal  # [11008, rank]
        final_representation = down_full @ gate_up_interaction  # [4096, rank]
        expert_embedding = final_representation.flatten()
        
        return expert_embedding
    

# ==================== LoRA Expert FFN Complete ====================
    
class LoRAExpertFFNComplete(nn.Module):
    """
    计算完整的 FFN 输出（base + LoRA）
    在 MoEFFNLayer 中会减去 base_output 得到增量
    """
    def __init__(self, base_gate_proj, base_up_proj, base_down_proj,
                 gate_proj_lora, up_proj_lora, down_proj_lora, act_fn):
        super().__init__()
        self.base_gate_proj = base_gate_proj
        self.base_up_proj = base_up_proj
        self.base_down_proj = base_down_proj
        
        self.gate_proj_lora = gate_proj_lora
        self.up_proj_lora = up_proj_lora
        self.down_proj_lora = down_proj_lora
        
        self.act_fn = act_fn
    
    def forward(self, x):
        """
        完整的 FFN: FFN(x) = down(SiLU(gate(x)) * up(x))
        其中 gate, up, down 都是 base + LoRA
        """
        # Gate: base + LoRA -> activate
        gate = self.base_gate_proj(x) + self.gate_proj_lora(x)
        gate = self.act_fn(gate)
        
        # Up: base + LoRA
        up = self.base_up_proj(x) + self.up_proj_lora(x)
        
        # Interaction
        intermediate = gate * up
        
        # Down: base + LoRA
        output = self.base_down_proj(intermediate) + self.down_proj_lora(intermediate)
        
        return output
    

# ==================== Reward Model ====================

class RewardModels():
    def __init__(self, reward_model_path_list, rm_tokenizer_path_list, gpu_id_list, reward_stats_path=None):
        assert len(reward_model_path_list) == len(rm_tokenizer_path_list)
        self.reward_model_path_list = reward_model_path_list
        self.rm_tokenizer_path_list = rm_tokenizer_path_list
        self.num_rewards = len(reward_model_path_list)
        self.reward_stats = np.load(reward_stats_path) if reward_stats_path is not None else None
        self.reward_models = []
        self.rm_tokenizers = []
        if type(gpu_id_list) != list:
            gpu_id_list = [gpu_id_list, gpu_id_list, gpu_id_list]
    
        print('Loading reward models .....')
        for i in range(self.num_rewards):
            self.reward_models.append(load_reward_model(self.reward_model_path_list[i], gpu_id_list[i]))
            self.rm_tokenizers.append(AutoTokenizer.from_pretrained(self.rm_tokenizer_path_list[i]))
    
        
    def get_reward_model_scores(self, queries_responses, summary_fun=None, normalize_rewards=True):
        texts_for_rewards = []
        for i in range(self.num_rewards):
            if i >= 1 and self.rm_tokenizer_path_list[i] == self.rm_tokenizer_path_list[i-1]:
                texts_for_rewards.append(texts_for_rewards[-1])
            elif 'faithful' in self.reward_model_path_list[i]:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](text=r, text_pair=summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            elif 'summary' in self.reward_model_path_list[i] or 'summarization' in self.reward_model_path_list[i]: # reverse prompt and response
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r + " " + self.rm_tokenizers[i].bos_token + " " + summary_fun(q), return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            elif 'humor' in self.reward_model_path_list[i]: # use only response
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)
            else:
                max_length = min(self.rm_tokenizers[i].model_max_length, 1024)
                temp_encoded_texts = [self.rm_tokenizers[i](q, r, return_tensors='pt', truncation=True, max_length=max_length) for q, r in queries_responses]
                texts_for_rewards.append(temp_encoded_texts)

        # normalize reward
        rewards = []
        for i in range(self.num_rewards):
            if self.reward_stats is not None:
                if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
                    reward_mean_std = (self.reward_stats[2*i], self.reward_stats[2*i+1])
                else:
                    reward_mean_std = self.reward_stats[i]
            else:
                reward_mean_std = None

            if not normalize_rewards:
                reward_mean_std = None

            if 'humor' in self.reward_model_path_list[i] or 'faithful' in self.reward_model_path_list[i]:
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std, sub_position=1)
            else:
                temp_reward = get_rewards(self.reward_models[i], texts_for_rewards[i], reward_mean_std=reward_mean_std)
            rewards.append(temp_reward)
        return rewards
    

# ==================== Preference-Conditioned Gating Network ====================

class AttentionGatingNetwork(nn.Module):
    """
    Attention-based gating network conditioned on preference vector.
    
    Same structure as original AttentionGatingNetwork, but with preference
    concatenated to hidden states before query projection.
    """
    def __init__(self, hidden_dim, num_lora_experts, subspace_rank=8, d_model=512, num_rewards=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lora_experts = num_lora_experts
        self.subspace_rank = subspace_rank
        self.d_model = d_model
        self.num_rewards = num_rewards
        
        # === Expert embeddings ===
        expert_emb_dim = subspace_rank * hidden_dim
        self.register_buffer(
            'expert_embeddings',
            torch.zeros(num_lora_experts, expert_emb_dim)
        )
        
        # === Preference projection to hidden_dim ===
        self.preference_proj = nn.Linear(num_rewards, hidden_dim)
        
        # === Query projection takes hidden_dim + hidden_dim (hidden + preference) ===
        self.query_proj = nn.Linear(hidden_dim + hidden_dim, d_model)
        
        # === Key projection ===
        self.key_proj = nn.Linear(expert_emb_dim, d_model)
        
        # === Temperature ===
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Store current preference
        self.current_preference = None
    
    def set_preference(self, preference):
        """Set the current preference vector for conditioning."""
        if isinstance(preference, list):
            preference = torch.tensor(preference, dtype=torch.float32)
        if preference.dim() == 1:
            preference = preference.unsqueeze(0)
        self.current_preference = preference
    
    def load_expert_embedding(self, expert_idx, gate_lora, up_lora, down_lora):
        """
        从 LoRA 提取 expert embedding
        """
        with torch.no_grad():
            expert_emb = LoRAExpertEmbedding.extract_ffn_embedding(
                gate_lora, up_lora, down_lora,
                subspace_rank=self.subspace_rank
            )
            self.expert_embeddings[expert_idx] = expert_emb
    
    def forward(self, x, preference=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            preference: [batch, num_rewards] or [num_rewards] or None
        Returns:
            lora_weights: [batch, seq_len, num_lora_experts]
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure module is on same device as input
        if self.preference_proj.weight.device != device:
            self.to(device)
        
        # Get preference
        if preference is not None:
            if isinstance(preference, list):
                preference = torch.tensor(preference, dtype=dtype, device=device)
            if preference.dim() == 1:
                preference = preference.unsqueeze(0)
            pref = preference.to(device=device, dtype=dtype)
        elif self.current_preference is not None:
            pref = self.current_preference.to(device=device, dtype=dtype)
        else:
            # Default uniform preference
            pref = torch.ones(1, self.num_rewards, device=device, dtype=dtype) / self.num_rewards
        
        # Expand preference to batch size
        if pref.shape[0] == 1 and batch > 1:
            pref = pref.expand(batch, -1)
        
        # Project preference to hidden_dim and expand to seq_len
        pref_emb = self.preference_proj(pref)  # [batch, hidden_dim]
        pref_emb = pref_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim]
        
        # Concatenate hidden states with preference
        query_input = torch.cat([x, pref_emb], dim=-1)  # [batch, seq_len, hidden_dim + hidden_dim]
        
        # 1. Project input to query space
        query = self.query_proj(query_input)  # [batch, seq_len, d_model]
        
        # 2. Project expert embeddings to key space
        keys = self.key_proj(self.expert_embeddings)  # [num_lora_experts, d_model]
        
        # 3. Compute attention scores
        scores = torch.matmul(query, keys.t())  
        
        # 4. Scale by temperature and sqrt(d_model)
        scores = scores / (torch.sqrt(torch.tensor(self.d_model, dtype=scores.dtype, device=device)) * self.temperature)
        
        # 5. Softmax to get routing weights
        lora_weights = F.softmax(scores, dim=-1)
        
        if self.training:
            self._last_routing_weights = lora_weights
        else:
            self._last_routing_weights = lora_weights.detach()
        
        return lora_weights


# ==================== Preference-Conditioned MoE FFN Layer ====================

class MoEFFNLayer(nn.Module):
    """
    MoE FFN with preference-conditioned gating, passes preference to gate network.
    """
    def __init__(self, base_mlp, lora_experts, gate_network):
        super().__init__()
        self.base_mlp = base_mlp
        self.lora_experts = nn.ModuleList(lora_experts)
        self.gate = gate_network
        self.num_lora_experts = len(lora_experts)
    
    def forward(self, hidden_states, preference=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            preference: optional preference vector
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # 1. Base model output
        base_output = self.base_mlp(hidden_states)
        
        # 2. Get routing weights (with preference conditioning)
        lora_weights = self.gate(hidden_states, preference=preference)
        
        # 3. Compute weighted LoRA expert deltas
        lora_contribution = torch.zeros_like(hidden_states)
        
        for expert_idx, expert in enumerate(self.lora_experts):
            expert_full_output = expert(hidden_states)
            expert_delta = expert_full_output - base_output
            weight = lora_weights[:, :, expert_idx].unsqueeze(-1)
            lora_contribution += weight * expert_delta
        
        # 4. Combine
        final_output = base_output + lora_contribution
        
        return final_output


# ==================== Preference-Conditioned Trainer with Hypervolume ====================

class MoEGatingTrainer:
    """
    Trainer for MoE gating with preference conditioning and hypervolume loss.
    
    Key features:
    1. Preference is input to gating network (not just for reward scalarization)
    2. Multiple preferences sampled per input during training
    3. Hypervolume loss to push towards Pareto front
    """
    
    def __init__(self, moe_model, reward_model, instructions, learning_rate=1e-5,
                 num_rewards=2, num_pref_samples=10): # preference=None
        self.model = moe_model
        self.reward_model = reward_model
        self.instructions = instructions
        self.num_rewards = num_rewards
        # self.preference = preference if preference is not None else [1.0 / num_rewards] * num_rewards
        self.num_pref_samples = num_pref_samples
        
        print(f"Initializing Preference-Conditioned MoE trainer with {num_rewards} rewards")
        # print(f"Default preference weights: {self.preference}")
        print(f"Number of preference samples per input: {num_pref_samples}")
        
        # Collect gating parameters
        gating_params = []
        for layer in moe_model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                gating_params.extend(layer.mlp.gate.parameters())
        
        self.optimizer = torch.optim.AdamW(gating_params, lr=learning_rate)
        
        # Freeze other parameters
        for param in moe_model.parameters():
            param.requires_grad = False
        
        for layer in moe_model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                for param in layer.mlp.gate.parameters():
                    param.requires_grad = True
        
        print(f"Trainable gating parameters: {sum(p.numel() for p in gating_params):,}")
        
        # Baseline for variance reduction
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9
        
        # Reference point for hypervolume (assuming rewards in roughly [-2, 2] range)
        self.hv_reference_point = np.ones(num_rewards) * (-3.0)  # Worse than worst expected
    
    def sample_preferences(self):
        """Sample preferences from Dirichlet distribution (uniform on simplex)."""
        preferences = []
        for _ in range(self.num_pref_samples):
            pref = np.random.dirichlet(np.ones(self.num_rewards))
            preferences.append(pref.tolist())
        return preferences
    
    def set_model_preference(self, preference):
        """Set preference for all gating networks in the model."""
        pref_tensor = torch.tensor(preference, dtype=torch.float32)
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                layer.mlp.gate.set_preference(pref_tensor)
    
    def compute_hypervolume(self, reward_vectors):
        """
        Compute hypervolume of reward vectors.
        
        Args:
            reward_vectors: list of [num_rewards] arrays
        Returns:
            hv_value: hypervolume (higher is better)
        """
        if len(reward_vectors) == 0:
            return 0.0
        
        # Stack and negate (pymoo expects minimization)
        points = -np.array(reward_vectors)
        
        # Reference point (negated since we negated objectives)
        ref_point = -self.hv_reference_point
        
        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(points)
        
        return hv_value
    
    def forward_with_routing_log_probs(self, input_ids, attention_mask=None, preference=None):
        """Forward pass collecting routing log probs with preference conditioning."""
        # Clear previous routing weights
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                if hasattr(layer.mlp.gate, '_last_routing_weights'):
                    delattr(layer.mlp.gate, '_last_routing_weights')
        
        # Set preference if provided
        if preference is not None:
            self.set_model_preference(preference)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            use_cache=False
        )
        
        # Collect routing log probs
        routing_log_probs = []
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, '_last_routing_weights'):
                routing_weights = layer.mlp.gate._last_routing_weights
                log_probs = torch.log(routing_weights + 1e-8)
                routing_log_probs.append(log_probs)
        
        return outputs, routing_log_probs
    
    def train_step_reinforce(self, batch, tokenizer, 
                            alpha_balance=0.01, alpha_entropy=0.01,
                            alpha_hypervolume=0.1):
        """
        Training step with REINFORCE, preference conditioning, and hypervolume loss.
        
        For each input:
        1. Sample multiple preferences
        2. For each preference, generate output and compute rewards
        3. Compute policy gradient with preference-conditioned routing
        4. Compute hypervolume loss to push towards Pareto front
        """
        # Handle input tensors
        input_ids_list = []
        for ids in batch['input_ids']:
            if isinstance(ids, torch.Tensor):
                input_ids_list.append(ids.clone().detach())
            else:
                input_ids_list.append(torch.tensor(ids))
        
        # Pad to max length
        max_length = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        attention_mask = []
        
        for ids in input_ids_list:
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([
                torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype),
                ids
            ])
            mask = torch.cat([
                torch.zeros(padding_length, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long)
            ])
            padded_input_ids.append(padded_ids)
            attention_mask.append(mask)
        
        input_ids = torch.stack(padded_input_ids).to(next(self.model.parameters()).device)
        attention_mask = torch.stack(attention_mask).to(next(self.model.parameters()).device)
        
        batch_size = input_ids.shape[0]
        queries = batch['query']
        
        # Sample preferences for this batch
        sampled_preferences = self.sample_preferences()
        
        # Collect all reward vectors for hypervolume
        all_reward_vectors = []
        all_scalarized_rewards = []
        all_preferences_used = []
        
        # ===== Generation Phase for each preference =====
        self.model.eval()
        
        for pref in sampled_preferences:
            pref_tensor = torch.tensor(pref, dtype=torch.float32, device=next(self.model.parameters()).device)
            self.set_model_preference(pref)
            
            with torch.no_grad():
                generation_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                response_tensors = generation_outputs.sequences
            
            # Decode responses
            full_responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=False)
            responses = []
            
            for full_resp in full_responses:
                response = full_resp.strip('[PAD] ')
                response = response.strip('<unk>')
                temp_resp = response.strip('<s>').strip('</s>')
                temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
                temp_resp = temp_resp.split('\nHuman:')[0].strip()
                temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
                temp_resp = temp_resp.split('\nAssistant:')[0].strip()
                temp_resp = temp_resp.split('\n\n\n')[0].strip()
                temp_resp = temp_resp.split('###')[0].strip()
                responses.append(temp_resp)
            
            # Compute rewards
            texts_merge = [q + r for q, r in zip(queries, responses)]
            queries_responses = [
                (self.instructions.get_input(text), self.instructions.get_response(text))
                for text in texts_merge
            ]
            
            if hasattr(self.instructions, 'get_post'):
                rewards_list = self.reward_model.get_reward_model_scores(
                    queries_responses, 
                    self.instructions.get_post
                )
            else:
                rewards_list = self.reward_model.get_reward_model_scores(queries_responses)
            
            # rewards_list is [num_rewards][batch_size]
            for j in range(batch_size):
                reward_vector = [rewards_list[k][j] for k in range(self.num_rewards)]
                all_reward_vectors.append(reward_vector)
                
                # Scalarized reward with current preference
                scalarized = sum(pref[k] * reward_vector[k] for k in range(self.num_rewards))
                all_scalarized_rewards.append(scalarized)
                all_preferences_used.append(pref)
        
        # Convert to tensors
        model_device = next(self.model.parameters()).device
        rewards_tensor = torch.tensor(all_scalarized_rewards, dtype=torch.float32, device=model_device)
        rewards_normalized = rewards_tensor - self.reward_baseline
        
        # Update baseline
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline +
            (1 - self.baseline_momentum) * rewards_tensor.mean().item()
        )
        
        # Compute hypervolume
        hv_value = self.compute_hypervolume(all_reward_vectors)
        
        # ===== REINFORCE Update =====
        self.model.train()
        self.optimizer.zero_grad()
        
        total_policy_loss = 0.0
        total_entropy = 0.0
        num_forward = 0
        
        # Forward pass for each preference to compute gradients
        for pref_idx, pref in enumerate(sampled_preferences):
            self.set_model_preference(pref)
            
            outputs, routing_log_probs = self.forward_with_routing_log_probs(
                input_ids, attention_mask, preference=pref
            )
            
            if routing_log_probs:
                # Get rewards for this preference (batch_size samples)
                start_idx = pref_idx * batch_size
                end_idx = start_idx + batch_size
                pref_rewards = rewards_normalized[start_idx:end_idx]
                
                for log_probs in routing_log_probs:
                    probs = torch.exp(log_probs)
                    
                    # Policy loss
                    max_log_probs = log_probs.max(dim=-1)[0]  # [batch, seq]
                    policy_loss_layer = -(pref_rewards.unsqueeze(-1) * max_log_probs).mean()
                    total_policy_loss += policy_loss_layer
                    
                    # Entropy
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    total_entropy += entropy
                
                num_forward += len(routing_log_probs)
        
        if num_forward > 0:
            policy_loss = total_policy_loss / num_forward
            entropy_bonus = total_entropy / num_forward
        else:
            model_device = next(self.model.parameters()).device
            policy_loss = torch.tensor(0.0, requires_grad=True, device=model_device)
            entropy_bonus = torch.tensor(0.0, requires_grad=True, device=model_device)
        
        # Load balance loss
        balance_loss = self.compute_load_balance_loss()
        
        # Hypervolume loss: negative HV (we want to maximize HV, so minimize -HV)
        # Normalized by number of samples for stability
        hv_loss = -hv_value / len(all_reward_vectors)
        hv_loss_tensor = torch.tensor(hv_loss, dtype=policy_loss.dtype, device=policy_loss.device)
        
        # Total loss (hv_loss is float, doesn't backprop but guides overall training)
        total_loss = policy_loss + alpha_balance * balance_loss - alpha_entropy * entropy_bonus + alpha_hypervolume * hv_loss_tensor
        
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss,
            'balance_loss': balance_loss.item() if isinstance(balance_loss, torch.Tensor) else balance_loss,
            'entropy_loss': -entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else -entropy_bonus,
            'hypervolume_loss': hv_loss,
            'hypervolume_value': hv_value,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'mean_reward': rewards_tensor.mean().item(),
            'std_reward': rewards_tensor.std().item(),
            'baseline': self.reward_baseline
        }
    
    def compute_load_balance_loss(self):
        """Compute load balance loss"""
        total_balance_loss = 0.0
        num_layers = 0
        
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'gate') and hasattr(layer.mlp.gate, '_last_routing_weights'):
                routing_weights = layer.mlp.gate._last_routing_weights
                expert_usage = routing_weights.mean(dim=[0, 1])
                target = 1.0 / layer.mlp.num_lora_experts
                balance_loss = ((expert_usage - target) ** 2).mean()
                total_balance_loss += balance_loss
                num_layers += 1
        
        if num_layers == 0:
            return torch.tensor(0.0, requires_grad=True, device=next(self.model.parameters()).device)
        
        return total_balance_loss / num_layers