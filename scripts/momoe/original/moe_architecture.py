import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from pymoo.indicators.hv import HV

script_dir = Path(__file__).resolve().parent  # project/scripts/momoe
project_root = script_dir.parent.parent       # project/
sys.path.insert(0, str(project_root))
from scripts.utils.utils import load_reward_model, get_rewards, get_clean_data


# ==================== LoRA Expert Embedding extraction ====================

class LoRAExpertEmbedding:
    """Extract expert embedding from LoRA parameters"""
    
    @staticmethod
    def extract_ffn_embedding(gate_lora, up_lora, down_lora, subspace_rank=8):
        """
        Extract complete embedding from the three FFN projections
        Simulate the data flow of down(gate * up)
        
        Args:
            gate_lora: gate_proj LoRA layer
            up_lora: up_proj LoRA layer  
            down_lora: down_proj LoRA layer
            subspace_rank: number of principal directions to extract
        
        Returns:
            expert_embedding: [subspace_rank * hidden_dim]
        """
        # Extract LoRA matrices
        gate_A = gate_lora.lora_A['default'].weight  # [rank, 4096]
        gate_B = gate_lora.lora_B['default'].weight  # [11008, rank]
        gate_full = (gate_B @ gate_A).float()  # [11008, 4096], convert to float32 for SVD
        
        up_A = up_lora.lora_A['default'].weight
        up_B = up_lora.lora_B['default'].weight
        up_full = (up_B @ up_A).float()  # [11008, 4096]
        
        down_A = down_lora.lora_A['default'].weight  # [rank, 11008]
        down_B = down_lora.lora_B['default'].weight  # [4096, rank]
        down_full = (down_B @ down_A).float()  # [4096, 11008]
        
        # SVD to extract principal directions
        U_gate, S_gate, _ = torch.svd(gate_full)
        U_up, S_up, _ = torch.svd(up_full)
        
        gate_principal = U_gate[:, :subspace_rank]  # [11008, rank]
        up_principal = U_up[:, :subspace_rank]
        
        # Interaction of Gate and Up, projection by Down and flatten
        gate_up_interaction = gate_principal * up_principal  # [11008, rank]
        final_representation = down_full @ gate_up_interaction  # [4096, rank]
        expert_embedding = final_representation.flatten()
        
        return expert_embedding
    

# ==================== LoRA Expert FFN Complete ====================
    
class LoRAExpertFFNComplete(nn.Module):
    """
    Complete FFN output (base + LoRA)
    The increment is obtained by subtracting base_output in MoEFFNLayer
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
        Complete FFN: FFN(x) = down(SiLU(gate(x)) * up(x))
        where gate, up, down are all base + LoRA
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
        
        # Expert embeddings
        expert_emb_dim = subspace_rank * hidden_dim
        self.register_buffer(
            'expert_embeddings',
            torch.zeros(num_lora_experts, expert_emb_dim)
        )
        
        # Preference projection to hidden_dim
        self.preference_proj = nn.Linear(num_rewards, hidden_dim)
        
        # Query projection takes hidden_dim + hidden_dim (hidden + preference)
        self.query_proj = nn.Linear(hidden_dim + hidden_dim, d_model)
        
        # Key projection
        self.key_proj = nn.Linear(expert_emb_dim, d_model)
        
        # Temperature
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
        From LoRA extract expert embedding
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
    
    def __init__(self, moe_model, reward_models, instructions, learning_rate=1e-5,
                 num_rewards=2, num_pref_samples=10): # preference=None
        self.model = moe_model
        self.model.gradient_checkpointing_enable()
        self.reward_models = reward_models
        self.instructions = instructions
        self.num_rewards = num_rewards
        # self.preference = preference if preference is not None else [1.0 / num_rewards] * num_rewards
        self.num_pref_samples = num_pref_samples
        
        print(f"Initializing Preference-Conditioned MoE trainer with {num_rewards} rewards")
        # print(f"Default preference weights: {self.preference}")
        print(f"Number of preference samples per input: {num_pref_samples}")
        
        # Collect gating parameters
        gating_params = []
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                gating_params.extend(layer.mlp.gate.parameters())
        
        self.optimizer = torch.optim.AdamW(gating_params, lr=learning_rate)
        
        # Freeze other parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'gate'):
                for param in layer.mlp.gate.parameters():
                    param.requires_grad = True
        
        print(f"Trainable gating parameters: {sum(p.numel() for p in gating_params):,}")
        
        # Baseline for variance reduction
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9
    
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
        ref_point = np.ones(len(reward_vectors[0])) * 4.0
        
        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(points)
        
        return hv_value

    def compute_load_balance(self):
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
        
        sampled_preferences = self.sample_preferences()
    
        all_reward_vectors = []
        all_scalarized_rewards = []
        
        # Accumulate gradients across preferences
        self.optimizer.zero_grad()
        accumulated_policy_loss = 0.0
        accumulated_entropy = 0.0
        accumulated_balance_loss = 0.0
        
        for pref_idx, pref in enumerate(sampled_preferences):
            self.set_model_preference(pref)
            
            # === Generation Phase (no grad) ===
            self.model.eval()
            with torch.no_grad():
                response_tensors = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # response_tensors = generation_outputs.sequences
            
            # Decode and get rewards
            full_responses = tokenizer.batch_decode(response_tensors)
            full_prompts = tokenizer.batch_decode(input_ids)
            full_prompts, full_responses = get_clean_data(full_responses, full_prompts)
            
            queries_responses = [
                (self.instructions.get_input(text), self.instructions.get_response(text))
                for text in full_responses
            ]
            
            if hasattr(self.instructions, 'get_post'):
                rewards_list = self.reward_models.get_reward_model_scores(
                    queries_responses, self.instructions.get_post
                )
            else:
                rewards_list = self.reward_models.get_reward_model_scores(queries_responses)
            
            # Compute rewards for this preference
            for j in range(batch_size):
                reward_vector = [rewards_list[k][j] for k in range(self.num_rewards)]
                all_reward_vectors.append(reward_vector)
                scalarized = sum(pref[k] * reward_vector[k] for k in range(self.num_rewards)) # TODO
                all_scalarized_rewards.append(scalarized)
            
            # === REINFORCE Update (with grad) - immediately after generation ===
            self.model.train()
            
            pref_rewards = torch.tensor(
                all_scalarized_rewards[-batch_size:],  # Only this preference's rewards
                dtype=torch.float32, 
                device=input_ids.device
            )
            pref_rewards_normalized = pref_rewards - self.reward_baseline
            
            outputs, routing_log_probs = self.forward_with_routing_log_probs(
                input_ids, attention_mask, preference=pref
            )
            
            if routing_log_probs:
                for log_probs in routing_log_probs:
                    probs = torch.exp(log_probs)
                    max_log_probs = log_probs.max(dim=-1)[0]
                    policy_loss = -(pref_rewards_normalized.unsqueeze(-1) * max_log_probs).mean()
                    entropy = (probs * log_probs).sum(dim=-1).mean()
                    
                    accumulated_policy_loss += policy_loss / len(sampled_preferences)
                    accumulated_entropy += entropy / len(sampled_preferences)
            
            accumulated_balance_loss += self.compute_load_balance() / len(sampled_preferences)
            
            # Clear cache after each preference
            del response_tensors, outputs, routing_log_probs
            torch.cuda.empty_cache()
        
        # Update baseline
        rewards_tensor = torch.tensor(all_scalarized_rewards, dtype=torch.float32)
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline +
            (1 - self.baseline_momentum) * rewards_tensor.mean().item()
        )
        
        # Compute hypervolume
        hv_value = self.compute_hypervolume(all_reward_vectors)
        hv_loss = -hv_value / len(all_reward_vectors)
        hv_loss_tensor = torch.tensor(hv_loss, device=input_ids.device)
        
        # Total loss
        total_loss = (accumulated_policy_loss + 
                    alpha_balance * accumulated_balance_loss + 
                    alpha_entropy * accumulated_entropy) 
                    # + alpha_hypervolume * hv_loss_tensor
        
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.optimizer.step()
        
        return {
            'policy_loss': accumulated_policy_loss.item(),
            'balance_loss': accumulated_balance_loss.item(),
            'entropy_loss': accumulated_entropy.item(),
            'hypervolume_loss': hv_loss,
            'hypervolume_value': hv_value,
            'total_loss': total_loss.item(),
            'mean_reward': rewards_tensor.mean().item(),
            'std_reward': rewards_tensor.std().item(),
            'baseline': self.reward_baseline
        }
