"""
Fallback implementations of layers without FlashAttention dependency
"""

from typing import Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_

CosSin = Tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float = 1e-6):
    """RMS normalization"""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Fallback attention implementation using standard PyTorch"""
    
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        
        self.q_proj = CastedLinear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = CastedLinear(hidden_size, num_key_value_heads * head_dim, bias=False)  
        self.v_proj = CastedLinear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = CastedLinear(num_heads * head_dim, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        
        # Apply rotary embeddings if available
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])
        
        # Transpose for attention computation: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)
        
        # Repeat k,v heads if needed (for grouped query attention)
        if self.num_key_value_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
            scores.masked_fill_(mask.bool(), float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape: [batch, seq_len, num_heads * head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Final projection
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, hidden_size, expansion):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * expansion)
        
        self.gate_proj = CastedLinear(hidden_size, self.intermediate_size, bias=False)
        self.up_proj = CastedLinear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = CastedLinear(self.intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = F.silu(gate) * up
        return self.down_proj(intermediate)