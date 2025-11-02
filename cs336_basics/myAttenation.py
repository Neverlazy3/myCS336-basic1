'''
Author: harmoon 92302745+Neverlazy3@users.noreply.github.com
Date: 2025-10-28 09:43:21
LastEditors: harmoon 92302745+Neverlazy3@users.noreply.github.com
LastEditTime: 2025-10-30 10:30:42
FilePath: \assignment1-basics-main\cs336_basics\myAttenation.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import token
from cycler import V
from regex import T
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import math

from cs336_basics.myLinear import Linear
from .mySoftMax import softmax
from .myRoPE import RoPE
from einops import einsum, rearrange

class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    
    def forward(
        self,  
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        in_type = Q.dtype
        Q = Q.to(torch.float32)
        
        q_k = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        input_softmax = q_k / math.sqrt(d_k)
        if mask is not None:
            input_softmax = input_softmax.masked_fill(~mask, float("-inf"))
            
        softmax_ret = softmax(input_softmax, dim=-1)
        attention = einsum(softmax_ret, V, "... queries keys, ... keys d_v -> ... queries d_v")
        out = attention
        return out.to(in_type)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
    def forward(
        self, 
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        in_type = in_features.dtype
        in_features = in_features.to(torch.float32)
        
        Q = einsum(q_proj_weight, in_features, " d_k d_in, ... seq_len d_in -> ... seq_len d_k")
        K = einsum(k_proj_weight, in_features, " d_k d_in, ... seq_len d_in -> ... seq_len d_k")
        V = einsum(v_proj_weight, in_features, " d_v d_in, ... seq_len d_in -> ... seq_len d_v")
        
        Q = rearrange(Q, "... seq_len (head head_dim) -> ... head seq_len head_dim", head=self.num_heads)
        K = rearrange(K, "... seq_len (head head_dim) -> ... head seq_len head_dim", head=self.num_heads)
        V = rearrange(V, "... seq_len (head head_dim) -> ... head seq_len head_dim", head=self.num_heads)
        
        q_k = einsum(Q, K, "... head queries head_dim, ... head keys head_dim -> ... head queries keys")
        
        seq_len = q_k.shape[-1]
        input_softMax = q_k / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
        input_softMax = input_softMax.masked_fill(causal_mask, float("-inf"))
        
        softMax_ret = softmax(input_softMax, dim=-1)
        # attention = einsum(softMax_ret, V, "... head queries keys, ... head keys head_dim -> ... head sequence head_dim")
        attention = einsum(softMax_ret, V, "... head query keys, ... head keys head_dim -> ... head query head_dim")
        multi_head = rearrange(attention, "... head seq head_dim -> ... seq (head head_dim)")
        multi_head_self_attention = einsum(multi_head, o_proj_weight, "... seq d_k, d_out d_k -> ... seq d_out")
        
        return multi_head_self_attention.to(in_type)
    
    
    
class MultiHeadSelfAttentionWithRoPe(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.head_dim = d_model // num_heads
        self.rope = RoPE(self.theta, self.head_dim, self.max_seq_len)
        
    def forward(
        self,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"], # d_v = d_k = d_model
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        assert token_positions != None, "token_position can't be None"
        # x_dim = in_features.shape[-2]
        in_type = in_features.dtype
        in_features = in_features.to(torch.float32)
        
        x = in_features
        
        Q = einsum(x, q_proj_weight," ... seq d_in, d_k d_in -> ... seq d_k")
        K = einsum(x, k_proj_weight," ... seq d_in, d_k d_in -> ... seq d_k")
        V = einsum(x, v_proj_weight," ... seq d_in, d_k d_in -> ... seq d_k")
        
        Q = rearrange(Q, "... seq (head head_dim) -> ... head seq head_dim", head=self.num_heads)
        K = rearrange(K, "... seq (head head_dim) -> ... head seq head_dim", head=self.num_heads)
        V = rearrange(V, "... seq (head head_dim) -> ... head seq head_dim", head=self.num_heads)
        
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        q_k = einsum(Q, K, "... head query head_dim, ... head keys head_dim -> ... head query keys")
        seq_len = q_k.shape[-1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
        softmax_input = q_k / math.sqrt(self.head_dim)
        softmax_input = softmax_input.masked_fill(causal_mask, float("-inf"))
        softmax_output = softmax(softmax_input, dim=-1)
        
        attention = einsum(softmax_output, V, "... head query keys, ... head keys head_dim -> ... head query head_dim")
        multi_head_attention = rearrange(attention, "... head seq head_dim -> ... seq (head head_dim)")
        multi_head_self_attention = einsum(multi_head_attention, o_proj_weight, "... seq dim, d_model dim -> ... seq d_model")
        
        return multi_head_self_attention.to(in_type)
    
    
class MultiHeadSelfAttentionWithRoPE_V2(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.head_dim = d_model // num_heads
        self.rope = RoPE(self.theta, self.head_dim, self.max_seq_len)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model) # 何以为？为什么要有o？
        
    def forward(self, 
                in_features: Float[Tensor, " ... sequence_length d_in"], 
                token_positions: Int[Tensor, " ... sequence_length"]
                ) -> Float[Tensor, " ... sequence_length d_out"]:
        # assert token_positions != None, "token_position can't be None"
        # x_dim = in_features.shape[-2]
        in_type = in_features.dtype
        in_features = in_features.to(torch.float32)
        
        x = in_features
        Q = rearrange(self.q_proj(x), "... seq (head head_dim) -> ... head seq head_dim", head=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (head head_dim) -> ... head seq head_dim", head=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (head head_dim) -> ... head seq head_dim", head=self.num_heads)
        
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        
        q_k = einsum(Q, K, "... head query head_dim, ... head keys head_dim -> ... head query keys")
        seq_len = q_k.shape[-1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
        softmax_input = q_k / math.sqrt(self.head_dim)
        softmax_input = softmax_input.masked_fill(causal_mask, float("-inf"))
        softmax_output = softmax(softmax_input, dim=-1)
        
        attention = einsum(softmax_output, V, "... head query keys, ... head keys head_dim -> ... head query head_dim")
        multi_head_attention = rearrange(attention, "... head seq head_dim -> ... seq (head head_dim)")
        
        return self.output_proj(multi_head_attention).to(in_type)