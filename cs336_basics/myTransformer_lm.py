'''
Author: harmoon 92302745+Neverlazy3@users.noreply.github.com
Date: 2025-10-30 10:52:20
LastEditors: harmoon 92302745+Neverlazy3@users.noreply.github.com
LastEditTime: 2025-10-30 20:55:02
FilePath: \assignment1-basics-main\cs336_basics\myTransformer_lm.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from turtle import forward
import tenacity
import torch
import torch.nn as nn
from jaxtyping import Int, Float
from torch import Tensor, device, is_floating_point, tensor

from cs336_basics.myEmbedding import Embedding
from cs336_basics.myLinear import Linear
from cs336_basics.myModuleList import transModuleList
from cs336_basics.myRMSNorm import RMSNorm
from cs336_basics.myTransformerBlock import transformerBlock

class Transformer_lm(nn.Module):
    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device=None,
                dtype=None
                ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.head_dim = d_model / num_heads
        self.device = device
        self.dtype = dtype
        
        dtype = (dtype 
                if(
                    dtype is not None 
                    and torch.is_floating_point(torch.tensor([], dtype=dtype))
                    )
                else torch.float32
                )
        
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = transModuleList(
            [transformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) 
            for _ in range(num_layers) ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)
        
    @torch.no_grad()
    def forward(self, x: Int[Tensor, " batch_size sequence_length"], token_positions: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x) #这里之所以没有先转变成float32，是因为embedding中需要x作为索引，而索引不应该是float类型
        
        input_type = x.dtype
        x = x.to(torch.float32)
        
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device).expand(
                x.size(0), -1
            )
        
        for layer in self.layers:
            x = layer(x, token_positions)
        
        out_rms = self.ln_final(x)
        logits = self.lm_head(out_rms)
        return logits.to(input_type)