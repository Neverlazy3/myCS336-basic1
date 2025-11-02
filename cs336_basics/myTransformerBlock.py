import torch
import torch.nn as nn
from transformers import torch_distributed_zero_first
from torch import Tensor
from jaxtyping import Float, Int
import einops

from cs336_basics.mySwiGLU import SwiGLU, SwiGLUFFN

from .myRMSNorm import RMSNorm
from .myAttenation import MultiHeadSelfAttentionWithRoPE_V2

class transformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
        ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.head_dim = d_model // num_heads
        
        self.attn = MultiHeadSelfAttentionWithRoPE_V2(self.d_model, self.num_heads, self.max_seq_len, self.theta)
        self.ln1 = RMSNorm(self.d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(self.d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(self.d_model, self.d_ff, device=device, dtype=dtype)
        
    def forward(
        self, 
        x: Float[Tensor, " batch sequence_length d_model"], 
        token_positions: Int[Tensor, " ... sequence_length"] | None,
        ) -> Float[Tensor, " batch sequence_length d_model"]:
        if token_positions is None:
            token_positions = torch.arange(x.size(1), device=x.device).expand(
                x.size(0), -1
            )
            
        x_type = x.dtype
        input_attention = x.to(torch.float32)
        
        out_atten_rms = self.ln1(input_attention)
        out_mha_rope = self.attn(out_atten_rms, token_positions)

        input_fnn = out_mha_rope + input_attention
        output_ffn_rms = self.ln2(input_fnn)
        output_ffn_last = self.ffn(output_ffn_rms)
        out_final = output_ffn_last + input_fnn
        
        return out_final.to(x_type)