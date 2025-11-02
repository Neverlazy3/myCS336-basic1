'''
Author: harmoon 92302745+Neverlazy3@users.noreply.github.com
Date: 2025-10-25 21:28:21
LastEditors: harmoon 92302745+Neverlazy3@users.noreply.github.com
LastEditTime: 2025-10-29 22:05:41
FilePath: \assignment1-basics-main\cs336_basics\mySwiGLU.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from turtle import forward
import torch
import torch.nn as nn
from torch import Tensor
from .myLinear import Linear
def SiLU(x: Tensor):
    in_type = x.dtype
    x = x.to(torch.float32)
    x = x * torch.sigmoid(x)
    return x.to(in_type)
    
class SwiGLU(nn.Module):
    def __init__(self, d_modle, d_ff, device=None, dtype=None) -> None:
        super().__init__()
        self.d_modle = d_modle
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_modle, device=self.device, dtype=self.dtype))
        self.w2_weight = nn.Parameter(torch.empty(d_modle, d_ff, device=self.device, dtype=self.dtype))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_modle, device=self.device, dtype=self.dtype))
        std = (2 / (d_modle + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1_weight, mean=0, std=std, a=(-3)*std, b=3*std)
        nn.init.trunc_normal_(self.w2_weight, mean=0, std=std, a=(-3)*std, b=3*std)
        nn.init.trunc_normal_(self.w3_weight, mean=0, std=std, a=(-3)*std, b=3*std)
        
        
        
    def forward(self, x: Tensor) -> Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        siLu_w1x = SiLU(torch.matmul(x, self.w1_weight.t()))
        w3x = torch.matmul(x, self.w3_weight.t())
        w2x = torch.matmul((siLu_w1x * w3x), self.w2_weight.t())
        
        return w2x.to(in_type)
    
class SwiGLUFFN(nn.Module):
    def __init__(self,
                d_model: int,
                d_ff: int,
                device=None,
                dtype=None
                ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        
        siLu_w1 = SiLU(self.w1(x))
        w3x = self.w3(x)
        w2x = self.w2(siLu_w1 * w3x)
        
        return w2x.to(in_type)
        