# '''
# Author: harmoon 92302745+Neverlazy3@users.noreply.github.com
# Date: 2025-10-24 16:46:34
# LastEditors: harmoon 92302745+Neverlazy3@users.noreply.github.com
# LastEditTime: 2025-10-25 10:44:49
# FilePath: \assignment1-basics-main\cs336_basics\myLinear.py
# Description: 

# Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
# '''
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=self.device, dtype=self.dtype))
        std = (2 /(in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0, std, a=(-3) * std, b = 3 * std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.matmul(x, self.weight.t())