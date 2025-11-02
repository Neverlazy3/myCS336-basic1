'''
Author: harmoon 92302745+Neverlazy3@users.noreply.github.com
Date: 2025-10-27 19:36:29
LastEditors: harmoon 92302745+Neverlazy3@users.noreply.github.com
LastEditTime: 2025-10-27 20:01:04
FilePath: \assignment1-basics-main\cs336_basics\mySoftMax.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    in_type = in_features.dtype
    in_features = in_features.to(torch.float64)
    
    max_value = in_features.max(dim=dim, keepdim=True).values
    in_features = in_features - max_value
    
    exp_in_features = torch.exp(in_features)
    exp_sum = exp_in_features.sum(dim=dim, keepdim=True)
    # exp_sum = torch.sum(exp_in_features, dim=dim, keepdim=True)
    
    ans = exp_in_features / exp_sum
    return ans.to(in_type)