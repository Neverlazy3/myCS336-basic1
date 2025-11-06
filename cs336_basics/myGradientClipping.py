from collections.abc import Iterable
import math

import torch

def gradientClipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    params = [p for p in parameters if p.grad is not None]
    if params is None:
        return None

    total_sum = torch.zeros(1)
    for p in params:
        total_sum += p.grad.pow(2).sum()
    
    total_norm_sqrt = torch.sqrt(total_sum)

    if total_norm_sqrt > max_l2_norm:
        scale = max_l2_norm / (total_norm_sqrt + eps)
        for p in params:
            p.grad.mul_(scale)
