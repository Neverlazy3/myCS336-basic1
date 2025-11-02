from typing import Iterable
from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn as nn

class transModuleList(nn.ModuleList):
    def __init__(self, modules=()) -> None:
        super().__init__(modules)
        
    def forward(self, x: Tensor, *args, **kwargs) -> Float[Tensor, "..."]:
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x