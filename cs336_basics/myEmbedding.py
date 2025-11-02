import token
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.devcie = device
        self.dtype = dtype
        std = 1
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=self.devcie, dtype=self.dtype), requires_grad=False)
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        
        return self.weight[token_ids]