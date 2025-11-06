import torch
import numpy.typing as npt
import einops
def get_batch(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.as_tensor(dataset, dtype=torch.long, device=device)

    num = data.numel()
    batch = batch_size
    seq_len = context_length

    # 得到一维的随机可选起始位置后，通过[: batch]得到batch_size数量的起始位置
    starts = torch.randperm(num - seq_len, device=device)[: batch]
    offsets = einops.rearrange(torch.arange(seq_len + 1, device=device), 'n -> 1 n')
    position = einops.rearrange(starts, 'b -> b 1') + offsets
    
    tokens = data[position]

    return tokens[:, :-1], tokens[:, 1:]

