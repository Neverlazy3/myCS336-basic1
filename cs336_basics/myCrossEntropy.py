import torch
from torch import Tensor
from jaxtyping import Float, Int
from .mySoftMax import softmax

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], 
    targets: Int[Tensor, " batch_size"]
    ) -> Float[Tensor, ""]:
    input_type = inputs.dtype
    inputs = inputs.to(torch.float32)

    inputs_max = inputs.max(dim=-1, keepdim=True)[0]
    outputs_softmax = (inputs - inputs_max).exp()
    outputs_log = inputs - inputs_max - torch.log(outputs_softmax.sum(dim=-1, keepdim=True))

    batch_id = torch.arange(inputs.size(0), device=inputs.device)
    outputs = -outputs_log[batch_id, targets].mean()
    return outputs.to(input_type)
