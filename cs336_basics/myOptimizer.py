import torch
from collections.abc import Callable, Iterable
from typing import Dict, Optional
import math 

class mySGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3) -> None:
        if lr < 0 :
            raise ValueError(f"Invaild lr :{lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
    
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = mySGD([weights], lr=1e2)
# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step()

class myAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"] = 0
                
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                
                # m = betas[0] * m + (1 - betas[0]) * grad
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0]) # add_,mul_都是在原数据的基础上进行操作，能够节省空间，它们跟不带_的函数不一样！
                # v = betas[1] * v + (1 - betas[1]) * grad * grad
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1]) # 这里忘记加_导致出现错误！编写代码一定要注意函数选取是否正确

                bias1 = 1 - betas[1]**t
                bias2 = 1 - betas[0]**t
                adjusted_lr_t = lr * (math.sqrt(bias1)) / (bias2)


                p.data.add_(m / (v.sqrt() + eps), alpha=-adjusted_lr_t)
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


