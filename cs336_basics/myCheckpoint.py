import os
from pathlib import Path
import typing
import torch

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    check_point : dict[str, object] = {
        "model_state" : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "iteration_state" : iteration
    }
    
    if isinstance(out, (str, os.PathLike)):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(check_point, out)
    else:
        torch.save(check_point, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    chekc_point = torch.load(src)
    model.load_state_dict(chekc_point["model_state"])
    optimizer.load_state_dict(chekc_point["optimizer_state"])

    return chekc_point["iteration_state"]