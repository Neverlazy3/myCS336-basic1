import math

def cosine_annealing_learning_rate(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    if warmup_iters <= it and it <= cosine_cycle_iters:
        cosNum = math.cos(math.pi * ((it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + 0.5 * (1 + cosNum) * (max_learning_rate - min_learning_rate)