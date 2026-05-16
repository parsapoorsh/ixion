"""Learning-rate utilities for the training loop.

Kept separate from the loop itself so the schedule shape is easy to swap out
(cosine -> linear -> wsd, etc.) without touching trainer code.
"""

import math

import torch


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup: int,
) -> float:
    """Returns an LR scale in [0, 1]. Linear warmup, then half-cosine to zero."""
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def set_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: list[float],
    scale: float,
) -> None:
    """Apply the same scalar scale to each param group's base LR."""
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale
