"""Exponential moving average wrapper for model weights.

Tracks a shadow copy of the model's parameters and buffers, updated
each training step with ``ema = decay * ema + (1 - decay) * param``.
Evaluated and checkpointed separately from the raw weights - EMA
smooths out late-training noise and typically validates a touch
better than the running model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn


class ModelEMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_denom: float = 1000.0,
    ) -> None:
        self.decay = decay
        self.warmup_denom = warmup_denom
        self.step = 0
        self.module = deepcopy(model)
        self.module.train(False)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Bias-corrected warm-up: the shadow starts as a deepcopy of the
        # random-init model, so a constant high decay leaves the EMA
        # weighted heavily toward noise for thousands of steps. Clamp the
        # effective decay so it ramps from ~0 up to self.decay.
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (self.warmup_denom + self.step))
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.module.load_state_dict(sd)
