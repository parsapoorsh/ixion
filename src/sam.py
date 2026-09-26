"""Sharpness-Aware Minimization (SAM).

SAM seeks parameters that sit in flat loss regions rather than sharp ones,
which tends to generalize better - useful for the out-of-distribution tail of
the orientation task. Each step takes two passes: ``first_step`` ascends to
the worst-case point inside a rho-radius ball (w -> w + e(w)), then the
gradient is recomputed there and applied to the original weights by
``second_step``. Costs two forward/backward passes per step.

Wraps any base optimizer (AdamW here); the base optimizer does the actual
parameter update in ``second_step``. The two helpers toggle BatchNorm running
stats so the extra forward pass doesn't update them twice per step.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization" (ICLR 2021).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable,
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = {"rho": rho, "adaptive": adaptive, **kwargs}
        super().__init__(params, defaults)
        # The base optimizer shares SAM's param groups, so per-group lr edits
        # (the cosine schedule sets group["lr"]) reach the real update step.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                weight = torch.pow(p, 2) if group["adaptive"] else 1.0
                e_w = weight * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local worst case
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p].pop("e_w"))  # back to the original weights
        self.base_optimizer.step()  # real update with the worst-case gradient
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                weight = torch.abs(p) if group["adaptive"] else 1.0
                norms.append((weight * p.grad).norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)

    def step(self, closure=None):  # noqa: ARG002 - SAM uses explicit two-step API
        raise RuntimeError(
            "SAM has no single-call step(); use first_step() and second_step() around two forward/backward passes."
        )

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def disable_running_stats(model: torch.nn.Module) -> None:
    """Freeze BatchNorm running stats (momentum -> 0) for the second pass."""
    for module in model.modules():
        if hasattr(module, "running_mean") and getattr(module, "momentum", None) is not None:
            module.backup_momentum = module.momentum
            module.momentum = 0.0  # type: ignore[assignment]


def enable_running_stats(model: torch.nn.Module) -> None:
    """Restore BatchNorm momentum saved by ``disable_running_stats``."""
    for module in model.modules():
        if hasattr(module, "running_mean") and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
