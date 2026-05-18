"""LR range test (Smith 2017).

Ramps the learning rate exponentially from ``lr_min`` to ``lr_max`` over
``num_steps`` batches, tracks EMA-smoothed loss, and returns the LR at
the steepest descent on the (log10 lr, loss) curve. Also writes a plot.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch
from matplotlib.figure import Figure
from torch import nn
from tqdm.auto import tqdm

import constants
from models import RotationModel
from train import RunConfig, build_train_loader

logger = logging.getLogger(__name__)


def find_lr(
    run: RunConfig,
    num_steps: int = 500,
    lr_min: float = 1e-7,
    lr_max: float = 10.0,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    smooth_beta: float = 0.05,
    divergence_factor: float = 4.0,
    out_path: Path = constants.CHECKPOINT_DIR / "lr_finder.png",
) -> float:
    torch.manual_seed(run.seed)
    train_loader = build_train_loader(run)
    model = RotationModel(pretrained=False).to(run.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr_min,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    mult = (lr_max / lr_min) ** (1.0 / max(1, num_steps - 1))
    lrs: list[float] = []
    losses: list[float] = []
    smoothed = 0.0
    best = float("inf")

    model.train()
    pbar = tqdm(
        total=num_steps,
        desc="lr-find",
        unit="step",
        file=sys.stdout,
    )
    step = 0
    for imgs, labels in train_loader:
        if step >= num_steps:
            break
        imgs = imgs.to(run.device, non_blocking=True)
        labels = labels.to(run.device, non_blocking=True)
        lr_now = lr_min * (mult ** step)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        with torch.autocast(device_type=run.device.type, dtype=torch.bfloat16):  # pyright: ignore[reportPrivateImportUsage]
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        smoothed = smooth_beta * loss_val + (1.0 - smooth_beta) * smoothed
        # bias correction so the first few steps aren't pulled toward zero
        corrected = smoothed / (1.0 - (1.0 - smooth_beta) ** (step + 1))
        if corrected < best:
            best = corrected
        if step > 10 and corrected > divergence_factor * best:
            logger.info(
                "diverged at step %d (lr=%.2e, loss=%.2f)",
                step, lr_now, corrected,
            )
            break

        lrs.append(lr_now)
        losses.append(corrected)
        pbar.update(1)
        pbar.set_postfix(lr=f"{lr_now:.2e}", loss=f"{corrected:.3f}")
        step += 1
    pbar.close()

    recommended = pick_steepest(lrs, losses)
    plot_curve(lrs, losses, recommended, out_path)
    logger.info("saved %s", out_path)
    return recommended


def pick_steepest(lrs: list[float], losses: list[float]) -> float:
    """LR at the most negative slope on the (log10 lr, loss) curve."""
    log_lrs = [math.log10(x) for x in lrs]
    steepest_idx = 0
    steepest_slope = 0.0
    for i in range(1, len(losses)):
        slope = (losses[i] - losses[i - 1]) / (log_lrs[i] - log_lrs[i - 1])
        if slope < steepest_slope:
            steepest_slope = slope
            steepest_idx = i
    return lrs[steepest_idx]


def plot_curve(
    lrs: list[float],
    losses: list[float],
    recommended: float,
    out_path: Path,
) -> None:
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.plot(lrs, losses, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed loss")
    ax.axvline(
        recommended,
        color="tab:red",
        linestyle="--",
        label=f"steepest descent: lr={recommended:.2e}",
    )
    ax.legend()
    ax.set_title(f"LR range test - {len(lrs)} steps")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
