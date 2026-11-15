"""Group-4 validation and the `validate` command."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

import constants
from checkpoint import WeightName, load_checkpoint_model
from config import build_dataset_specs
from data import build_dataset, build_loader

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

N_ROTS = len(constants.ROTATIONS)

def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, float, float, int]:
    """Return group CE, group4 accuracy, margin means, and group count.

    ``RotationPairDataset`` is ordered as base0-rot0..3,
    base1-rot0..3, and so on when ``shuffle=False``.
    """
    model.eval()
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    # Carry logits across batch boundaries so batch_size need not be % 4.
    logit_buf: list[torch.Tensor] = []
    margins: list[float] = []
    corrects: list[bool] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = batch[0].to(device, non_blocking=True)
        logits = model(imgs)

        for logit in logits.detach().float().reshape(-1):
            logit_buf.append(logit)
            if len(logit_buf) < N_ROTS:
                continue
            chunk = torch.stack(logit_buf)
            logit_buf.clear()
            # Training target: rot 0 is upright. Product pick: argmax logit.
            probs = chunk.softmax(dim=0)
            target = torch.zeros(1, dtype=torch.long, device=device)
            total_loss += float(loss_fn(chunk.unsqueeze(0), target).item())
            # top1 - top2 margin on four-view softmax probabilities.
            sorted_p, _ = probs.sort(descending=True)
            margin = float((sorted_p[0] - sorted_p[1]).item())
            ok = int(probs.argmax().item()) == 0
            margins.append(margin)
            corrects.append(ok)

        if batch_idx % constants.LOG_STEP == 0:
            g4 = sum(corrects) / max(1, len(corrects))
            logger.info(
                "validation batch %d/%d group4_acc=%.4f",
                batch_idx,
                n_batches,
                g4,
            )

    if logit_buf:
        logger.warning(
            "dropped %d trailing val samples not forming a full rotation group",
            len(logit_buf),
        )

    if margins:
        margin_t = torch.tensor(margins, dtype=torch.float64)
        ok_t = torch.tensor(corrects, dtype=torch.bool)
        logger.info("margin threshold -> coverage / accuracy:")
        for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            kept = margin_t >= threshold
            n = int(kept.sum())
            if n == 0:
                continue
            acc = float(ok_t[kept].float().mean())
            logger.info(
                "  margin>=%.2f  coverage=%.3f  acc=%.4f  (n=%d)",
                threshold,
                n / len(margins),
                acc,
                n,
            )

    group4_total = len(corrects)
    loss = total_loss / max(1, group4_total)
    group4_acc = sum(corrects) / max(1, group4_total)
    margin_ok = [m for m, ok in zip(margins, corrects, strict=True) if ok]
    margin_bad = [m for m, ok in zip(margins, corrects, strict=True) if not ok]
    mean_margin_ok = sum(margin_ok) / max(1, len(margin_ok))
    mean_margin_bad = sum(margin_bad) / max(1, len(margin_bad))

    return (
        loss,
        group4_acc,
        mean_margin_ok,
        mean_margin_bad,
        group4_total,
    )


def validate_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    device: torch.device,
    which_weights: WeightName,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_batches: int | None,
) -> None:
    """Load a checkpoint's weights and run validation in-process.

    which_weights: "ema" or "raw" - pick which set of weights from the
    checkpoint to load. Helpful for separating "EMA shadow is bad" from
    "validator code is bad".
    """
    _, model, mean_std = load_checkpoint_model(
        checkpoint_path,
        device,
        weights=which_weights,
    )
    val_ds, _ = build_dataset(
        split="val",
        datasets=build_dataset_specs(data_root),
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        mean_std=mean_std,
    )
    val_loader = build_loader(
        val_ds,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )
    logger.info(
        "val pairs: %d, device=%s, batch_size=%d, workers=%d, prefetch=%d, max_batches=%s",
        len(val_ds),
        device,
        batch_size,
        num_workers,
        prefetch_factor,
        max_batches,
    )

    t0 = time.monotonic()
    (
        loss,
        group4_acc,
        mean_margin_ok,
        mean_margin_bad,
        group4_n,
    ) = run_validation(
        model,
        val_loader,
        device=device,
        max_batches=max_batches,
    )
    dt = time.monotonic() - t0
    logger.info(
        "checkpoint %s -> group_ce=%.4f group4_acc=%.4f mean_margin_ok=%.4f mean_margin_bad=%.4f group4_n=%d in %.0fs",
        checkpoint_path.name,
        loss,
        group4_acc,
        mean_margin_ok,
        mean_margin_bad,
        group4_n,
        dt,
    )


def run_validate(args: argparse.Namespace) -> None:
    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=torch.device(args.device),
        which_weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_batches=args.max_batches,
    )
