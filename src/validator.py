#!/usr/bin/env python3
"""Standalone validation CLI for a single checkpoint.

Loads a checkpoint, builds the val dataset from ``--data-root``
at the model's preprocess sizes, and runs one validation pass. No
autocast, no async machinery, no train-loop coupling - point it at
any checkpoint with any device.

Reports group cross-entropy and product-oriented group4 accuracy (argmax
over the four synthetic rotations of each base image) from the same
forward pass - no extra model work.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
import constants
from datasets import build_dataset
from models import RotationModel
from stats import NormStats, load_stats_cache
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)

N_ROTS = len(constants.ROTATIONS)


def log_coverage_curve(margins: list[float], corrects: list[bool]) -> None:
    """Log accuracy and coverage at fixed abstention thresholds."""
    margin_t = torch.tensor(margins, dtype=torch.float64)
    ok_t = torch.tensor(corrects, dtype=torch.bool)
    total = len(margins)
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
            n / total,
            acc,
            n,
        )


@torch.no_grad()
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
    model.train(False)
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    # Carry logits across batch boundaries so batch_size need not be % 4.
    logit_buf: list[torch.Tensor] = []
    group4_correct = 0
    group4_total = 0
    margin_ok_sum = 0.0
    margin_bad_sum = 0.0
    margin_ok_n = 0
    margin_bad_n = 0
    margins: list[float] = []
    corrects: list[bool] = []

    def flush_groups() -> None:
        nonlocal group4_correct, group4_total, total_loss
        nonlocal margin_ok_sum, margin_bad_sum, margin_ok_n, margin_bad_n
        while len(logit_buf) >= N_ROTS:
            chunk = torch.stack(logit_buf[:N_ROTS])
            del logit_buf[:N_ROTS]
            # Training target: rot 0 is upright. Product pick: argmax logit.
            probs = chunk.softmax(dim=0)
            pred = int(probs.argmax().item())
            target = torch.zeros(1, dtype=torch.long, device=device)
            total_loss += float(loss_fn(chunk.unsqueeze(0), target).item())
            # top1 - top2 margin on four-view softmax probabilities.
            sorted_p, _ = probs.sort(descending=True)
            margin = float((sorted_p[0] - sorted_p[1]).item())
            ok = pred == 0
            margins.append(margin)
            corrects.append(ok)
            group4_total += 1
            if ok:
                group4_correct += 1
                margin_ok_sum += margin
                margin_ok_n += 1
            else:
                margin_bad_sum += margin
                margin_bad_n += 1

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = batch[0].to(device, non_blocking=True)
        logits = model(imgs)

        for logit in logits.detach().float().reshape(-1):
            logit_buf.append(logit)
        flush_groups()

        if batch_idx % constants.LOG_STEP == 0:
            g4 = group4_correct / max(1, group4_total)
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
        log_coverage_curve(margins, corrects)

    loss = total_loss / max(1, group4_total)
    group4_acc = group4_correct / max(1, group4_total)
    mean_margin_ok = margin_ok_sum / max(1, margin_ok_n)
    mean_margin_bad = margin_bad_sum / max(1, margin_bad_n)

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
    which_weights: str,
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
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info(
        "loaded %s (epoch=%s, ema_step=%s, weights=%s)",
        checkpoint_path,
        payload.get("epoch"),
        payload.get("ema_step"),
        which_weights,
    )

    state_key = {"ema": "ema", "raw": "model"}[which_weights]
    if state_key not in payload:
        raise KeyError(f"checkpoint missing '{state_key}' state dict; keys={list(payload)}")

    if "stats" in payload:
        raw_stats = payload["stats"]
        stats = NormStats(mean=tuple(raw_stats["mean"]), std=tuple(raw_stats["std"]))  # type: ignore[arg-type]
    else:
        cache_file = constants.INDEX_CACHE_DIR / "stats_train.json"
        cached = load_stats_cache(cache_file)
        if cached is None:
            raise RuntimeError(f"stats cache has no matching fingerprint: {cache_file}")
        stats = cached
        logger.info("loaded stats from cache file %s", cache_file)

    model = RotationModel(
        pretrained=False,
        pooling=payload.get("pooling", "avg"),
    ).to(device)
    model.load_state_dict(payload[state_key])

    val_ds, _ = build_dataset(
        split="val",
        datasets=config.build_datasets(data_root),
        device=device,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        stats=stats,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
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


def main(args) -> None:
    setup_logging(constants.LOGS_DIR)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Validate a single checkpoint against the val split under --data-root.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to an epoch_NNN.pt checkpoint to validate",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "root directory that contains the ImageNet 2017 and COCO 2017 "
            "layout (see config.build_datasets)"
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        required=True,
        help="torch device string (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="validation batch size",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="DataLoader worker processes for image decoding",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        required=True,
        help="batches each worker keeps queued ahead of the consumer",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="which weights to validate: EMA shadow (production) or raw training weights",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="stop after N batches (quick smoke test; default: full val set)",
    )
    args = p.parse_args()

    main(args)
