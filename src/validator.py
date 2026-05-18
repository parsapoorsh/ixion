"""Standalone validation CLI for a single checkpoint.

Loads a checkpoint, builds the val dataset from ``config.DATASETS`` at
the configured backbone's preprocess sizes, and runs one validation
pass. No autocast, no async machinery, no train-loop coupling - point
it at any checkpoint with any device.
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
from datasets import build_val_dataset
from models import RotationModel
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)

VALIDATION_LOG_EVERY = 50


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, list[float]]:
    model.train(False)
    correct = torch.zeros(constants.NUM_CLASSES, device=device)
    seen = torch.zeros(constants.NUM_CLASSES, device=device)
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()
    n_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    for batch_idx, (imgs, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        for c in range(constants.NUM_CLASSES):
            mask = labels == c
            seen[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)
        if batch_idx % VALIDATION_LOG_EVERY == 0:
            logger.info(
                "validation batch %d/%d running_acc=%.4f",
                batch_idx, n_batches,
                (correct.sum() / seen.sum().clamp(min=1)).item(),
            )

    return (
        total_loss / max(1, total_n),
        (correct.sum() / seen.sum().clamp(min=1)).item(),
        (correct / seen.clamp(min=1)).cpu().tolist(),
    )


def validate_checkpoint(
    checkpoint_path: Path,
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
    backbone = config.DEFAULT_BACKBONE
    recipe = backbone.train_defaults

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
        raise KeyError(
            f"checkpoint missing '{state_key}' state dict; keys={list(payload)}"
        )

    model = RotationModel(
        backbone=backbone,
        pretrained=False,
    ).to(device)
    model.load_state_dict(payload[state_key])

    val_ds = build_val_dataset(
        config.DATASETS,
        input_size=recipe.input_size,
        resize_size=recipe.resize_size,
        num_workers=num_workers,
        device=device,
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
        len(val_ds), device, batch_size, num_workers, prefetch_factor, max_batches,
    )

    t0 = time.monotonic()
    loss, acc, per_class = run_validation(
        model,
        val_loader,
        device=device,
        max_batches=max_batches,
    )
    dt = time.monotonic() - t0
    logger.info(
        "checkpoint %s -> loss=%.4f acc=%.4f per_class=%s in %.0fs",
        checkpoint_path.name, loss, acc, per_class, dt,
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Validate a single checkpoint against config.DATASETS.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to an epoch_NNN.pt checkpoint to validate",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="torch device string (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="validation batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="DataLoader worker processes for image decoding",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        required=True,
        help="batches each worker keeps queued ahead of the consumer",
    )
    parser.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="which weights to validate: EMA shadow (production) or raw training weights",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="stop after N batches (quick smoke test; default: full val set)",
    )
    args = parser.parse_args()
    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        device=torch.device(args.device),
        which_weights=args.weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
