"""Single-phase training loop.

Run from the project root:
    venv/bin/python src/train.py

Recipe knobs live on the chosen backbone's ``train_defaults``; CLI
flags override individual fields. RunConfig holds the process/IO
knobs that are independent of model and recipe.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
import constants
import stats
from datasets import build_train_dataset, build_val_dataset
from models import Backbone, ModelEMA, RotationModel, TrainDefaults
from utils.index import clear_cache
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Process / IO / reproducibility — independent of model & recipe."""
    num_workers: int = constants.NUM_WORKERS
    prefetch_factor: int = constants.PREFETCH_FACTOR
    seed: int = 42
    log_every: int = 50
    val_interval_epochs: int = 1


@dataclass
class ValidationResult:
    loss: float
    acc: float
    per_class: list[float]


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
) -> ValidationResult:
    model.train(False)
    correct = torch.zeros(constants.NUM_CLASSES, device=config.DEVICE)
    seen = torch.zeros(constants.NUM_CLASSES, device=config.DEVICE)
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()

    for imgs, labels in tqdm(
        loader,
        desc="validation",
        unit="batch",
        leave=False,
        file=sys.stdout,
    ):
        imgs = imgs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)
        with torch.autocast(device_type=config.DEVICE.type, dtype=torch.bfloat16):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        for c in range(constants.NUM_CLASSES):
            mask = labels == c
            seen[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

    return ValidationResult(
        loss=total_loss / max(1, total_n),
        acc=(correct.sum() / seen.sum().clamp(min=1)).item(),
        per_class=(correct / seen.clamp(min=1)).cpu().tolist(),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    extras: dict | None = None,
) -> None:
    payload: dict = {"model": model.state_dict(), "ema": ema.state_dict()}
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def cosine_with_warmup(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def set_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    scale: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def build_loaders(
    recipe: TrainDefaults,
    run: RunConfig,
) -> tuple[DataLoader, DataLoader]:
    logger.info("building datasets...")
    train_ds = build_train_dataset(
        config.DATASETS,
        input_size=recipe.input_size,
        resize_size=recipe.resize_size,
    )
    val_ds = build_val_dataset(
        config.DATASETS,
        input_size=recipe.input_size,
        resize_size=recipe.resize_size,
    )
    logger.info("train pairs: %d, val pairs: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=recipe.batch_size,
        shuffle=True,
        num_workers=run.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=run.prefetch_factor,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=run.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=run.prefetch_factor,
    )
    return train_loader, val_loader


def train_one_run(
    backbone: Backbone,
    recipe: TrainDefaults,
    run: RunConfig,
) -> None:
    torch.manual_seed(run.seed)
    train_loader, val_loader = build_loaders(recipe, run)

    model = RotationModel(
        backbone=backbone,
        pretrained=False,
    ).to(config.DEVICE)
    ema = ModelEMA(model, decay=recipe.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.lr,
        betas=(0.9, 0.999),
        weight_decay=recipe.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * recipe.epochs
    warmup_steps = steps_per_epoch * recipe.warmup_epochs
    logger.info(
        "%d epochs x %d steps = %d total (warmup %d steps)",
        recipe.epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    global_step = 0
    for epoch in range(recipe.epochs):
        model.train()
        epoch_start = time.monotonic()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        for imgs, labels in pbar:
            imgs = imgs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            scale = cosine_with_warmup(global_step, total_steps, warmup_steps)
            set_lr(optimizer, recipe.lr, scale)

            with torch.autocast(device_type=config.DEVICE.type, dtype=torch.bfloat16):
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.grad_clip)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_seen += imgs.size(0)

            lr_now = optimizer.param_groups[0]["lr"]
            acc = running_correct / max(1, running_seen)
            avg_loss = running_loss / max(1, running_seen)
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{acc:.3f}",
                lr=f"{lr_now:.2e}",
            )
            if global_step % run.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d loss=%.4f acc=%.3f lr=%.2e",
                    epoch, global_step, total_steps, avg_loss, acc, lr_now,
                )
            global_step += 1

        dt = time.monotonic() - epoch_start
        logger.info(
            "epoch %d done in %.0fs, throughput %.1f img/s",
            epoch, dt, running_seen / dt,
        )

        if (epoch + 1) % run.val_interval_epochs == 0:
            result = run_validation(ema.module, val_loader)
            logger.info(
                "validation epoch %d loss=%.4f acc=%.4f per_class=%s",
                epoch, result.loss, result.acc, result.per_class,
            )
            extras = {"epoch": epoch, "acc": result.acc}
        else:
            extras = {"epoch": epoch}

        ckpt = constants.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        save_checkpoint(ckpt, model, ema, extras=extras)
        logger.info("saved %s", ckpt)

    logger.info("done.")


def main() -> None:
    setup_logging()
    backbone = config.DEFAULT_BACKBONE
    defaults = backbone.train_defaults

    p = argparse.ArgumentParser(
        description="Train the rotation detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="number of full passes over the (4x-rotated) training set",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="images per training step (sized for the 5060 Ti shared with the desktop)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=RunConfig.num_workers,
        help="DataLoader worker processes for image decoding",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=RunConfig.prefetch_factor,
        help="batches each worker keeps queued ahead of the consumer (deeper queues mask HDD latency, use more RAM)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RunConfig.seed,
        help="torch RNG seed for the training loop",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index and per-dataset stats JSONs before running",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()
        stats.clear_cache()

    recipe = dataclasses.replace(
        defaults,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    logger.info("backbone=%s recipe=%s", backbone.name, recipe)
    run = RunConfig(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    train_one_run(backbone, recipe, run)


if __name__ == "__main__":
    main()
