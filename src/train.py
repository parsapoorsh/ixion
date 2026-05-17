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
from async_validator import AsyncValidator
from datasets import build_train_dataset
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


def build_train_loader(
    recipe: TrainDefaults,
    run: RunConfig,
) -> DataLoader:
    logger.info("building train dataset...")
    train_ds = build_train_dataset(
        config.DATASETS,
        input_size=recipe.input_size,
        resize_size=recipe.resize_size,
    )
    logger.info("train pairs: %d", len(train_ds))
    return DataLoader(
        train_ds,
        batch_size=recipe.batch_size,
        shuffle=True,
        num_workers=run.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=run.prefetch_factor,
        drop_last=True,
    )


def train_one_run(
    backbone: Backbone,
    recipe: TrainDefaults,
    run: RunConfig,
) -> None:
    torch.manual_seed(run.seed)
    train_loader = build_train_loader(recipe, run)

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
    mid_epoch_step = steps_per_epoch // 2
    logger.info(
        "%d epochs x %d steps = %d total (warmup %d steps)",
        recipe.epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    validator = AsyncValidator(
        backbone=backbone,
        recipe=recipe,
        dataset_specs=config.DATASETS,
        device_index=1,
    )

    def log_results() -> None:
        for r in validator.drain():
            logger.info(
                "validation epoch %d phase=%s loss=%.4f acc=%.4f per_class=%s (%.0fs)",
                r.epoch, r.phase, r.loss, r.acc, r.per_class, r.seconds,
            )

    try:
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
            for batch_idx, (imgs, labels) in enumerate(pbar):
                log_results()
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
                if batch_idx == mid_epoch_step:
                    validator.submit(ema.module, epoch, "mid")
                global_step += 1

            dt = time.monotonic() - epoch_start
            logger.info(
                "epoch %d done in %.0fs, throughput %.1f img/s",
                epoch, dt, running_seen / dt,
            )
            validator.submit(ema.module, epoch, "end")

            ckpt = constants.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
            save_checkpoint(ckpt, model, ema, extras={"epoch": epoch})
            logger.info("saved %s", ckpt)

        log_results()
        logger.info("done.")
    finally:
        validator.shutdown()
        log_results()


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
