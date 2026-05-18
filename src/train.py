"""Single-phase training loop.

Run from the project root:
    venv/bin/python src/train.py --device cuda --batch-size N \\
        --num-workers N --prefetch-factor N

Optimisation knobs are plain kwargs on ``train_one_run`` with the
project's defaults baked in. ``RunConfig`` carries the process/IO
knobs. Validation is a separate process - see ``validator.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
import constants
import stats
from datasets import build_train_dataset
from models import ModelEMA, RotationModel
from utils.index import clear_cache
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Process / IO / reproducibility - independent of recipe."""
    device: torch.device
    batch_size: int
    num_workers: int
    prefetch_factor: int
    seed: int
    log_every: int = constants.LOG_STEP


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    extras: dict | None = None,
) -> None:
    payload: dict = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "ema_step": ema.step,
        "optimizer": optimizer.state_dict(),
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup: int,
) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def build_train_loader(run: RunConfig) -> DataLoader:
    logger.info("building train dataset...")
    train_ds = build_train_dataset(
        config.DATASETS,
        input_size=constants.INPUT_SIZE,
        resize_size=constants.RESIZE_SIZE,
        num_workers=run.num_workers,
        device=run.device,
    )
    logger.info("train pairs: %d", len(train_ds))
    return DataLoader(
        train_ds,
        batch_size=run.batch_size,
        shuffle=True,
        num_workers=run.num_workers,
        pin_memory=run.device.type == "cuda",
        persistent_workers=run.num_workers > 0,
        prefetch_factor=run.prefetch_factor if run.num_workers > 0 else None,
        drop_last=True,
    )


def train_one_run(
    run: RunConfig,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    warmup_epochs: int = 5,
    grad_clip: float = 1.0,
    ema_decay: float = 0.9999,
    resume_from: Path | None = None,
) -> None:
    torch.manual_seed(run.seed)
    train_loader = build_train_loader(run)

    model = RotationModel(pretrained=False).to(run.device)
    ema = ModelEMA(model, decay=ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    logger.info(
        "%d epochs x %d steps = %d total (warmup %d steps)",
        epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    start_epoch = 0
    if resume_from is not None:
        payload = torch.load(resume_from, map_location=run.device, weights_only=False)
        model.load_state_dict(payload["model"])
        ema.module.load_state_dict(payload["ema"])
        ema.step = int(payload.get("ema_step", 0))
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        logger.info(
            "resumed from %s (next epoch=%d, ema_step=%d)",
            resume_from, start_epoch, ema.step,
        )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
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
            imgs = imgs.to(run.device, non_blocking=True)
            labels = labels.to(run.device, non_blocking=True)

            scale = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_steps,
            )

            # set lr
            for group in optimizer.param_groups:
                group["lr"] = lr * scale

            with torch.autocast(device_type=run.device.type, dtype=torch.bfloat16):  # pyright: ignore[reportPrivateImportUsage]
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

        ckpt = constants.CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt"
        save_checkpoint(ckpt, model, ema, optimizer, extras={"epoch": epoch})
        logger.info("saved %s", ckpt)

    logger.info("done.")


def main() -> None:
    setup_logging()

    p = argparse.ArgumentParser(
        description="Train the rotation detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="images per training step",
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
        "--seed",
        type=int,
        required=True,
        help="torch RNG seed for the training loop",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index and per-dataset stats JSONs before running",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to an epoch_NNN.pt checkpoint to resume from (model, ema, ema_step restored; optimizer re-warms)",
    )
    args = p.parse_args()

    if args.invalidate_cache:
        clear_cache()
        stats.clear_cache()

    run = RunConfig(
        device=torch.device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    train_one_run(run, resume_from=args.resume)


if __name__ == "__main__":
    main()
