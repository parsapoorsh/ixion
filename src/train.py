"""Single-phase training loop.

Run from the project root:
    venv/bin/python src/train.py

Knobs are on TrainConfig; defaults match the recipe in the project plan. Most
of the moving pieces (eval, schedule, checkpointing, logging) live in their
own modules so this file just orchestrates them.
"""

import argparse
import logging
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from builders import build_train_dataset, build_val_dataset
from checkpoint import save_checkpoint
from ema import ModelEMA
from evaluate import run_validation
from logsetup import setup_logging
from model import build_model
from schedule import cosine_with_warmup, set_lrs

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 128
    num_workers: int = 12
    backbone_lr: float = 2e-4
    head_lr: float = 2e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    log_every: int = 50
    val_every_epoch: int = 1
    seed: int = 42


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    log.info("building datasets...")
    train_ds = build_train_dataset(config.DATASETS)
    val_ds = build_val_dataset(config.DATASETS)
    log.info("train pairs: %d, val pairs: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


def train_one_run(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(pretrained=True).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    param_groups = model.param_groups(
        backbone_lr=cfg.backbone_lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    base_lrs = [g["lr"] for g in param_groups]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    log.info("%d epochs x %d steps = %d total", cfg.epochs, steps_per_epoch, total_steps)

    global_step = 0
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.monotonic()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            scale = cosine_with_warmup(global_step, total_steps, cfg.warmup_steps)
            set_lrs(optimizer, base_lrs, scale)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_seen += imgs.size(0)

            if global_step % cfg.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                acc = running_correct / max(1, running_seen)
                avg_loss = running_loss / max(1, running_seen)
                log.info(
                    "ep %d step %d/%d loss=%.4f acc=%.3f lr=%.2e",
                    epoch, global_step, total_steps, avg_loss, acc, lr_now,
                )
            global_step += 1

        dt = time.monotonic() - epoch_start
        log.info(
            "epoch %d done in %.0fs, throughput %.1f img/s",
            epoch, dt, running_seen / dt,
        )

        if (epoch + 1) % cfg.val_every_epoch == 0:
            result = run_validation(ema.module, val_loader, device)
            log.info(
                "val ep %d loss=%.4f acc=%.4f per_class=%s",
                epoch, result.loss, result.acc, result.per_class,
            )
            if result.acc > best_acc:
                best_acc = result.acc
                ckpt = config.CHECKPOINT_DIR / "best.pt"
                save_checkpoint(ckpt, model, ema, extras={"epoch": epoch, "acc": best_acc})
                log.info("saved %s (acc=%.4f)", ckpt, best_acc)

    last_path = config.CHECKPOINT_DIR / "last.pt"
    save_checkpoint(last_path, model, ema)
    log.info("done. best val acc = %.4f, last ckpt = %s", best_acc, last_path)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    args = p.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_one_run(cfg)


if __name__ == "__main__":
    main()
