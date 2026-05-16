"""Single-phase training loop for the rotation detector.

Run from the project root:
    venv/bin/python src/train.py

Knobs are at the top of main(); kept inline rather than as CLI flags so the
intended recipe is self-documenting. Override by editing or by importing
train_one_run() from another script.
"""

import argparse
import logging
import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from datasets import build_train_dataset, build_val_dataset
from ema import ModelEMA
from model import build_model

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


def cosine_with_warmup(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def set_lrs(optimizer: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.train(False)
    correct = torch.zeros(config.NUM_CLASSES, device=device)
    seen = torch.zeros(config.NUM_CLASSES, device=device)
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(imgs)
            loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        for c in range(config.NUM_CLASSES):
            mask = labels == c
            seen[c] += mask.sum()
            correct[c] += (preds[mask] == c).sum()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)
    per_class = (correct / seen.clamp(min=1)).cpu().tolist()
    overall = (correct.sum() / seen.sum().clamp(min=1)).item()
    return {
        "loss": total_loss / max(1, total_n),
        "acc": overall,
        "per_class": per_class,
    }


def train_one_run(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

    model = build_model(pretrained=True).to(device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    param_groups = model.param_groups(cfg.backbone_lr, cfg.head_lr, cfg.weight_decay)
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
        epoch_start = time.time()
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

        dt = time.time() - epoch_start
        log.info("epoch %d done in %.0fs, throughput %.1f img/s",
                 epoch, dt, running_seen / dt)

        if (epoch + 1) % cfg.val_every_epoch == 0:
            metrics = run_validation(ema.module, val_loader, device)
            log.info(
                "val ep %d loss=%.4f acc=%.4f per_class=%s",
                epoch, metrics["loss"], metrics["acc"], metrics["per_class"],
            )
            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                ckpt_path = config.CHECKPOINT_DIR / "best.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "epoch": epoch,
                        "acc": best_acc,
                    },
                    ckpt_path,
                )
                log.info("saved %s (acc=%.4f)", ckpt_path, best_acc)

    last_path = config.CHECKPOINT_DIR / "last.pt"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, last_path)
    log.info("done. best val acc = %.4f, last ckpt = %s", best_acc, last_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
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
