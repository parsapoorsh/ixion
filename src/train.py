#!/usr/bin/env python3
"""Single-phase training loop.

Run from the project root:
    venv/bin/python src/train.py --device cuda --batch-size N \\
        --num-workers N --prefetch-factor N

Each view is a binary upright/not example (BCEWithLogitsLoss). Four-way
ranking is eval/infer only - see ``validator.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import shutil
import sys
import sysconfig
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
import constants
from datasets import build_dataset
from models import ModelEMA, RotationModel
from sam import SAM, disable_running_stats, enable_running_stats
from stats import NormStats
from utils.logsetup import setup_logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EpochStats:
    loss_sum: float = 0.0
    correct: int = 0
    seen: int = 0
    grad_norm_sum: float = 0.0
    grad_steps: int = 0

    def merge(
        self,
        loss_sum: float,
        correct: int,
        grad_norm_sum: float,
        n: int,
        steps: int,
    ) -> None:
        self.loss_sum += loss_sum
        self.correct += correct
        self.seen += n
        self.grad_norm_sum += grad_norm_sum
        self.grad_steps += steps

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(1, self.seen)

    @property
    def acc(self) -> float:
        return self.correct / max(1, self.seen)

    @property
    def avg_grad_norm(self) -> float:
        return self.grad_norm_sum / max(1, self.grad_steps)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    extras: dict | None = None,
) -> None:
    # torch.compile() wraps the module and prefixes state_dict keys with
    # "_orig_mod." - unwrap so checkpoints stay portable across compile.
    raw_model = getattr(model, "_orig_mod", model)
    # Under SAM the real state lives in the wrapped base optimizer; save that
    # so the on-disk state is plain AdamW and resumes with or without --sam.
    base_optimizer = getattr(optimizer, "base_optimizer", optimizer)
    payload: dict = {
        "model": raw_model.state_dict(),
        "ema": ema.state_dict(),
        "ema_step": ema.step,
        "optimizer": base_optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
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
    t = min(1.0, t)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def python_headers_available() -> bool:
    for key in ("include", "platinclude"):
        include_dir = sysconfig.get_path(key)
        if include_dir is not None and (Path(include_dir) / "Python.h").exists():
            return True
    return False


def python_headers_package_hint() -> str:
    version = sysconfig.get_python_version()
    return f"sudo apt install python{version}-dev"


def can_compile_model(device: torch.device) -> tuple[bool, str | None]:
    if shutil.which("gcc") is None:
        return False, "gcc is not available; install it with sudo apt install build-essential"
    if not python_headers_available():
        return False, (
            f"Python.h is not available under {sysconfig.get_path('include')}; "
            f"install Python headers with {python_headers_package_hint()}"
        )
    return True, None


def setup_model(
    device: torch.device,
    seed: int,
    lr: float,
    weight_decay: float,
    ema_decay: float,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.1,
    pooling: str = "avg",
    use_sam: bool = False,
    sam_rho: float = 0.05,
    compile_model: bool = True,
    resume_from: Path | None = None,
) -> tuple[
    nn.Module,
    ModelEMA,
    torch.optim.Optimizer,
    nn.Module,
    int,
    int | None,
    int | None,
]:
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # Peek at the checkpoint before building the model so resumes honour the
    # pooling variant it was trained with (attention vs global average).
    resume_payload = None
    resume_pooling = pooling
    if resume_from is not None:
        resume_payload = torch.load(resume_from, map_location=device, weights_only=False)
        resume_pooling = str(resume_payload.get("pooling", pooling))

    model = RotationModel(
        pretrained=False,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pooling=resume_pooling,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)  # type: ignore[call-arg]
    ema = ModelEMA(model, decay=ema_decay)

    if use_sam:
        optimizer: torch.optim.Optimizer = SAM(
            model.parameters(),
            base_optimizer=torch.optim.AdamW,
            rho=sam_rho,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        logger.info("using SAM (rho=%.3f) over AdamW", sam_rho)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
    # Inference picks the rotation with the highest upright logit
    # (argmax over 4 views), so the absolute decision threshold
    # doesn't matter - only the relative ranking.
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    schedule_total_steps = None
    schedule_warmup_steps = None
    if resume_from is not None:
        payload = resume_payload
        model.load_state_dict(payload["model"])

        ema.module.load_state_dict(payload["ema"])
        ema.step = int(payload.get("ema_step", 0))

        if "optimizer" in payload:
            # Saved state is plain base-optimizer (AdamW) state; load it into
            # the base so resume works whether or not SAM wraps it this run.
            base_optimizer = getattr(optimizer, "base_optimizer", optimizer)
            base_optimizer.load_state_dict(payload["optimizer"])

        if "rng_cpu" in payload:
            torch.set_rng_state(payload["rng_cpu"].cpu())

        if payload.get("rng_cuda") is not None and torch.cuda.is_available():
            # map_location=device pulls the saved ByteTensors to CUDA, but
            # set_rng_state_all requires CPU ByteTensors - move them back.
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in payload["rng_cuda"]]
            )

        start_epoch = int(payload["epoch"]) + 1
        if "schedule_total_steps" in payload:
            schedule_total_steps = int(payload["schedule_total_steps"])
        if "schedule_warmup_steps" in payload:
            schedule_warmup_steps = int(payload["schedule_warmup_steps"])
        logger.info(
            "resumed from %s (next epoch=%d, ema_step=%d)",
            resume_from,
            start_epoch,
            ema.step,
        )

    # Compile after any state_dict load so checkpoints are saved/loaded in
    # uncompiled form (no "_orig_mod." prefix in the on-disk dict).
    if compile_model:
        can_compile, reason = can_compile_model(device)
        if can_compile:
            model = torch.compile(model)
            logger.info("enabled torch.compile")
        else:
            logger.warning("disabled torch.compile: %s", reason)
    else:
        logger.info("disabled torch.compile by request")

    return (
        model,
        ema,
        optimizer,
        loss_fn,
        start_epoch,
        schedule_total_steps,
        schedule_warmup_steps,
    )


def train(
    device: torch.device,
    train_loader: DataLoader,
    checkpoint_dir: Path,
    seed: int,
    stats: NormStats,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.1,
    pooling: str = "avg",
    warmup_epochs: int = 5,
    grad_clip: float = 1.0,
    ema_decay: float = 0.9999,
    use_sam: bool = False,
    sam_rho: float = 0.05,
    compile_model: bool = True,
    resume_from: Path | None = None,
) -> None:
    (
        model,
        ema,
        optimizer,
        loss_fn,
        start_epoch,
        saved_schedule_total_steps,
        saved_schedule_warmup_steps,
    ) = setup_model(
        device=device,
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pooling=pooling,
        use_sam=use_sam,
        sam_rho=sam_rho,
        compile_model=compile_model,
        resume_from=resume_from,
    )

    steps_per_epoch = len(train_loader)
    requested_total_steps = steps_per_epoch * epochs
    total_steps = (
        saved_schedule_total_steps
        if saved_schedule_total_steps is not None
        else requested_total_steps
    )
    requested_warmup_steps = steps_per_epoch * warmup_epochs
    warmup_steps = (
        saved_schedule_warmup_steps
        if saved_schedule_warmup_steps is not None
        else requested_warmup_steps
    )
    logger.info(
        "%d epochs x %d steps, schedule total %d (warmup %d steps)",
        epochs,
        steps_per_epoch,
        total_steps,
        warmup_steps,
    )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.monotonic()
        epoch_stats = EpochStats()

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        # GPU-side accumulators flushed to epoch_stats at LOG_STEP boundaries
        # and once at epoch end, so .item() doesn't sync every step.
        loss_buf = torch.zeros((), device=device)
        correct_buf = torch.zeros((), device=device, dtype=torch.long)
        gn_buf = torch.zeros((), device=device)
        pending_n = 0
        pending_steps = 0

        for imgs, labels in pbar:
            imgs = imgs.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            labels = labels.to(device, non_blocking=True).float()

            scale = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_steps,
            )

            # set lr
            for group in optimizer.param_groups:
                group["lr"] = lr * scale

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):  # pyright: ignore[reportPrivateImportUsage]
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if isinstance(optimizer, SAM):
                # First pass climbs to the worst-case point; the second pass
                # recomputes the gradient there and applies the real update.
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss_adv = loss_fn(model(imgs), labels)
                loss_adv.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                optimizer.second_step(zero_grad=True)
                enable_running_stats(model)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                optimizer.step()
            ema.update(model)

            preds = (logits > 0).float()
            n = imgs.size(0)
            loss_buf += loss.detach() * n
            correct_buf += (preds == labels).sum()
            gn_buf += grad_norm.detach()
            pending_n += n
            pending_steps += 1

            if global_step % constants.LOG_STEP == 0:
                window_loss = loss_buf.item() / max(1, pending_n)
                window_acc = int(correct_buf.item()) / max(1, pending_n)
                window_gn = gn_buf.item() / max(1, pending_steps)
                epoch_stats.merge(
                    loss_sum=loss_buf.item(),
                    correct=int(correct_buf.item()),
                    grad_norm_sum=gn_buf.item(),
                    n=pending_n,
                    steps=pending_steps,
                )
                loss_buf.zero_()
                correct_buf.zero_()
                gn_buf.zero_()
                pending_n = 0
                pending_steps = 0

                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    acc=f"{window_acc:.5f}",
                    loss=f"{window_loss:.5f}",
                    lr=f"{lr_now:.2e}",
                    gn=f"{window_gn:.5f}",
                )
                logger.info(
                    "epoch %d step %d/%d loss=%.5f acc=%.5f lr=%.2e grad_norm=%.5f",
                    epoch, global_step, total_steps,
                    window_loss, window_acc, lr_now, window_gn,
                )
            global_step += 1

        if pending_steps > 0:
            epoch_stats.merge(
                loss_sum=loss_buf.item(),
                correct=int(correct_buf.item()),
                grad_norm_sum=gn_buf.item(),
                n=pending_n,
                steps=pending_steps,
            )

        dt = time.monotonic() - epoch_start
        logger.info(
            "epoch %d done in %.0fs, throughput %.1f img/s",
            epoch,
            dt,
            epoch_stats.seen / dt,
        )

        ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt,
            model,
            ema,
            optimizer,
            extras={
                "epoch": epoch,
                "schedule_total_steps": total_steps,
                "schedule_warmup_steps": warmup_steps,
                "pooling": pooling,
                "stats": {"mean": list(stats.mean), "std": list(stats.std)},
            },
        )
        logger.info("saved %s", ckpt)

    logger.info("done.")


def main(args) -> None:
    setup_logging(constants.LOGS_DIR)

    device = torch.device(args.device)

    ds, stats = build_dataset(
        split="train",
        datasets=config.DATASETS,
        device=device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch_factor,
        invalidate_cache=args.invalidate_cache,
        erasing_p=args.erasing_p,
        crop_policy=args.crop_policy,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        shuffle=True,
    )

    train(
        device=device,
        train_loader=dl,
        seed=args.seed,
        checkpoint_dir=constants.CHECKPOINT_DIR,
        stats=stats,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path,
        pooling=args.pooling,
        use_sam=args.sam,
        sam_rho=args.sam_rho,
        compile_model=not args.no_compile,
        resume_from=args.resume,
    )


if __name__ == "__main__":
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
        "--epochs",
        type=int,
        default=30,
        help="total number of training epochs",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="head dropout rate",
    )
    p.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        help="backbone stochastic depth rate",
    )
    p.add_argument(
        "--erasing-p",
        type=float,
        default=0.25,
        help="random erasing probability (0 disables)",
    )
    p.add_argument(
        "--crop-policy",
        choices=["random", "center", "none"],
        default="random",
        help="train spatial preprocessing: resize+random crop, resize+center crop, or direct resize",
    )
    p.add_argument(
        "--pooling",
        choices=["attention", "avg"],
        default="avg",
        help="how to aggregate backbone features before the head",
    )
    p.add_argument(
        "--sam",
        action="store_true",
        help="use Sharpness-Aware Minimization over AdamW (2x passes/step, flatter minima)",
    )
    p.add_argument(
        "--sam-rho",
        type=float,
        default=0.05,
        help="SAM neighbourhood radius for the worst-case ascent step",
    )
    p.add_argument(
        "--invalidate-cache",
        action="store_true",
        help="delete cached file-index and per-dataset stats JSONs before running",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile even when build dependencies are available",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="path to an epoch_NNN.pt checkpoint to resume from "
             "(model, EMA, optimizer, and saved schedule restored)",
    )
    args = p.parse_args()

    main(args)
