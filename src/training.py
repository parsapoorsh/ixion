"""Training loop and `train` / `finetune` command."""

from __future__ import annotations

import logging
import math
import shutil
import sys
import sysconfig
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import constants
from checkpoint import save_checkpoint
from config import build_dataset_specs
from data import build_dataset, build_loader
from model import SAM, ModelEMA, RotationModel

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

def move_batch(
    imgs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a training batch to *device* with channels-last images."""
    return (
        imgs.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        ),
        labels.to(device, non_blocking=True).float(),
    )


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


def can_compile_model() -> tuple[bool, str | None]:
    if shutil.which("gcc") is None:
        return False, "gcc is not available; install it with sudo apt install build-essential"

    include_dirs = (sysconfig.get_path(key) for key in ("include", "platinclude"))
    if not any(
        include_dir is not None and (Path(include_dir) / "Python.h").exists()
        for include_dir in include_dirs
    ):
        version = sysconfig.get_python_version()
        return False, (
            f"Python.h is not available under {sysconfig.get_path('include')}; "
            f"install Python headers with sudo apt install python{version}-dev"
        )
    return True, None


def train(
    device: torch.device,
    train_loader: DataLoader,
    checkpoint_dir: Path,
    seed: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    epochs: int,
    lr: float,
    weight_decay: float,
    drop_rate: float,
    drop_path_rate: float,
    pooling: str,
    warmup_epochs: float,
    grad_clip: float,
    ema_horizon_epochs: float,
    sam_rho: float,
    backbone_lr_mult: float,
    pretrained: bool,
    compile_model: bool,
    resume_from: Path | None,
) -> None:
    steps_per_epoch = len(train_loader)
    # Per-step EMA decay so the shadow averages over ema_horizon_epochs
    # epochs regardless of --batch-size; the warmup covers one epoch.
    ema_decay = math.exp(-1.0 / (ema_horizon_epochs * steps_per_epoch))
    ema_warmup_steps = steps_per_epoch

    # A resume keeps the LR schedule from the checkpoint; the run still stops
    # after --epochs, which may cut the cosine short.
    total_steps = requested_total_steps = steps_per_epoch * epochs
    warmup_steps = requested_warmup_steps = int(steps_per_epoch * warmup_epochs)

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    # bfloat16 needs Ampere+ tensor cores. Older GPUs (e.g. Turing GTX 16xx)
    # emulate it, which also breaks torch.compile's layout propagation, so
    # fall back to fp32 there.
    capability = torch.cuda.get_device_capability(device)
    autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float32

    # Peek at the checkpoint before building the model so resumes honour the
    # pooling variant it was trained with (attention vs global average).
    resume_payload = (
        torch.load(resume_from, map_location=device, weights_only=False) if resume_from is not None else None
    )

    # The checkpoint's own values win over this run's flags; say so when they
    # disagree instead of silently dropping what the user asked for.
    resume_pooling = str(resume_payload["pooling"]) if resume_payload is not None else None
    if resume_pooling is not None and resume_pooling != pooling:
        logger.info(
            "resuming with the checkpoint's pooling=%s (ignoring --pooling %s)",
            resume_pooling,
            pooling,
        )
    if pretrained and resume_payload is not None:
        logger.info("ignoring --pretrain: the resume overwrites every weight")

    # Resumes overwrite every weight anyway, so skip the download there.
    model = RotationModel(
        pretrained=pretrained and resume_from is None,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pooling=pooling if resume_pooling is None else resume_pooling,
    ).to(device)
    model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    ema = ModelEMA(
        model,
        decay=ema_decay,
        warmup_steps=ema_warmup_steps,
    )

    # lr_scale is applied on top of the schedule each step; a lower backbone
    # scale keeps pretrained features from being wrecked while the head learns.
    param_groups = [
        {"params": model.backbone.parameters(), "lr_scale": backbone_lr_mult},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("backbone.")], "lr_scale": 1.0},
    ]
    optimizer = SAM(
        param_groups,
        base_optimizer=torch.optim.AdamW,
        rho=sam_rho,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    logger.info("using SAM (rho=%.3f) over AdamW", sam_rho)
    # Inference picks the rotation with the highest upright logit
    # (argmax over 4 views), so the absolute decision threshold
    # doesn't matter - only the relative ranking.
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if resume_from is not None:
        assert resume_payload is not None
        model.load_state_dict(resume_payload["raw"])

        ema.module.load_state_dict(resume_payload["ema"])
        ema.step = int(resume_payload["ema_step"])

        # Saved state is plain AdamW; load it into SAM's base optimizer.
        optimizer.base_optimizer.load_state_dict(resume_payload["optimizer"])

        # map_location=device pulls the saved ByteTensors to CUDA, but
        # set_rng_state_all requires CPU ByteTensors - move them back.
        torch.set_rng_state(resume_payload["rng_cpu"].cpu())
        if resume_payload["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in resume_payload["rng_cuda"]])

        start_epoch = int(resume_payload["epoch"]) + 1
        total_steps = int(resume_payload["schedule_total_steps"])
        warmup_steps = int(resume_payload["schedule_warmup_steps"])
        if (total_steps, warmup_steps) != (requested_total_steps, requested_warmup_steps):
            logger.info(
                "checkpoint schedule wins for the LR curve: total %d, warmup %d "
                "(--epochs %d and --warmup-epochs %s would give total %d, warmup %d; "
                "the run still stops after --epochs)",
                total_steps,
                warmup_steps,
                epochs,
                warmup_epochs,
                requested_total_steps,
                requested_warmup_steps,
            )
        logger.info(
            "resumed from %s (next epoch=%d, ema_step=%d)",
            resume_from,
            start_epoch,
            ema.step,
        )

    # Compile after any state_dict load so checkpoints are saved/loaded in
    # uncompiled form (no "_orig_mod." prefix in the on-disk dict).
    if compile_model:
        can_compile, reason = can_compile_model()
        if can_compile:
            model = cast(RotationModel, torch.compile(model))
            logger.info("enabled torch.compile")
        else:
            logger.warning("disabled torch.compile: %s", reason)
    else:
        logger.info("disabled torch.compile by request")

    if pretrained and resume_from is None:
        logger.info("initialized backbone from ImageNet-pretrained weights")

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
        seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        # GPU-side accumulators flushed at LOG_STEP boundaries so .item()
        # doesn't sync every step.
        loss_buf = torch.zeros((), device=device)
        correct_buf = torch.zeros((), device=device, dtype=torch.long)
        gn_buf = torch.zeros((), device=device)
        pending_n = 0
        pending_steps = 0

        for imgs, labels in pbar:
            imgs, labels = move_batch(imgs, labels, device)

            scale = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_steps,
            )

            # set lr
            for group in optimizer.param_groups:
                group["lr"] = lr * scale * group.get("lr_scale", 1.0)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):  # pyright: ignore[reportPrivateImportUsage]
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # First pass climbs to the worst-case point; the second pass
            # recomputes the gradient there and applies the real update.
            optimizer.first_step(zero_grad=True)
            optimizer.disable_running_stats(model)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss_adv = loss_fn(model(imgs), labels)
            loss_adv.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
            optimizer.second_step(zero_grad=True)
            optimizer.enable_running_stats(model)
            ema.update(model)

            preds = (logits > 0).float()
            n = imgs.size(0)
            loss_buf += loss.detach() * n
            correct_buf += (preds == labels).sum()
            gn_buf += grad_norm.detach()
            seen += n
            pending_n += n
            pending_steps += 1

            if global_step % constants.LOG_STEP == 0:
                window_loss = loss_buf.item() / max(1, pending_n)
                window_acc = int(correct_buf.item()) / max(1, pending_n)
                window_gn = gn_buf.item() / max(1, pending_steps)
                loss_buf.zero_()
                correct_buf.zero_()
                gn_buf.zero_()
                pending_n = 0
                pending_steps = 0

                lr_now = optimizer.param_groups[-1]["lr"]
                pbar.set_postfix(
                    acc=f"{window_acc:.5f}",
                    loss=f"{window_loss:.5f}",
                    lr=f"{lr_now:.2e}",
                    gn=f"{window_gn:.5f}",
                )
                logger.info(
                    "epoch %d step %d/%d loss=%.5f acc=%.5f lr=%.2e grad_norm=%.5f",
                    epoch,
                    global_step,
                    total_steps,
                    window_loss,
                    window_acc,
                    lr_now,
                    window_gn,
                )
            global_step += 1

        dt = time.monotonic() - epoch_start
        logger.info(
            "epoch %d done in %.0fs, throughput %.1f img/s",
            epoch,
            dt,
            seen / dt,
        )

        ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt,
            model,
            ema,
            optimizer,
            mean,
            std,
            extras={
                "epoch": epoch,
                "schedule_total_steps": total_steps,
                "schedule_warmup_steps": warmup_steps,
                "pooling": pooling,
                "pretrained": pretrained,
            },
        )
        logger.info("saved %s", ckpt)

    logger.info("done.")


def run_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # Pretrained features expect the ImageNet normalization they were trained
    # with; from-scratch runs compute stats over the train split instead.
    pretrain_mean_std = (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD) if args.pretrain else None
    ds, (mean, std) = build_dataset(
        split="train",
        datasets=build_dataset_specs(args.data_root),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch_factor,
        mean_std=pretrain_mean_std,
        invalidate_cache=args.invalidate_cache,
        erasing_p=args.erasing_p,
        crop_policy=args.crop_policy,
    )

    dl = build_loader(
        ds,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )

    train(
        device=device,
        train_loader=dl,
        seed=args.seed,
        checkpoint_dir=constants.CHECKPOINT_DIR,
        mean=mean,
        std=std,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        drop_rate=args.dropout,
        drop_path_rate=args.drop_path,
        pooling=args.pooling,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        ema_horizon_epochs=args.ema_horizon_epochs,
        sam_rho=args.sam_rho,
        backbone_lr_mult=args.backbone_lr_mult,
        pretrained=args.pretrain,
        compile_model=not args.no_compile,
        resume_from=args.resume,
    )
