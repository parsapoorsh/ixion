"""Rotation detector core: data pipeline, training, evaluation, and tools.

Everything that is not a model building block or a file-system helper lives
here: transform factories, datasets, norm stats, the training loop,
checkpoint loading, validation, four-view inference, GradCAM, the browse
browser, DFR head retraining, ONNX export, and the subcommand bodies that
``cli.py`` dispatches after parsing.

Keeping the tools in one module lets them share a single checkpoint loader,
stats decoder, and vote construction instead of each carrying a copy.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import shutil
import sys
import sysconfig
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

import constants
from checkpoint import WeightName, load_checkpoint_model, save_checkpoint
from config import build_dataset_specs
from data import (
    RotationPairDataset,
    build_dataset,
    build_eval_transform,
    build_loader,
    build_train_transform,
    display_preprocess,
)
from inference import (
    RotationVote,
    circular_mean_angle,
    predict_rotations,
    stack_rotation_views,
)
from model import SAM, ModelEMA, RotationModel
from utils.image_io import open_rgb
from utils.index import index_files
from utils.logsetup import setup_logging

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)

N_ROTS = len(constants.ROTATIONS)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Deep Feature Reweighting (DFR)
# ---------------------------------------------------------------------------


def retrain_head(
    model: RotationModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> None:
    """Retrain the head only, with the backbone frozen in eval mode.

    Same binary BCE objective as the main training loop.
    """
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.head.train()
        epoch_start = time.monotonic()
        loss_sum = 0.0
        correct = 0
        seen = 0

        pbar = tqdm(
            loader,
            desc=f"dfr epoch {epoch}",
            unit="batch",
            leave=False,
            file=sys.stdout,
        )
        for step, (imgs, labels) in enumerate(pbar):
            imgs, labels = move_batch(imgs, labels, device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
            optimizer.step()

            n = imgs.size(0)
            loss_sum += float(loss.detach()) * n
            preds = (logits.detach() > 0).float()
            correct += int((preds == labels).sum())
            seen += n

            if step % constants.LOG_STEP == 0:
                pbar.set_postfix(
                    acc=f"{correct / max(1, seen):.5f}",
                    loss=f"{loss_sum / max(1, seen):.5f}",
                )

        dt = time.monotonic() - epoch_start
        logger.info(
            "dfr epoch %d done in %.0fs: loss=%.5f acc=%.5f",
            epoch,
            dt,
            loss_sum / max(1, seen),
            correct / max(1, seen),
        )


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


class GradCAM:
    """GradCAM and guided-GradCAM heatmap computation for one loaded model."""

    def __init__(
        self,
        model: RotationModel,
        device: torch.device,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.model = model
        self.device = device
        self.preprocess = build_eval_transform(mean=mean, std=std)

    def compute(self, tensor: torch.Tensor) -> tuple[list[np.ndarray], list[float]]:
        """Run one forward+backward and return per-sample GradCAM heatmaps
        (each in [0, 1]) plus per-sample upright logits."""
        target_layer: nn.Module = self.model.backbone.conv_head  # type: ignore[assignment]

        saved_features: list[torch.Tensor] = []
        saved_grads: list[torch.Tensor] = []

        def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
            saved_features.append(out)

        def bwd(_m: nn.Module, _gin: object, gout: tuple[torch.Tensor, ...]) -> None:
            saved_grads.append(gout[0])

        fh = target_layer.register_forward_hook(fwd)
        bh = target_layer.register_full_backward_hook(bwd)  # type: ignore[arg-type]

        try:
            output = self.model(tensor)  # (B,) or (B, 1)
            self.model.zero_grad()
            output.sum().backward()  # per-sample grads are independent
        finally:
            fh.remove()
            bh.remove()

        feat = saved_features[0]  # (B, C, H, W)
        grad = saved_grads[0]  # (B, C, H, W)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = nn.functional.relu((weights * feat).sum(dim=1))  # (B, H, W)
        cam_max = cam.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        cam_np = (cam / cam_max).cpu().detach().float().numpy()

        logits = output.detach().cpu().float().reshape(-1).tolist()
        return list(cam_np), logits

    def guided(self, tensor: torch.Tensor, cams: list[np.ndarray]) -> list[np.ndarray]:
        """Guided backprop gated by the supplied GradCAM heatmaps.

        Produces sharp, class-discriminative, edge-like maps at full input
        resolution. SiLU stand-in for ReLU in the guided-BP gate: only pass
        gradient where the SiLU output was positive AND the incoming gradient
        is positive. Runs a single batched forward+backward over the batch.
        """
        silu_outputs: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_fwd(idx: int):
            def fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
                silu_outputs[idx] = out

            return fwd

        def make_bwd(idx: int):
            def bwd(
                _m: nn.Module,
                _gin: object,
                gout: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...]:
                out = silu_outputs[idx]
                new_grad = gout[0].clone()
                new_grad[out <= 0] = 0
                new_grad[new_grad < 0] = 0
                return (new_grad,)

            return bwd

        silu_modules = [m for m in self.model.modules() if isinstance(m, nn.SiLU)]
        original_inplace = [sm.inplace for sm in silu_modules]
        for sm in silu_modules:
            sm.inplace = False  # backward hooks reject view+inplace combo
        for i, sm in enumerate(silu_modules):
            handles.append(sm.register_forward_hook(make_fwd(i)))
            handles.append(sm.register_full_backward_hook(make_bwd(i)))  # type: ignore[arg-type]

        try:
            x = tensor.detach().clone().requires_grad_(True)
            output = self.model(x)
            self.model.zero_grad()
            output.sum().backward()
        finally:
            for h in handles:
                h.remove()
            for sm, was_inplace in zip(silu_modules, original_inplace, strict=True):
                sm.inplace = was_inplace

        grad = x.grad.abs().sum(dim=1)  # type: ignore[union-attr]  # (B, H, W)
        grad_np = grad.cpu().detach().float().numpy()

        guided_out: list[np.ndarray] = []
        for i, cam in enumerate(cams):
            h_in, w_in = grad_np[i].shape
            cam_up = (
                np.array(
                    Image.fromarray((cam * 255).astype(np.uint8)).resize((w_in, h_in), Image.Resampling.BICUBIC),
                ).astype(np.float32)
                / 255.0
            )

            guided = grad_np[i] * cam_up
            p99 = float(np.percentile(guided, 99))
            if p99 > 0:
                guided = np.clip(guided / p99, 0.0, 1.0)
            guided_out.append(guided)
        return guided_out

    def rotations(
        self,
        pil_img: Image.Image,
    ) -> tuple[list[Image.Image], list[float], list[np.ndarray], torch.Tensor]:
        # Display images: resize + center-crop, then PIL-rotate each view.
        crop = display_preprocess(pil_img)
        displays = [crop.rotate(a) if a != 0 else crop for a in constants.ROTATIONS]

        # Model input: transform once (resize/crop/to_tensor/normalize), then
        # rotate the tensor - same geometry as train/infer (RotationPairDataset,
        # predict_rotations), not PIL-rotate-then-transform.
        base = self.preprocess(pil_img)
        tensors = stack_rotation_views(base, self.device)

        cams, logits = self.compute(tensors)
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=0).tolist()
        return displays, probs, cams, tensors


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def overlay_heatmap(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.65,
    colormap: str = "jet",
) -> Image.Image:
    cam = np.sqrt(cam)  # spread activation - avoids tiny concentrated hot spots
    heatmap = mpl.colormaps[colormap](cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_img.size, Image.Resampling.BICUBIC)
    blended = np.array(pil_img, dtype=np.float32) * (1 - alpha) + np.array(heatmap_pil, dtype=np.float32) * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def render_rotation_grid(
    displays: list[Image.Image],
    probs: list[float],
    heatmaps: list[np.ndarray],
    title: str,
    out_path: Path,
    colormap: str = "jet",
) -> None:
    """Render a 2x2 rotation grid with `heatmaps[i]` overlaid on `displays[i]`."""
    pred = int(np.argmax(probs))

    fig = Figure(figsize=(10, 10))
    axes = fig.subplots(2, 2)

    for idx, (angle, heat, prob, disp) in enumerate(
        zip(constants.ROTATIONS, heatmaps, probs, displays, strict=True)
    ):
        img_overlay = overlay_heatmap(disp, heat, colormap=colormap)
        ax = axes[idx // 2, idx % 2]
        marker = " *" if idx == pred else ""
        ax.imshow(img_overlay)
        ax.set_title(f"{angle}°{marker}  softmax={prob:.0%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Browse (tkinter)
# ---------------------------------------------------------------------------

IMAGE_EXTS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
        ".jxl",
    }
)
PREFETCH_SIZE = 10
DONE = object()  # sentinel pushed by worker when exhausted


class InferResult(TypedDict):
    path: Path
    corrected: Image.Image
    vote: RotationVote
    abstained: bool
    error: str | None


def prefetch_worker(
    paths: list[Path],
    model: RotationModel,
    device: torch.device,
    preprocess: T.Compose,
    out: queue.Queue,
    bar: tqdm,
    abstain_margin: float | None,
) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)

    for path in paths:
        try:
            original = open_rgb(path)
            vote = predict_rotations(
                model=model,
                img=original,
                device=device,
                preprocess=preprocess,
            )
            abstained = abstain_margin is not None and vote["margin"] < abstain_margin
            # Low margin: keep original (no auto-rotate).
            if abstained:
                corrected = original
            else:
                corrected = original.rotate(vote["best_deg"], expand=True)
            result: InferResult = {
                "path": path,
                "corrected": corrected,
                "vote": vote,
                "abstained": abstained,
                "error": None,
            }
        except Exception as exc:
            blank = Image.new("RGB", (constants.INPUT_SIZE, constants.INPUT_SIZE), color=(30, 30, 30))
            result = {
                "path": path,
                "corrected": blank,
                "vote": {
                    "probs": dict.fromkeys(constants.ROTATIONS, 0.25),
                    "best_deg": 0,
                    "margin": 0.0,
                },
                "abstained": False,
                "error": str(exc),
            }

        out.put(result)
        bar.update(1)

    out.put(DONE)


def show_result(result: InferResult, index: int, total: int) -> None:
    """Open a blocking tkinter window for *result*; returns when it is closed."""
    import tkinter as tk

    from PIL import ImageTk

    prob_str = "  ".join(f"{deg}°: {prob:.3f}" for deg, prob in sorted(result["vote"]["probs"].items()))
    if result["abstained"]:
        decision = f"ABSTAIN (margin={result['vote']['margin']:.3f})"
    else:
        decision = f"correction: {result['vote']['best_deg']}° (margin={result['vote']['margin']:.3f})"
    title = f"[{index}/{total}]  {result['path'].name}  ->  {decision}  |  {prob_str}"

    root = tk.Tk()
    root.title(title)
    root.resizable(True, True)
    root.bind("<q>", lambda _e: root.destroy())

    img = result["corrected"].copy()
    img.thumbnail((1280, 900), Image.Resampling.LANCZOS)

    photo = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, image=photo, bg="black")
    img_label.pack(fill=tk.BOTH, expand=True)

    if result["error"]:
        err_label = tk.Label(
            root,
            text=f"⚠  {result['error']}",
            fg="#ff6060",
            bg="black",
            font=("Helvetica", 11),
        )
        err_label.pack()

    info = tk.Label(
        root,
        text=prob_str,
        fg="#cccccc",
        bg="#1a1a1a",
        font=("Courier", 10),
        pady=4,
    )
    info.pack(fill=tk.X)

    root.mainloop()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def export_onnx(
    model: RotationModel,
    out_path: Path,
    device: torch.device,
    opset: int,
    dynamo: bool,
    verify: bool,
) -> None:
    """Trace ``model`` to ONNX with a dynamic batch axis."""
    dummy = torch.randn(1, 3, constants.INPUT_SIZE, constants.INPUT_SIZE, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs: dict = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": {
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        "dynamo": dynamo,
    }

    logger.info(
        "exporting ONNX -> %s (opset=%d, dynamo=%s, device=%s)",
        out_path,
        opset,
        dynamo,
        device,
    )

    # Legacy exporter writes ``f`` directly and returns None. The dynamo
    # exporter returns an ONNXProgram that must be saved when f is None.
    try:
        result = torch.onnx.export(
            model,
            (dummy,),
            str(out_path) if not dynamo else None,
            **export_kwargs,
        )
    except ModuleNotFoundError as exc:
        if dynamo and "onnxscript" in str(exc):
            raise RuntimeError(
                "dynamo export needs the onnxscript package "
                "(pip install onnxscript); or omit --dynamo for the legacy exporter"
            ) from exc
        raise

    if result is not None:
        result.save(str(out_path))

    if not out_path.is_file():
        raise RuntimeError(f"ONNX export produced no file at {out_path}")

    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("verify requires onnxruntime") from exc

        with torch.no_grad():
            torch_out = model(dummy).detach().float().cpu().numpy()

        session = ort.InferenceSession(
            str(out_path),
            providers=["CPUExecutionProvider"],
        )
        ort_out = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: dummy.detach().float().cpu().numpy()},
        )[0]

        max_abs = float(abs(torch_out.reshape(-1) - ort_out.reshape(-1)).max())
        logger.info("verify max|torch-ort|=%.6g on dummy batch", max_abs)
        if max_abs > 1e-4:
            logger.warning("torch/ort diverge more than 1e-4; check opset / dynamo flags")


# ---------------------------------------------------------------------------
# CLI subcommand bodies
# ---------------------------------------------------------------------------


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


def run_browse(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"Loading {args.checkpoint}  ({args.weights} weights)...")
    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    preprocess = build_eval_transform(mean=mean, std=std)
    print("Model ready.\n")

    root_dir = Path(args.dir)
    print(f"Scanning {root_dir}...")
    paths = sorted(p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print("No images found.")
        return
    print(f"Found {len(paths)} image(s).\n")

    result_queue: queue.Queue[InferResult | object] = queue.Queue(maxsize=PREFETCH_SIZE)
    bar = tqdm(total=len(paths), desc="Inferring", unit="img", dynamic_ncols=True)

    worker = threading.Thread(
        target=prefetch_worker,
        args=(
            paths,
            model,
            device,
            preprocess,
            result_queue,
            bar,
            args.abstain_margin,
        ),
        daemon=True,
    )
    worker.start()

    shown = 0
    skipped = 0
    abstained = 0
    processed = 0
    while True:
        item = result_queue.get()
        if item is DONE:
            break

        result = cast(InferResult, item)
        processed += 1

        best_norm = max(result["vote"]["probs"].values())
        if not result["error"] and best_norm >= args.conf_threshold:
            skipped += 1
            tqdm.write(
                f"[{processed}/{len(paths)}]  {result['path'].name}"
                f"  skipped (conf={best_norm:.3f} >= {args.conf_threshold})"
            )
            continue

        shown += 1
        if result["abstained"]:
            abstained += 1

        if result["error"]:
            tqdm.write(f"[{processed}/{len(paths)}]  {result['path']}  ERROR: {result['error']}")
        else:
            status = "abstain" if result["abstained"] else f"best={result['vote']['best_deg']}°"
            norm_str = ", ".join(f"{d}°:{p:.3f}" for d, p in sorted(result["vote"]["probs"].items()))
            tqdm.write(
                f"[{processed}/{len(paths)}]  {result['path'].name}"
                f"  {status}"
                f"  margin={result['vote']['margin']:.3f}"
                f"  conf={best_norm:.3f}"
                f"  norm={{{norm_str}}}"
            )

        show_result(result, shown, len(paths))

    bar.close()
    worker.join()
    print(
        f"\nDone - displayed {shown} / {len(paths)} image(s)"
        f"  ({skipped} skipped conf >= {args.conf_threshold},"
        f" {abstained} abstained on margin)."
    )


def run_gradcam(args: argparse.Namespace) -> None:
    images = [p for p in args.images if p.is_file()]

    device = torch.device(args.device)
    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    gradcam = GradCAM(model, device, mean, std)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        pil = open_rgb(img_path)
        displays, probs, cams, tensors = gradcam.rotations(pil)
        angle = circular_mean_angle(probs)

        render_rotation_grid(
            displays,
            probs,
            cams,
            f"{img_path.name} - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_gradcam.png",
            colormap="jet",
        )
        print(f"{img_path.name} - predicted angle {angle:.1f}°")

        guided_cams = gradcam.guided(tensors, cams)
        render_rotation_grid(
            displays,
            probs,
            guided_cams,
            f"{img_path.name} - guided GradCAM - predicted angle {angle:.1f}°",
            args.out_dir / f"{img_path.stem}_guidedcam.png",
            colormap="inferno",
        )


def run_dfr(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    _, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
        strict_stats=True,
    )
    model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

    # Freeze everything outside the head; the backbone also stays in eval mode
    # so its BatchNorm running stats remain at the full model's values.
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    model.backbone.eval()

    if not args.no_reinit_head:
        for module in model.head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore[operator]
        logger.info("reinitialised head for a fresh DFR fit")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable params: %d | frozen params: %d", trainable, total - trainable)

    # Indexed under its own cache name so it never collides with the main
    # train/val file indexes.
    files = index_files(
        name="dfr",
        roots=list(args.data_root),
        cache_dir=constants.INDEX_CACHE_DIR,
        invalidate_cache=args.invalidate_cache,
    )
    dataset = RotationPairDataset(
        files=files,
        transform=build_train_transform(mean=mean, std=std),
    )
    logger.info("DFR reweighting set: %d files -> %d pairs", len(files), len(dataset))
    loader = build_loader(
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )

    retrain_head(
        model=model,
        loader=loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )

    state = model.state_dict()
    payload = {
        "raw": state,
        "ema": state,
        "stats": {"mean": list(mean), "std": list(std)},
        "epoch": "dfr",
        "pooling": model.pooling,
        "dfr_source": str(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info("saved DFR checkpoint %s", args.output)


def run_export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    payload, model, (mean, std) = load_checkpoint_model(
        args.checkpoint,
        device,
        weights=args.weights,
    )
    meta = {
        "checkpoint": str(args.checkpoint),
        "epoch": payload.get("epoch"),
        "weights": args.weights,
        "pooling": payload.get("pooling", "avg"),
        "pretrained": payload.get("pretrained"),
        "mean": list(mean),
        "std": list(std),
        "input_size": constants.INPUT_SIZE,
        "resize_size": constants.RESIZE_SIZE,
        "rotations": list(constants.ROTATIONS),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }
    out_path = args.out or args.checkpoint.with_suffix(".onnx")

    export_onnx(
        model=model,
        out_path=out_path,
        device=device,
        opset=args.opset,
        dynamo=args.dynamo,
        verify=args.verify,
    )

    meta_path = out_path.with_suffix(".json")
    meta_payload = {
        **meta,
        "onnx": str(out_path),
        "opset": args.opset,
        "dynamo": args.dynamo,
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n")
    logger.info("wrote metadata %s", meta_path)
    logger.info("done: %s (%.1f MiB)", out_path, out_path.stat().st_size / (1024 * 1024))


HANDLERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "train": run_train,
    "finetune": run_train,
    "validate": run_validate,
    "browse": run_browse,
    "gradcam": run_gradcam,
    "dfr": run_dfr,
    "export": run_export,
}


def run(args: argparse.Namespace) -> None:
    """Configure logging and dispatch one parsed subcommand."""
    setup_logging(constants.LOGS_DIR)
    HANDLERS[args.command](args)
