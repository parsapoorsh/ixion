"""GradCAM visualiser for the upright detector.

Shows the image at all 4 rotations; for each, displays the model's
P(upright) and a GradCAM heatmap of the upright logit.
Target layer: backbone.conv_head (last 1x1 conv before global pool,
320->1280 channels, spatial resolution ~7x7 for 224x224 input).

Usage:
    python src/gradcam.py \\
        --checkpoint checkpoints/epoch_005.pt \\
        --device cuda:1 \\
        --images examples/*.jpg \\
        --out-dir gradcam_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib.figure import Figure
from PIL import Image
from torchvision import transforms as T

import constants
from models.rotation import RotationModel
from stats import NormStats, load_stats_cache
from utils.image_io import open_rgb


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[RotationModel, NormStats]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "stats" in payload:
        raw_stats = payload["stats"]
        stats = NormStats(mean=tuple(raw_stats["mean"]), std=tuple(raw_stats["std"]))  # type: ignore[arg-type]
    else:
        cache_file = constants.INDEX_CACHE_DIR / "stats_train.json"
        stats = load_stats_cache(cache_file)
    model = RotationModel()
    model.load_state_dict(payload["ema"])
    model.to(device)
    model.train(False)
    return model, stats


def compute_gradcam(
    model: RotationModel,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Return a (H, W) heatmap in [0, 1] for the upright logit."""
    target_layer: nn.Module = model.backbone.conv_head  # type: ignore[assignment]

    saved_features: list[torch.Tensor] = []
    saved_grads: list[torch.Tensor] = []

    def _fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
        saved_features.append(out)

    def _bwd(_m: nn.Module, _gin: object, gout: tuple[torch.Tensor, ...]) -> None:
        saved_grads.append(gout[0])

    fh = target_layer.register_forward_hook(_fwd)
    bh = target_layer.register_full_backward_hook(_bwd)  # type: ignore[arg-type]

    output = model(tensor)
    model.zero_grad()
    output[0].backward()

    fh.remove()
    bh.remove()

    feat = saved_features[0]           # (1, C, H, W)
    grad = saved_grads[0]              # (1, C, H, W)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = nn.functional.relu((weights * feat).sum(dim=1)).squeeze(0)
    cam = cam.cpu().detach().float().numpy()

    if cam.max() > 0:
        cam /= cam.max()
    return cam


def make_overlay(
    pil_img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.65,
    colormap: str = "jet",
) -> Image.Image:
    cam = np.sqrt(cam)  # spread activation — avoids tiny concentrated hot spots
    heatmap = mpl.colormaps[colormap](cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        pil_img.size, Image.Resampling.BICUBIC
    )
    blended = (
        np.array(pil_img, dtype=np.float32) * (1 - alpha)
        + np.array(heatmap_pil, dtype=np.float32) * alpha
    )
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def compute_guided_gradcam(
    model: RotationModel,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Guided backprop gated by the GradCAM heatmap.

    Produces a sharp, class-discriminative, edge-like map at full input
    resolution — the classic 'guided GradCAM' visualisation. SiLU stand-in
    for ReLU in the guided-BP gate: only pass gradient where the SiLU output
    was positive AND the incoming gradient is positive.
    """
    cam = compute_gradcam(model, tensor.detach().clone())

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

    silu_modules = [m for m in model.modules() if isinstance(m, nn.SiLU)]
    original_inplace = [sm.inplace for sm in silu_modules]
    for sm in silu_modules:
        sm.inplace = False  # backward hooks reject view+inplace combo
    for i, sm in enumerate(silu_modules):
        handles.append(sm.register_forward_hook(make_fwd(i)))
        handles.append(sm.register_full_backward_hook(make_bwd(i)))  # type: ignore[arg-type]

    try:
        x = tensor.detach().clone().requires_grad_(True)
        output = model(x)
        model.zero_grad()
        output[0].backward()
    finally:
        for h in handles:
            h.remove()
        for sm, was_inplace in zip(silu_modules, original_inplace):
            sm.inplace = was_inplace

    grad = x.grad[0].abs().sum(dim=0)              # type: ignore[union-attr]  # (H, W)
    grad_np = grad.cpu().detach().float().numpy()

    h_in, w_in = grad_np.shape
    cam_up = np.array(
        Image.fromarray((cam * 255).astype(np.uint8))
        .resize((w_in, h_in), Image.Resampling.BICUBIC),
    ).astype(np.float32) / 255.0

    guided = grad_np * cam_up
    p99 = float(np.percentile(guided, 99))
    if p99 > 0:
        guided = np.clip(guided / p99, 0.0, 1.0)
    return guided


def compute_occlusion(
    model: RotationModel,
    tensor: torch.Tensor,
    patch: int = 32,
    stride: int = 16,
    batch_size: int = 64,
) -> np.ndarray:
    """Drop in P(upright) when each spatial patch is masked to the mean colour.

    Returns an (H, W) heatmap where higher = bigger drop = the model relied
    on that region for its upright decision. ~170 forward passes per call
    with default patch/stride at 224×224, batched for speed. Most direct
    readout of 'what the model is actually using' since it perturbs the
    real input rather than reasoning about gradients.
    """
    _, _, h, w = tensor.shape
    with torch.no_grad():
        base = float(model(tensor).sigmoid().item())

    positions = [
        (y, x)
        for y in range(0, h - patch + 1, stride)
        for x in range(0, w - patch + 1, stride)
    ]

    drops = torch.empty(len(positions), device=tensor.device)
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            batch = tensor.expand(len(chunk), -1, -1, -1).clone()
            for i, (y, x) in enumerate(chunk):
                batch[i, :, y:y + patch, x:x + patch] = 0.0
            probs = model(batch).sigmoid()
            drops[start:start + len(chunk)] = base - probs

    heat = torch.zeros(h, w, device=tensor.device)
    counts = torch.zeros(h, w, device=tensor.device)
    for (y, x), d in zip(positions, drops.tolist()):
        heat[y:y + patch, x:x + patch] += d
        counts[y:y + patch, x:x + patch] += 1
    heat_np = (heat / counts.clamp(min=1)).cpu().numpy()
    heat_np = np.clip(heat_np, 0.0, None)  # only positive drops matter
    if heat_np.max() > 0:
        heat_np /= heat_np.max()
    return heat_np


def visualise_grid(
    displays: list[Image.Image],
    probs: list[float],
    heatmaps: list[np.ndarray],
    title: str,
    out_path: Path,
    colormap: str = "jet",
) -> None:
    """Render a 2×2 rotation grid with `heatmaps[i]` overlaid on `displays[i]`."""
    pred = int(np.argmax(probs))

    fig = Figure(figsize=(10, 10))
    axes = fig.subplots(2, 2)

    for idx, (angle, heat, prob, disp) in enumerate(
        zip(constants.ROTATIONS, heatmaps, probs, displays)
    ):
        img_overlay = make_overlay(disp, heat, colormap=colormap)
        ax = axes[idx // 2, idx % 2]
        marker = " *" if idx == pred else ""
        ax.imshow(img_overlay)
        ax.set_title(f"{angle}°{marker}  P(upright)={prob:.1%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def compute_rotation_diffs(
    model: RotationModel,
    displays: list[Image.Image],
    preprocess: T.Compose,
    device: torch.device,
    ref_idx: int,
) -> list[tuple[int, np.ndarray]]:
    """Per-rotation diff in conv_head features vs the reference rotation.

    Captures `backbone.conv_head` activations for every rotation, counter-
    rotates each non-reference feature map back to the reference's frame,
    then returns mean |diff| over channels. High values = spatial locations
    where the model's late-stage representation changes between upright and
    rotated views — i.e., what the model is using to *tell rotations apart*.
    """
    target = model.backbone.conv_head  # type: ignore[assignment]

    features: list[torch.Tensor] = []
    for disp in displays:
        saved: list[torch.Tensor] = []

        def _fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
            saved.append(out)

        h = target.register_forward_hook(_fwd)
        with torch.no_grad():
            t = preprocess(disp).unsqueeze(0).to(device)  # type: ignore[union-attr]
            model(t)
        h.remove()
        features.append(saved[0].squeeze(0))  # (C, H, W)

    ref_feat = features[ref_idx]
    ref_angle = constants.ROTATIONS[ref_idx]

    diffs: list[tuple[int, np.ndarray]] = []
    for i, feat in enumerate(features):
        if i == ref_idx:
            continue
        delta = (constants.ROTATIONS[i] - ref_angle) % 360
        k = (-delta // 90) % 4  # counter-rotate to align with ref frame
        feat_aligned = torch.rot90(feat, k=int(k), dims=(-2, -1))
        diff = (ref_feat - feat_aligned).abs().mean(dim=0)
        d = diff.cpu().float().numpy()
        if d.max() > 0:
            d /= d.max()
        diffs.append((constants.ROTATIONS[i], d))
    return diffs


def visualise_rotation_diffs(
    ref_display: Image.Image,
    ref_angle: int,
    diffs: list[tuple[int, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    """Show the reference image with each rotation-vs-reference diff overlaid."""
    fig = Figure(figsize=(4 * len(diffs), 4.5))
    axes = fig.subplots(1, len(diffs))
    if len(diffs) == 1:
        axes = [axes]

    for ax, (angle, heat) in zip(axes, diffs):
        overlay = make_overlay(ref_display, heat, alpha=0.6, colormap="magma")
        ax.imshow(overlay)
        ax.set_title(f"{ref_angle}° vs {angle}°", fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def compute_stage_maps(
    model: RotationModel,
    tensor: torch.Tensor,
) -> list[tuple[str, np.ndarray]]:
    """Mean |activation| over channels for every EfficientNet stage.

    Returns one (name, H×W heatmap) tuple per stage, ordered from input to
    output: stem → blocks[0..6] → conv_head. Late stages have very small
    spatial resolution (7×7) — the overlay's upsample handles that.
    """
    backbone = model.backbone
    targets: list[tuple[str, nn.Module]] = [("stem", backbone.bn1)]  # type: ignore[arg-type]
    for i, block in enumerate(backbone.blocks):  # type: ignore[union-attr]
        targets.append((f"block{i}", block))
    targets.append(("conv_head", backbone.conv_head))  # type: ignore[arg-type]

    saved: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def _fwd(_m: nn.Module, _inp: object, out: torch.Tensor) -> None:
            saved[name] = out
        return _fwd

    handles = [layer.register_forward_hook(_make_hook(name)) for name, layer in targets]
    with torch.no_grad():
        model(tensor)
    for h in handles:
        h.remove()

    out: list[tuple[str, np.ndarray]] = []
    for name, _ in targets:
        feat = saved[name].squeeze(0)              # (C, H, W)
        act = feat.abs().mean(dim=0)               # (H, W)
        act_np = act.cpu().float().numpy()
        if act_np.max() > 0:
            act_np /= act_np.max()
        out.append((name, act_np))
    return out


def visualise_stages(
    display: Image.Image,
    stage_maps: list[tuple[str, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    """3×3 grid showing how the activation pattern evolves across stages."""
    n = len(stage_maps)
    cols = 3
    rows = (n + cols - 1) // cols

    fig = Figure(figsize=(4 * cols, 4 * rows))
    axes = fig.subplots(rows, cols)

    for idx, (name, heat) in enumerate(stage_maps):
        ax = axes[idx // cols, idx % cols]
        overlay = make_overlay(display, heat, alpha=0.6, colormap="viridis")
        ax.imshow(overlay)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def compute_rotations(
    model: RotationModel,
    pil_img: Image.Image,
    preprocess: T.Compose,
    device: torch.device,
) -> tuple[list[Image.Image], list[float], list[np.ndarray]]:
    crop = T.Compose([
        T.Resize(constants.RESIZE_SIZE),
        T.CenterCrop(constants.INPUT_SIZE),
    ])(pil_img)

    probs: list[float] = []
    cams: list[np.ndarray] = []
    displays: list[Image.Image] = []

    for angle in constants.ROTATIONS:
        rotated = crop.rotate(angle) if angle != 0 else crop
        displays.append(rotated)

        with torch.no_grad():
            tensor = preprocess(rotated).unsqueeze(0).to(device)  # type: ignore[union-attr]
            prob = float(model(tensor).sigmoid().item())
        probs.append(prob)

        t = preprocess(rotated).unsqueeze(0).to(device)  # type: ignore[union-attr]
        cams.append(compute_gradcam(model, t))

    return displays, probs, cams


def visualise(
    displays: list[Image.Image],
    probs: list[float],
    cams: list[np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    visualise_grid(displays, probs, cams, title, out_path, colormap="jet")


def main(args) -> None:
    args.images = [p for p in args.images if p.is_file()]

    device = torch.device(args.device)
    model, stats = load_model(args.checkpoint, device)

    preprocess = T.Compose([
        T.Resize(constants.RESIZE_SIZE),
        T.CenterCrop(constants.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(stats.mean, stats.std),
    ])

    args.out_dir.mkdir(exist_ok=True)

    for img_path in args.images:
        pil = open_rgb(img_path)
        displays, probs, cams = compute_rotations(model, pil, preprocess, device)

        visualise(
            displays,
            probs,
            cams,
            img_path.name,
            args.out_dir / f"{img_path.stem}_gradcam.png",
        )

        occlusions = [
            compute_occlusion(model, preprocess(disp).unsqueeze(0).to(device))  # type: ignore[union-attr]
            for disp in displays
        ]
        visualise_grid(
            displays,
            probs,
            occlusions,
            f"{img_path.name} — occlusion sensitivity",
            args.out_dir / f"{img_path.stem}_occlusion.png",
            colormap="inferno",
        )

        guided_cams = [
            compute_guided_gradcam(model, preprocess(disp).unsqueeze(0).to(device))  # type: ignore[union-attr]
            for disp in displays
        ]
        visualise_grid(
            displays,
            probs,
            guided_cams,
            f"{img_path.name} — guided GradCAM",
            args.out_dir / f"{img_path.stem}_guidedcam.png",
            colormap="inferno",
        )

        pred = int(np.argmax(probs))
        pred_display = displays[pred]
        pred_tensor = preprocess(pred_display).unsqueeze(0).to(device)  # type: ignore[union-attr]
        stage_maps = compute_stage_maps(model, pred_tensor)
        visualise_stages(
            pred_display,
            stage_maps,
            f"{img_path.name} — per-stage activations @ {constants.ROTATIONS[pred]}°",
            args.out_dir / f"{img_path.stem}_stages.png",
        )

        diffs = compute_rotation_diffs(model, displays, preprocess, device, pred)
        visualise_rotation_diffs(
            pred_display,
            constants.ROTATIONS[pred],
            diffs,
            f"{img_path.name} — late-feature diff vs predicted upright",
            args.out_dir / f"{img_path.stem}_rotdiff.png",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("gradcam_out"))
    args = p.parse_args()

    main(args)
