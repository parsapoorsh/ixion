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
import matplotlib.cm as cm
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
        if not cache_file.exists():
            raise FileNotFoundError(
                f"checkpoint has no 'stats' and cache file not found: {cache_file}"
            )
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


def make_overlay(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.65) -> Image.Image:
    cam = np.sqrt(cam)  # spread activation — avoids tiny concentrated hot spots
    heatmap = cm.jet(cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        pil_img.size, Image.Resampling.BICUBIC
    )
    blended = (
        np.array(pil_img, dtype=np.float32) * (1 - alpha)
        + np.array(heatmap_pil, dtype=np.float32) * alpha
    )
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


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
    pred = int(np.argmax(probs))

    fig = Figure(figsize=(10, 10))
    axes = fig.subplots(2, 2)

    for idx, (angle, cam, prob, disp) in enumerate(
        zip(constants.ROTATIONS, cams, probs, displays)
    ):
        img_overlay = make_overlay(disp, cam)
        ax = axes[idx // 2, idx % 2]
        marker = " *" if idx == pred else ""
        ax.imshow(img_overlay)
        ax.set_title(f"{angle}°{marker}  P(upright)={prob:.1%}", fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("gradcam_out"))
    args = parser.parse_args()
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
        out_path = args.out_dir / f"{img_path.stem}_gradcam.png"
        displays, probs, cams = compute_rotations(model, pil, preprocess, device)
        visualise(displays, probs, cams, img_path.name, out_path)


if __name__ == "__main__":
    main()
