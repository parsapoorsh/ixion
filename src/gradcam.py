"""GradCAM visualiser for the upright detector.

Shows which image regions drive the upright (P(upright)) prediction.
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
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from PIL import Image
from torchvision import transforms as T

import constants
from models.rotation import RotationModel
from utils.image_io import open_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

gpu_lock = threading.Lock()


def load_cached_stats(cache_dir: Path) -> tuple[tuple, tuple]:
    """Return (mean, std) from the first stats cache file found, or ImageNet defaults."""
    for path in sorted(cache_dir.glob("stats_*.json")):
        try:
            data = json.loads(path.read_text())
            return tuple(data["mean"]), tuple(data["std"])
        except Exception:
            continue
    return IMAGENET_MEAN, IMAGENET_STD


def load_model(checkpoint: Path, device: torch.device) -> RotationModel:
    model = RotationModel()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    model.to(device)
    model.train(False)
    return model


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


def visualise(
    model: RotationModel,
    pil_img: Image.Image,
    preprocess: T.Compose,
    device: torch.device,
    title: str,
    out_path: Path,
) -> None:
    display = T.Compose([
        T.Resize(constants.RESIZE_SIZE),
        T.CenterCrop(constants.INPUT_SIZE),
    ])(pil_img)

    with gpu_lock:
        with torch.no_grad():
            tensor = preprocess(pil_img).unsqueeze(0).to(device)  # type: ignore[union-attr]
            prob = float(model(tensor).sigmoid().item())

        t = preprocess(pil_img).unsqueeze(0).to(device)  # type: ignore[union-attr]
        cam = compute_gradcam(model, t)

    img_overlay = make_overlay(display, cam)

    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    ax.imshow(img_overlay)
    ax.set_title(f"P(upright) = {prob:.1%}", fontsize=11)
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
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        default=1,
        help="worker threads (GPU work is serialised; threads help I/O and overlay)",
    )
    args = parser.parse_args()
    args.images = [p for p in args.images if p.is_file()]

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    project_root = Path(__file__).parent.parent
    mean, std = load_cached_stats(project_root / "cache")

    preprocess = T.Compose([
        T.Resize(constants.RESIZE_SIZE),
        T.CenterCrop(constants.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    args.out_dir.mkdir(exist_ok=True)

    def run_one(img_path: Path) -> None:
        pil = open_rgb(img_path)
        out_path = args.out_dir / f"{img_path.stem}_gradcam.png"
        visualise(model, pil, preprocess, device, img_path.name, out_path)

    if args.thread > 1:
        with ThreadPoolExecutor(max_workers=args.thread) as pool:
            list(pool.map(run_one, args.images))
    else:
        for img_path in args.images:
            run_one(img_path)


if __name__ == "__main__":
    main()
