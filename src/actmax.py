"""Activation-maximisation visualiser for the rotation model.

Synthesises one image per rotation class by optimising a learnable image
to maximise the model logit for that class.

Usage:
    python src/actmax.py \\
        --checkpoint checkpoints/epoch_005.pt \\
        --device cuda:1 \\
        --out-dir actmax_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

import constants
from models.rotation import RotationModel

LABELS = ["0°", "90°", "180°", "270°"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_cached_stats(cache_dir: Path) -> tuple[tuple, tuple]:
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    return (
        (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
        + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    )


def denorm(img: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    out = img.cpu().float().numpy() * std_arr + mean_arr
    return (out.clip(0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def maximise(
    model: RotationModel,
    target_class: int,
    mean: tuple,
    std: tuple,
    device: torch.device,
    steps: int,
    lr: float,
    lambda_tv: float,
    lambda_l2: float,
    blur_every: int,
) -> np.ndarray:
    """Return a uint8 HxWx3 array that maximises logit[target_class]."""
    size = constants.INPUT_SIZE
    img = torch.randn(1, 3, size, size, device=device) * 0.01
    img.requires_grad_(True)

    opt = torch.optim.Adam([img], lr=lr)

    for step in range(steps):
        opt.zero_grad()
        logits = model(img)
        loss = (
            -logits[0, target_class]
            + lambda_tv * tv_loss(img)
            + lambda_l2 * img.pow(2).mean()
        )
        loss.backward()
        opt.step()

        if blur_every > 0 and (step + 1) % blur_every == 0:
            with torch.no_grad():
                img.data = gaussian_blur(img.data, kernel_size=[3], sigma=[0.5])

    with torch.no_grad():
        prob = model(img).softmax(dim=1)[0, target_class].item()
    print(f"  class {LABELS[target_class]:5s}  prob={prob:.1%}")

    return denorm(img.squeeze(0).detach(), mean, std)


def save_grid(images: list[np.ndarray], out_path: Path) -> None:
    fig = Figure(figsize=(8, 8))
    axes = fig.subplots(2, 2)

    for idx, img_arr in enumerate(images):
        ax = axes[idx // 2, idx % 2]
        ax.imshow(img_arr)
        ax.set_title(LABELS[idx], fontsize=12)
        ax.axis("off")

    fig.suptitle("Activation Maximisation", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--out-dir", type=Path, default=Path("actmax_out"))
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=1e-3)
    parser.add_argument("--lambda-l2", type=float, default=1e-4)
    parser.add_argument(
        "--blur-every",
        type=int,
        default=4,
        help="apply Gaussian blur every N steps (0 to disable)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    project_root = Path(__file__).parent.parent
    mean, std = load_cached_stats(project_root / "cache")
    print(f"normalisation  mean={mean}  std={std}")

    args.out_dir.mkdir(exist_ok=True)

    images: list[np.ndarray] = []
    for cls_idx in range(constants.NUM_CLASSES):
        img_arr = maximise(
            model,
            cls_idx,
            mean,
            std,
            device,
            steps=args.steps,
            lr=args.lr,
            lambda_tv=args.lambda_tv,
            lambda_l2=args.lambda_l2,
            blur_every=args.blur_every,
        )
        images.append(img_arr)
        label_safe = LABELS[cls_idx].replace("°", "deg")
        Image.fromarray(img_arr).save(args.out_dir / f"actmax_{label_safe}.png")

    save_grid(images, args.out_dir / "actmax_grid.png")


if __name__ == "__main__":
    main()
