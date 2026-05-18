"""Activation-maximisation wallpaper generator.

Maintains a large Fourier canvas and samples random 224x224 crops each
optimisation step — gradients flow back into the full canvas, so every
region develops coherent structure without ever rescaling the model input.

Usage:
    python src/wallpaper.py \\
        --checkpoint checkpoints/epoch_005.pt \\
        --device cuda:1 \\
        --width 1920 --height 1080 \\
        --out wallpaper.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import constants
from models.rotation import RotationModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(checkpoint: Path, device: torch.device) -> RotationModel:
    model = RotationModel()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    model.to(device)
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_freq_scale(h: int, w: int, decay: float, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    freqs = (fy[:, None].pow(2) + fx[None, :].pow(2)).sqrt()
    freqs[0, 0] = 1.0
    return freqs.pow(-decay).unsqueeze(0)  # (1, H, W//2+1)


def spectrum_to_image(
    coeffs: torch.Tensor,
    scale: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """Fourier coefficients → (1, 3, H, W) image in normalised space."""
    spectrum = torch.view_as_complex(coeffs) * scale
    img = torch.fft.irfftn(spectrum, s=(h, w))
    img = img / (img.std(dim=(-2, -1), keepdim=True) + 1e-8)
    return img.unsqueeze(0)


def random_crop(img: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = img.shape
    y = int(torch.randint(0, h - size + 1, ()).item())
    x = int(torch.randint(0, w - size + 1, ()).item())
    return img[:, :, y : y + size, x : x + size]


def random_jitter(img: torch.Tensor, max_shift: int) -> torch.Tensor:
    dy = int(torch.randint(-max_shift, max_shift + 1, ()).item())
    dx = int(torch.randint(-max_shift, max_shift + 1, ()).item())
    return torch.roll(img, shifts=(dy, dx), dims=(2, 3))


def denorm(img: torch.Tensor, mean: tuple, std: tuple) -> np.ndarray:
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    out = (img.cpu().float() * s + m).clamp(0, 1)
    return (out.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def generate(
    model: RotationModel,
    target_class: int,
    h: int,
    w: int,
    mean: tuple,
    std: tuple,
    device: torch.device,
    steps: int,
    lr: float,
    decay: float,
    jitter: int,
    seed: int | None,
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)

    crop_size = constants.INPUT_SIZE
    scale = make_freq_scale(h, w, decay, device)

    coeffs = torch.randn(3, h, w // 2 + 1, 2, device=device) * 0.01
    coeffs.requires_grad_(True)

    opt = torch.optim.Adam([coeffs], lr=lr)

    bar = tqdm(range(steps), unit="step")
    for _ in bar:
        opt.zero_grad()
        img = spectrum_to_image(coeffs, scale, h, w)
        crop = random_crop(img, crop_size)
        if jitter > 0:
            crop = random_jitter(crop, jitter)
        logits = model(crop)
        loss = -logits[0, target_class]
        loss.backward()
        opt.step()

        prob = logits.detach().softmax(dim=1)[0, target_class].item()
        bar.set_postfix(prob=f"{prob:.1%}")

    with torch.no_grad():
        img = spectrum_to_image(coeffs, scale, h, w)

    return denorm(img.detach(), mean, std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--target-class", type=int, default=0, choices=range(constants.NUM_CLASSES))
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--decay",
        type=float,
        default=1.5,
        help="1/f^decay frequency scaling; higher = smoother / lower-freq patterns",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=8,
        help="random pixel shift per crop step (0 to disable)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    img_arr = generate(
        model=model,
        target_class=args.target_class,
        h=args.height,
        w=args.width,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device=device,
        steps=args.steps,
        lr=args.lr,
        decay=args.decay,
        jitter=args.jitter,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_arr).save(args.out)


if __name__ == "__main__":
    main()
