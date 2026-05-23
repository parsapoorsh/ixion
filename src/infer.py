"""Single-pass inference for the binary upright detector.

The model has one output neuron. A single forward pass yields a logit;
its sigmoid is the probability the image is upright (rotation 0°).

When that probability lands in an uncertain band (default 0.4-0.6) the
image is treated as genuinely ambiguous (uniform texture, circle,
symmetric content) and we abstain rather than force a yes/no answer.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

from builders import build_eval_transform
from constants import ROTATIONS


@dataclass
class RotationVote:
    """Per-rotation upright probabilities and the circular-mean angle.

    `probs` maps each rotation in degrees (0/90/180/270) to the model's
    P(upright) for the image rotated by that amount. `angle` is the
    weighted circular mean of those votes, expressed as the rotation
    (clockwise, in degrees) that should bring the image upright.
    """

    probs: dict[int, float]
    angle: float


@torch.no_grad()
def predict_image(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_size: int,
    resize_size: int,
) -> float:
    transform = build_eval_transform(
        mean=mean,
        std=std,
        input_size=input_size,
        resize_size=resize_size,
    )
    tensor = transform(img).unsqueeze(0).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logit = model(tensor)
    prob = float(logit.float().sigmoid().item())

    return prob


@torch.no_grad()
def predict_rotations(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_size: int,
    resize_size: int,
) -> RotationVote:
    transform = build_eval_transform(
        mean=mean,
        std=std,
        input_size=input_size,
        resize_size=resize_size,
    )
    views = [img.rotate(-deg, expand=True) for deg in ROTATIONS]
    batch = torch.stack([transform(v) for v in views]).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
    probs = logits.float().sigmoid().squeeze(-1).cpu().numpy()

    angles = np.deg2rad(ROTATIONS)
    x = float(np.sum(probs * np.cos(angles)))
    y = float(np.sum(probs * np.sin(angles)))
    angle = np.rad2deg(np.arctan2(y, x)) % 360.0

    return RotationVote(
        probs={deg: float(p) for deg, p in zip(ROTATIONS, probs)},
        angle=angle,
    )
