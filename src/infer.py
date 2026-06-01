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
    P(upright) for the image rotated by that amount.

    `normalized_probs` is the same mapping after L1-normalisation (divide
    by the sum), so the four values form a proper probability distribution.
    When one rotation dominates (e.g. 0.78 vs ~0.00) its normalised share
    will be close to 1.0 even though the raw sigmoid never reaches it.

    `angle` is the weighted circular mean of the raw `probs` votes,
    expressed as the rotation (clockwise, in degrees) that should bring
    the image upright.
    """

    probs: dict[int, float]
    normalized_probs: dict[int, float]
    angle: float


@torch.no_grad()
def predict_image(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> float:
    transform = build_eval_transform(
        mean=mean,
        std=std,
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
) -> RotationVote:
    transform = build_eval_transform(
        mean=mean,
        std=std,
    )
    views = [img.rotate(-deg, expand=True) for deg in ROTATIONS]
    batch = torch.stack([transform(v) for v in views]).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
    probs = logits.float().sigmoid().squeeze(-1).cpu().numpy()

    prob_sum = probs.sum()
    norm = probs / prob_sum if prob_sum > 0 else np.ones_like(probs) / len(probs)

    angles = np.deg2rad(ROTATIONS)
    x = float(np.sum(probs * np.cos(angles)))
    y = float(np.sum(probs * np.sin(angles)))
    angle = np.rad2deg(np.arctan2(y, x)) % 360.0

    return RotationVote(
        probs={deg: float(p) for deg, p in zip(ROTATIONS, probs)},
        normalized_probs={deg: float(p) for deg, p in zip(ROTATIONS, norm)},
        angle=angle,
    )
