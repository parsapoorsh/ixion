"""Single-pass inference for the binary upright detector.

The model has one output neuron. A single forward pass yields a logit;
its sigmoid is the probability the image is upright (rotation 0°).

At product time, score all four 90° views and take argmax of P(upright).
When the top-1 vs top-2 margin is small, prefer abstaining rather than
forcing a correction (uniform texture, circles, symmetric content).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

from builders import build_eval_transform
from constants import ROTATIONS


@dataclass
class RotationVote:
    """Per-rotation upright probabilities and derived product fields.

    `probs` maps each rotation in degrees (0/90/180/270) to the model's
    P(upright) for the image rotated by that amount (same geometry as
    training: transform first, then TF.rotate).

    `normalized_probs` is L1-normalised `probs` so the four values sum to 1.

    `best_deg` is argmax of raw `probs` — the rotation that should be
    applied (clockwise, matching training) to bring the image upright.

    `margin` is top1 − top2 raw probability; small values mean ambiguous.

    `angle` is the weighted circular mean of the raw `probs` votes
    (secondary; prefer `best_deg` for discrete 90° correction).
    """

    probs: dict[int, float]
    normalized_probs: dict[int, float]
    best_deg: int
    margin: float
    angle: float

    def should_abstain(self, threshold: float) -> bool:
        """True when top-1 vs top-2 margin is below *threshold*."""
        return self.margin < threshold


def _vote_from_probs(probs: np.ndarray) -> RotationVote:
    prob_sum = float(probs.sum())
    if prob_sum > 0:
        norm = probs / prob_sum
    else:
        norm = np.ones_like(probs) / len(probs)

    order = np.argsort(probs)
    top1_i = int(order[-1])
    top2_i = int(order[-2]) if len(order) > 1 else top1_i
    best_deg = int(ROTATIONS[top1_i])
    margin = float(probs[top1_i] - probs[top2_i])

    angles = np.deg2rad(ROTATIONS)
    x = float(np.sum(probs * np.cos(angles)))
    y = float(np.sum(probs * np.sin(angles)))
    angle = float(np.rad2deg(np.arctan2(y, x)) % 360.0)

    return RotationVote(
        probs={deg: float(p) for deg, p in zip(ROTATIONS, probs)},
        normalized_probs={deg: float(p) for deg, p in zip(ROTATIONS, norm)},
        best_deg=best_deg,
        margin=margin,
        angle=angle,
    )


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
    """Score four orientations with the same geometry as training.

    Train applies resize/crop/normalize first, then ``TF.rotate`` on the
    tensor (see ``RotationPairDataset``). Inference matches that order so
    checkpoints are evaluated under the distribution they saw.
    """
    transform = build_eval_transform(
        mean=mean,
        std=std,
    )
    base = transform(img)
    # expand=False: after center-crop the tensor is square; 90° keeps H×W.
    # fill=0 is mean-gray in normalized space (same family as train).
    views = [
        TF.rotate(base, deg, expand=False)
        for deg in ROTATIONS
    ]
    batch = torch.stack(views).to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
    probs = logits.float().sigmoid().reshape(-1).cpu().numpy()

    return _vote_from_probs(probs)
