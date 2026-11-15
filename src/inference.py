"""Four-view rotation scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import constants

# ---------------------------------------------------------------------------


class RotationVote(TypedDict):
    """Per-rotation upright probabilities and derived product fields.

    `probs` maps each rotation in degrees (0/90/180/270) to its four-view
    softmax probability (same geometry as training: transform first, then
    ``TF.rotate``).

    `best_deg` is argmax of `probs` - the counter-clockwise rotation
    that should be applied, matching ``TF.rotate``, to bring the image upright.

    `margin` is top1 - top2 softmax probability; small values mean ambiguous.
    """

    probs: dict[int, float]
    best_deg: int
    margin: float


def circular_mean_angle(probs: Sequence[float] | np.ndarray) -> float:
    """Weighted circular mean of the four rotation angles, in degrees."""
    angles = np.deg2rad(constants.ROTATIONS)
    values = np.asarray(probs, dtype=np.float64)
    x = float(np.sum(values * np.cos(angles)))
    y = float(np.sum(values * np.sin(angles)))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def stack_rotation_views(base: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Stack the four 90° tensor views of one transformed image.

    ``expand=False``: after center-crop the tensor is square, so 90° keeps
    HxW. ``fill=0`` is mean-gray in normalized space (same family as train).
    """
    views = [TF.rotate(base, deg, expand=False) for deg in constants.ROTATIONS]
    return torch.stack(views).to(device, non_blocking=True)


@torch.no_grad()
def predict_rotations(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    preprocess: T.Compose,
) -> RotationVote:
    """Score four orientations with the same geometry as training.

    Train applies resize/crop/normalize first, then ``TF.rotate`` on the
    tensor (see ``RotationPairDataset``). Inference matches that order so
    checkpoints are evaluated under the distribution they saw.
    """
    batch = stack_rotation_views(preprocess(img), device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
    probs = logits.float().softmax(dim=0).reshape(-1).cpu().numpy()

    order = np.argsort(probs)
    top1_i = int(order[-1])
    top2_i = int(order[-2])
    return {
        "probs": {deg: float(p) for deg, p in zip(constants.ROTATIONS, probs, strict=True)},
        "best_deg": int(constants.ROTATIONS[top1_i]),
        "margin": float(probs[top1_i] - probs[top2_i]),
    }
