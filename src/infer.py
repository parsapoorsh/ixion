"""Single-pass inference for the binary upright detector.

The model has one output neuron. A single forward pass yields a logit;
its sigmoid is the probability the image is upright (rotation 0°).

When that probability lands in an uncertain band (default 0.4-0.6) the
image is treated as genuinely ambiguous (uniform texture, circle,
symmetric content) and we abstain rather than force a yes/no answer.
"""

from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn

from builders import build_eval_transform


@dataclass
class Prediction:
    """Result of running inference on one image."""
    is_upright: bool | None      # True upright, False rotated, None when abstained
    confidence_pct: float        # probability the image is upright, in [0, 100]


@torch.no_grad()
def predict_image(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_size: int,
    resize_size: int,
    abstain_low: float = 0.4,
    abstain_high: float = 0.6,
) -> Prediction:
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

    if abstain_low <= prob <= abstain_high:
        is_upright = None
    else:
        is_upright = prob > 0.5

    return Prediction(
        is_upright=is_upright,
        confidence_pct=prob * 100.0,
    )
