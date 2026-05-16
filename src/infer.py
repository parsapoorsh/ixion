"""Inference with 4-rotation consistency ambiguity check.

The rotation task is equivariant: if we feed the same image rotated by r and the
model predicts p, then the implied original rotation is (p - r) mod 4. For an
unambiguous image, the implied rotation from all four views should agree. When
they disagree, the image is genuinely ambiguous (texture, circle, symmetric
content) and we abstain instead of forcing a confident wrong answer.

Designed for batch processing — runs all four rotations through the model in a
single forward pass on a (4, C, H, W) tensor.
"""

from collections import Counter
from dataclasses import dataclass

import pillow_jxl  # noqa: F401  ensure JXL handler is registered
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

import config
from builders import build_eval_transform


@dataclass
class Prediction:
    """Result of running the consistency-aware inference on one image."""
    rotation: int | None         # 0/90/180/270, or None when ambiguous
    confidence: str              # "confident" | "weak" | "ambiguous"
    votes: tuple[int, int, int, int]  # implied rotation from each view (degrees)
    entropy: float               # mean softmax entropy across the 4 views


def softmax_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


@torch.no_grad()
def predict_image(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Prediction:
    transform = build_eval_transform(mean=mean, std=std)
    views = torch.stack(
        [transform(TF.rotate(img, r, expand=False)) for r in config.ROTATIONS]
    )
    views = views.to(device, non_blocking=True)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(views)
        if hasattr(model, "temperature"):
            logits = logits / model.temperature
    probs = logits.float().softmax(dim=-1)
    preds = probs.argmax(dim=-1)  # shape [4], values in {0,1,2,3}

    # Convert per-view prediction back to the implied source rotation index.
    # If we rotated the image by r and the model says it's at p, then the
    # source was at (p - r) mod 4 quarters.
    implied = [
        (int(preds[i].item()) - i) % config.NUM_CLASSES
        for i in range(config.NUM_CLASSES)
    ]
    votes_deg = tuple(config.ROTATIONS[v] for v in implied)

    counts = Counter(implied)
    top, top_count = counts.most_common(1)[0]
    if top_count == config.NUM_CLASSES:
        conf = "confident"
    elif top_count == config.NUM_CLASSES - 1:
        conf = "weak"
    else:
        conf = "ambiguous"

    mean_entropy = float(softmax_entropy(probs).mean().item())
    return Prediction(
        rotation=config.ROTATIONS[top] if conf != "ambiguous" else None,
        confidence=conf,
        votes=votes_deg,
        entropy=mean_entropy,
    )
