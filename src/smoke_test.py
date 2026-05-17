"""Wiring sanity check on the example images.

Builds the model with random init (no training), runs a synthetic
forward/backward through the training transforms, and runs the 4-rotation
consistency inference on the examples/ directory. Catches import errors,
shape bugs, JXL/PIL/transform issues, and obvious model-side problems before
launching a real training run.
"""

import logging

import torch

import config
import constants
from builders import build_train_transform
from utils.image_io import open_rgb
from infer import predict_image
from utils.logsetup import setup_logging
from models import RotationModel
from models.backbones import EfficientNetB0
from stats import get_stats

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("device = %s", config.DEVICE)

    backbone = EfficientNetB0()
    defaults = backbone.train_defaults
    model = RotationModel(
        backbone=backbone,
        pretrained=False,
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("model params = %.2f M", n_params)

    eg_dir = constants.PROJECT_ROOT / "examples"
    sample_files = sorted(eg_dir.glob("*"))
    assert sample_files, f"no example images in {eg_dir}"

    spec = config.DATASETS[0]
    channel_stats = get_stats(
        spec,
        input_size=defaults.input_size,
        resize_size=defaults.resize_size,
    )
    train_tf = build_train_transform(
        mean=channel_stats.mean,
        std=channel_stats.std,
        input_size=defaults.input_size,
        resize_size=defaults.resize_size,
    )
    batch = torch.stack(
        [train_tf(open_rgb(p)) for p in sample_files[:constants.NUM_CLASSES]]
    ).to(config.DEVICE)
    labels = torch.tensor(range(constants.NUM_CLASSES), device=config.DEVICE)

    model.train()
    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.bfloat16):
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    logger.info(
        "train forward/backward ok. loss=%.3f, logits shape=%s",
        loss.item(), tuple(logits.shape),
    )

    model.train(False)
    logger.info("inference on examples/:")
    for p in sample_files:
        try:
            img = open_rgb(p)
        except Exception as exc:
            logger.warning("%s: failed to open (%s)", p.name, exc)
            continue
        pred = predict_image(
            model=model,
            img=img,
            device=config.DEVICE,
            mean=channel_stats.mean,
            std=channel_stats.std,
            input_size=defaults.input_size,
            resize_size=defaults.resize_size,
        )
        logger.info(
            "%-25s rot=%s conf=%-9s (%.1f%%) votes=%s entropy=%.3f",
            p.name, pred.rotation, pred.confidence, pred.confidence_pct,
            pred.votes, pred.entropy,
        )


if __name__ == "__main__":
    main()
