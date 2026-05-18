"""Wiring sanity check on the example images.

Builds the model with random init (no training), runs a synthetic
forward/backward through the training transforms, runs the 4-rotation
consistency inference on the examples/ directory, and exercises the
validator code path in-process against the same examples folder.
Catches import errors, shape bugs, JXL/PIL/transform issues, and
obvious model-side problems before launching a real training run.

Device: auto-detected (cuda if available, else cpu). Smoke test, not a
training entry point - no CLI needed.
"""

import logging

import torch
from torch.utils.data import DataLoader

import config
import constants
from builders import build_eval_transform, build_train_transform
from datasets import ImageLoader, RotationPairDataset
from utils.image_io import open_rgb
from infer import predict_image
from utils.logsetup import setup_logging
from models import RotationModel
from models.backbones import EfficientNetB0
from stats import get_stats
from validator import run_validation

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device = %s", device)

    backbone = EfficientNetB0()
    defaults = backbone.train_defaults
    model = RotationModel(
        backbone=backbone,
        pretrained=False,
    ).to(device)
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
        num_workers=0,
        device=device,
    )
    train_tf = build_train_transform(
        mean=channel_stats.mean,
        std=channel_stats.std,
        input_size=defaults.input_size,
        resize_size=defaults.resize_size,
    )
    batch = torch.stack(
        [train_tf(open_rgb(p)) for p in sample_files[:constants.NUM_CLASSES]]
    ).to(device)
    labels = torch.tensor(range(constants.NUM_CLASSES), device=device)

    model.train()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
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
            device=device,
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

    eval_tf = build_eval_transform(
        mean=channel_stats.mean,
        std=channel_stats.std,
        input_size=defaults.input_size,
        resize_size=defaults.resize_size,
    )
    val_ds = RotationPairDataset(ImageLoader(sample_files), eval_tf)
    val_loader = DataLoader(
        val_ds,
        batch_size=min(constants.NUM_CLASSES * 2, len(val_ds)),
        shuffle=False,
        num_workers=0,
    )
    logger.info("validator: run_validation on %d examples (x4)", len(sample_files))
    val_loss, val_acc, per_class = run_validation(
        model,
        val_loader,
        device=device,
    )
    logger.info(
        "validator ok. loss=%.3f acc=%.3f per_class=%s",
        val_loss, val_acc, per_class,
    )


if __name__ == "__main__":
    main()
