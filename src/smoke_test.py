"""Wiring sanity check on the example images.

Builds the model with random init (no training), runs a synthetic
forward/backward through the training transforms, runs the 4-rotation
consistency inference on the examples/ directory, and exercises the
validator + ambiguity-scan code paths in-process against the same
examples folder. Catches import errors, shape bugs, JXL/PIL/transform
issues, and obvious model-side problems before launching a real
training run.
"""

import logging

import torch
from torch.utils.data import DataLoader

import config
import constants
from async_validator import run_validation, scan_dataset
from builders import build_eval_transform, build_train_transform
from datasets import DatasetSpec, ImageLoader, RotationPairDataset
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

    smoke_spec = DatasetSpec(
        name="smoke_examples",
        train=eg_dir,
        val=eg_dir,
        mean=channel_stats.mean,
        std=channel_stats.std,
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
        device=config.DEVICE,
        autocast_dtype=torch.bfloat16,
    )
    logger.info(
        "validator ok. loss=%.3f acc=%.3f per_class=%s",
        val_loss, val_acc, per_class,
    )

    logger.info("ambiguity scan over examples/ ...")
    paths, n_weak, n_ambiguous = scan_dataset(
        model,
        dataset_specs=[smoke_spec],
        device=config.DEVICE,
        batch_size=min(8, len(sample_files)),
        recipe_input_size=defaults.input_size,
        recipe_resize_size=defaults.resize_size,
    )
    logger.info(
        "ambiguity scan ok. kept=%d weak=%d ambiguous=%d",
        len(paths), n_weak, n_ambiguous,
    )


if __name__ == "__main__":
    main()
