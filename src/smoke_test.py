"""Wiring sanity check on the example images.

Builds the model with random init (no training), runs a synthetic
forward/backward through the training transforms, and runs the 4-rotation
consistency inference on the examples/ directory. Catches import errors,
shape bugs, JXL/PIL/transform issues, and obvious model-side problems before
launching a real training run.
"""

import logging

import torch
from PIL import Image

import config
from builders import build_train_transform
from infer import predict_image
from logsetup import setup_logging
from model import RotationModel

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("device = %s", config.DEVICE)

    model = RotationModel(pretrained=False).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("model params = %.2f M", n_params)

    eg_dir = config.PROJECT_ROOT / "examples"
    sample_files = sorted(eg_dir.glob("*"))
    assert sample_files, f"no example images in {eg_dir}"

    spec = config.DATASETS[0]
    train_tf = build_train_transform(mean=spec.mean, std=spec.std)
    imgs = [train_tf(Image.open(p).convert("RGB")) for p in sample_files[:4]]
    batch = torch.stack(imgs).to(config.DEVICE)
    labels = torch.tensor([0, 1, 2, 3], device=config.DEVICE)

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
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            logger.warning("%s: failed to open (%s)", p.name, exc)
            continue
        pred = predict_image(
            model=model,
            img=img,
            device=config.DEVICE,
            mean=spec.mean,
            std=spec.std,
        )
        logger.info(
            "%-25s rot=%s conf=%-9s votes=%s entropy=%.3f",
            p.name, pred.rotation, pred.confidence, pred.votes, pred.entropy,
        )


if __name__ == "__main__":
    main()
