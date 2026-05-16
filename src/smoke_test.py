"""Wiring sanity check on the example images.

Builds the model from pretrained weights (no training), runs a synthetic
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
from model import build_model

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device = %s", device)

    model = build_model(pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("model params = %.2f M", n_params)

    eg_dir = config.PROJECT_ROOT / "examples"
    sample_files = sorted(eg_dir.glob("*"))
    assert sample_files, f"no example images in {eg_dir}"

    train_tf = build_train_transform(
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
    )
    imgs = [train_tf(Image.open(p).convert("RGB")) for p in sample_files[:4]]
    batch = torch.stack(imgs).to(device)
    labels = torch.tensor([0, 1, 2, 3], device=device)

    model.train()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    log.info(
        "train forward/backward ok. loss=%.3f, logits shape=%s",
        loss.item(), tuple(logits.shape),
    )

    model.train(False)
    log.info("inference on examples/:")
    for p in sample_files:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            log.warning("%s: failed to open (%s)", p.name, exc)
            continue
        pred = predict_image(
            model=model,
            img=img,
            device=device,
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
        )
        log.info(
            "%-25s rot=%s conf=%-9s votes=%s entropy=%.3f",
            p.name, pred.rotation, pred.confidence, pred.votes, pred.entropy,
        )


if __name__ == "__main__":
    main()
