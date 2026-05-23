# Ixion

A photo rotation detector trained to sort ~500k personal photos for a self-hosted
[Immich](https://immich.app/) instance. The goal is a model that can reliably identify
whether a photo is upright — so misrotated images can be detected and corrected at
scale before or after ingestion into Immich.

## What it does

Trains an EfficientNet-B0 binary classifier (single output neuron, BCEWithLogitsLoss)
to predict P(upright) for any input image. At inference time, the four 90° rotations
of an image are tested and the one with the highest upright probability is taken as
the correct orientation. The model is trained with EMA weights and compiled with
`torch.compile`.

Training data is ImageNet + COCO (general-purpose image corpora), which gives the
model diverse texture/content without needing a labelled rotation dataset. Each base
image yields 4 samples: 1 upright (label 1) + 3 rotated (label 0). The 1:3 class
imbalance is corrected via `pos_weight=3.0` in the loss.

## Source layout

```
src/
  train.py       — training loop (cosine-with-warmup LR, gradient clipping, checkpoint saving)
  validator.py   — standalone CLI to eval a checkpoint against the val split
  infer.py       — single-image inference (returns P(upright) as a float)
  gradcam.py     — visualise model attention at all 4 rotations via GradCAM
  datasets.py    — DatasetSpec, ImageLoader, RotationPairDataset, build_dataset
  builders.py    — torchvision transform factories (train/eval)
  stats.py       — dataset mean/std computation with JSON caching
  constants.py   — paths, sizes, rotation list; no imports from config so models can depend on it safely
  config.py      — dataset roots and sampling config
  models/
    rotation.py  — RotationModel (EfficientNet-B0 + LayerNorm/Dropout/Linear head)
    ema.py       — ModelEMA (exponential moving average shadow)
  utils/
    image_io.py  — PIL open with JXL plugin + truncated-image tolerance
    index.py     — recursive file scanner with compressed JSON cache
    logsetup.py  — timestamped log file + stderr handler
```

## Running

All scripts are run from the project root with `venv/bin/python src/<script>.py`.

**Train:**
```
venv/bin/python src/train.py \
    --device cuda --batch-size 256 --num-workers 8 \
    --prefetch-factor 4 --seed 0
```

**Validate a checkpoint:**
```
venv/bin/python src/validator.py \
    --checkpoint checkpoints/epoch_010.pt \
    --device cuda --batch-size 256 --num-workers 4 \
    --prefetch-factor 4 --weights ema
```

**GradCAM visualisation:**
```
venv/bin/python src/gradcam.py \
    --checkpoint checkpoints/epoch_010.pt \
    --device cuda --images examples/*.jxl --out-dir gradcam_out
```

## Key design decisions

- **No pretrained weights.** The model is trained from scratch; ImageNet weights
  encode strong orientation priors that would bias the classifier.
- **Only photometric augmentation.** Spatial flips/rotations during training would
  corrupt the binary label, so augmentation is limited to ColorJitter and RandomGrayscale.
- **EMA weights are the production weights.** Raw training weights are also saved but
  EMA is what you should validate and ship.
- **JXL support.** `pillow-jxl-plugin` is registered at import time in `utils/image_io.py`
  so both `.jxl` and standard formats work everywhere without any per-callsite setup.
- **Stats caching.** Dataset mean/std are computed once, cached to `cache/stats_*.json`,
  and reloaded on subsequent runs. Use `--invalidate-cache` to force a recompute.
- **Checkpoint portability.** `torch.compile` wraps parameters under `_orig_mod.*`;
  `save_checkpoint` unwraps to the raw module before saving so checkpoints load
  correctly whether or not `torch.compile` is active.

## Dataset paths

Configured in `src/config.py`. Currently points to ImageNet 2017 and COCO 2017 at:
```
/media/poweruser/data/2-stuff/datasets/ILSVRC2017/...
/media/poweruser/data/2-stuff/datasets/coco/...
```

## Dependencies

Python 3.12, PyTorch 2.11 + CUDA 12.8, torchvision, timm, Pillow + pillow-jxl-plugin.
See `requirements.txt` for pinned versions.
