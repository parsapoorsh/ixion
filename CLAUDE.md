# Ixion

A photo rotation detector trained to sort ~500k personal photos for a self-hosted
[Immich](https://immich.app/) instance. The goal is a model that can reliably identify
whether a photo is upright — so misrotated images can be detected and corrected at
scale before or after ingestion into Immich.

## What it does

Trains an EfficientNet-B0 binary classifier (single output neuron, BCEWithLogitsLoss)
to predict P(upright) for any input image. At inference time, the four 90° rotations
are scored with the **same geometry as training** (eval transform, then `TF.rotate`
on the tensor). The view with the highest upright probability (`best_deg`) is the
correction; `margin` (top1 − top2) supports abstaining on ambiguous images.
Ship **EMA** weights. Training uses `torch.compile`.

Training data is ImageNet + COCO (general-purpose image corpora), which gives the
model diverse texture/content without needing a labelled rotation dataset. Each base
image yields 4 samples: 1 upright (label 1) + 3 rotated (label 0).

**Product metric:** `group4_acc` (validator) — for each base image, argmax over the
four rotation scores should pick the upright view. Prefer this over binary single-view
accuracy when comparing checkpoints. Multi-view ranking **loss** is not used in the
train loop (would multiply step cost); ranking is eval/infer only.

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
  config.py      — build_datasets(root) for the ImageNet+COCO layout
  models/
    rotation.py  — RotationModel (EfficientNet-B0 + LayerNorm/Dropout/Linear head)
    ema.py       — ModelEMA (exponential moving average shadow)
  utils/
    image_io.py  — PIL open with JXL plugin + EXIF orientation + truncated-image tolerance
    index.py     — recursive file scanner with compressed JSON cache
    logsetup.py  — timestamped log file + stderr handler
```

## Running

All scripts are run from the project root with `venv/bin/python src/<script>.py`.

**Train:**
```
venv/bin/python src/train.py \
    --data-root /path/to/datasets \
    --device cuda --batch-size 256 --num-workers 8 \
    --prefetch-factor 4 --seed 0
```

**Fine-tune (ImageNet-pretrained backbone, short schedule):**
```
venv/bin/python src/train.py \
    --data-root /path/to/datasets \
    --device cuda --batch-size 256 --num-workers 8 \
    --prefetch-factor 4 --seed 0 --fine-tune
```
`--fine-tune` implies `--pretrain` (timm ImageNet weights + ImageNet mean/std) and
sets a 5-epoch preset (peak `--lr 1e-3`, `--backbone-lr-mult 0.1`, `--warmup-epochs 0.5`,
`--weight-decay 0.05`). Any flag passed explicitly overrides the preset; `--pretrain`
alone uses the same pretrained init but keeps the from-scratch schedule defaults.

**Validate a checkpoint:**
```
venv/bin/python src/validator.py \
    --checkpoint checkpoints/epoch_010.pt \
    --data-root /path/to/datasets \
    --device cuda --batch-size 256 --num-workers 4 \
    --prefetch-factor 4 --weights ema
```
Logs binary loss/acc/per_class plus `group4_acc` and mean margins on correct vs
incorrect 4-way picks (same forward count as binary val).

**Browse a directory (optional margin abstain):**
```
venv/bin/python src/browse.py \
    --dir /path/to/photos \
    --checkpoint checkpoints/epoch_010.pt \
    --device cuda --abstain-margin 0.15
```

**GradCAM visualisation:**
```
venv/bin/python src/gradcam.py \
    --checkpoint checkpoints/epoch_010.pt \
    --device cuda --images examples/*.jpg --out-dir gradcam_out
```

## Key design decisions

- **From scratch by default.** `train.py` builds the backbone with `pretrained=False`.
  `--pretrain` opts into timm's ImageNet weights, and `--fine-tune` is a short-schedule
  preset over them (see Running).
- **Label-safe augmentation only.** No spatial flips/rotations (would corrupt the
  binary label). Train uses ColorJitter, RandomGrayscale, and RandomErasing after
  normalize; backbone uses stochastic depth (`drop_path_rate=0.1`).
- **Train–serve geometry match.** Train and multi-view infer both do
  transform → tensor rotate (not PIL-rotate then transform).
- **EMA weights are the production weights.** Raw training weights are also saved but
  EMA is what you should validate and ship.
- **Abstain on low margin.** Prefer leaving ambiguous images alone over a forced wrong
  90° correction (`RotationVote.margin` / browse `--abstain-margin`).
- **JXL support.** `pillow-jxl-plugin` is registered at import time in `utils/image_io.py`
  so both `.jxl` and standard formats work everywhere without any per-callsite setup.
- **Stats caching.** Dataset mean/std are computed once, cached to `cache/stats_*.json`,
  and reloaded on subsequent runs. Use `--invalidate-cache` to force a recompute.
- **Checkpoint portability.** `torch.compile` wraps parameters under `_orig_mod.*`;
  `save_checkpoint` unwraps to the raw module before saving so checkpoints load
  correctly whether or not `torch.compile` is active.

## Dataset paths

Pass the datasets root with `--data-root` on `train.py` / `validator.py`.
`config.build_datasets(root)` expands the standard ImageNet 2017 + COCO 2017
layout under it:

```
<root>/ILSVRC2017/Data/CLS-LOC/{train,test,val}/
<root>/coco/{train2017,test2017,val2017}/
```

## Dependencies

Python 3.12, PyTorch 2.11 + CUDA 12.8, torchvision, timm, Pillow + pillow-jxl-plugin.
See `requirements.txt` for pinned versions.
