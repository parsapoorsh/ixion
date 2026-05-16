# CLAUDE.md

Project context, design decisions, and the user's coding preferences. Read
this first.

## What this project is

An image rotation detector that classifies each input as one of 4 rotations
(0°/90°/180°/270°) and abstains on genuinely ambiguous images (uniform
textures, circles, symmetric content). Built to pre-sort ~500k personal
photos for a self-hosted Immich instance — so it's batch-processing, not
real-time.

## Hardware target

- **Training**: RTX 5060 Ti 16GB (Blackwell, sm_120). Requires PyTorch with
  CUDA 12.8+ wheels. BF16 mixed precision is the native fast path.
- **Display GPU**: GTX 1660 SUPER 6GB. Useful for driving the desktop so the
  5060 Ti is 100% dedicated to training. **Do not** try to use it for
  training/DDP — Turing has no BF16 tensor cores and the 6GB would bottleneck
  the per-GPU batch size on the faster card.
- **Inference VRAM budget**: ≤ 250 MB. Comfortable via ONNX Runtime at
  deployment time; PyTorch's CUDA context alone eats ~180 MB so dev
  measurements look higher than the eventual deployment.

## Model architecture

- **Backbone**: `convnextv2_tiny` from timm, **trained from scratch** (no
  pretrained weights). 28M params. Architecture chosen over EfficientNet-B1
  once the user OK'd the size: better capacity for ambiguous edge cases.
  The timm model id still references the `.fcmae_ft_in22k_in1k` variant
  because we want that architecture; `pretrained=False` skips the weights.
- **Head**: `AdaptiveAvgPool → LayerNorm(768) → Dropout(0.2) → Linear(768, 4)`.
- **Temperature buffer**: `register_buffer("temperature", ones(1))` for
  post-hoc calibration. Frozen during training; fit on val set after.
- **Single LR**: backbone and head learn at the same rate (random init, no
  reason for a layer-wise split). Earlier the recipe used backbone 2e-4 /
  head 2e-3; that ratio is now gone.

## Training recipe

Single phase, 100 epochs from random init. No frozen warm-up phase, no
noise-filtering pass — both were over-engineered.

- Batch 128, BF16 autocast, AdamW (β=0.9/0.999), weight_decay 0.05
- LR 1e-3, cosine schedule with 5-epoch linear warmup
- Label smoothing 0.1 — handles the ~1% wrong-orientation images in ImageNet
- EMA decay 0.9999, evaluated and checkpointed off the EMA weights
- Gradient clip at norm=1.0
- Wall-clock estimate: ~5 days for 100 epochs on the 5060 Ti (extrapolated
  from the earlier ~9h-for-8-epochs fine-tune estimate; verify on first run
  and adjust)

### Augmentations — be careful here

**Safe** (preserve rotation label):
- ColorJitter, RandomGrayscale, RandomErasing, Normalize, mild RandomCrop

**Forbidden** (corrupt the label):
- RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
  RandomPerspective, RandomAffine — any spatial op that re-orients pixels.
  Pay attention if you add new augmentations.

## Data pipeline

- **Sources**: ImageNet ILSVRC2017 (`.../CLS-LOC/{train,val}`) + COCO 2017
  (`.../coco/{train2017,val2017}`).
- **Format**: JPEG-XL (`.jxl`). `import pillow_jxl` auto-registers a PIL
  handler so `PIL.Image.open` decodes transparently. The directories may
  also contain other formats — do **not** hard-code `.jxl` anywhere, just
  read every file and let PIL fail at decode if anything's wrong.
- **File index**: scanned once per (split, dataset) and cached as JSON under
  `cache/`. CLI flag `--invalidate-cache` deletes the JSONs.
- **Rotation pairs**: each image yields 4 samples deterministically.
  `len(dataset) = 4 * n_images`, sample `i` → `(img_idx, rot_idx) = divmod(i, 4)`.
  Guarantees balanced classes and full coverage every epoch.
- **Per-dataset normalization**: each `DatasetSpec` can carry an explicit
  `mean`/`std`, or leave them as `None`. When `None`, `stats.get_stats`
  samples ~5000 train images for that spec, computes per-channel mean/std
  in float64, and caches the result as JSON under `cache/stats_<name>.json`.
  Default `DATASETS` in `config.py` pass `None` for both shipped datasets;
  `IMAGENET_MEAN`/`IMAGENET_STD` remain in `config.py` so a spec can be
  pinned to fixed constants when wanted. `build_*_dataset` calls
  `get_stats(spec)` per spec, then `ConcatDataset`s them.

## Ambiguity detection — the key insight

Rotation is **equivariant**: if the model predicts rotation `p` for image `I`,
then on `rotate(I, r)` it should predict `(p + r) mod 360`. We use this:

For each image, run all 4 rotations through the model. From each view `i ∈
{0..3}`, the implied source rotation is `(pred_i - i) mod 4`. If all four
views agree → confident. 3 agree → weak. ≤2 agree → ambiguous, abstain.

This is a structural test, not a confidence score, and is far more reliable
than entropy thresholding for the grass/circle case. It costs 4× inference,
which is fine for batch processing 500k photos.

Entropy + temperature scaling exists as a calibrated tiebreaker.

## File layout (src/)

```
spec.py        DatasetSpec dataclass (name, train, val, mean, std)
config.py      DEVICE, PROJECT_ROOT, CHECKPOINT_DIR, INDEX_CACHE_DIR,
               IMAGENET_MEAN/STD, DATASETS (instances), sizes, ROTATIONS,
               build_train_transform, build_eval_transform
index.py       index_files(name, roots) + clear_cache() — filesystem scan with
               on-disk JSON cache
datasets.py    ImageLoader (PIL by index) + RotationPairDataset (4× wrapper)
stats.py       get_stats(spec) — per-dataset channel mean/std, sampled and
               cached, or returned from the spec when set explicitly
builders.py    build_train_dataset, build_val_dataset (dataset wiring only;
               transform builders live in config.py)
model.py       RotationModel (ConvNeXt-V2-Tiny + head + temperature)
ema.py         ModelEMA — deepcopy + per-step weight averaging
logsetup.py    setup_logging() — shared logger config
calibration.py gather_logits, fit_temperature (LBFGS), expected_calibration_error
train.py       TrainConfig, ValidationResult, run_validation,
               save_checkpoint, cosine_with_warmup, set_lrs, train_one_run, main
calibrate.py   CLI: load checkpoint → gather logits → fit T → save
infer.py       Prediction dataclass + predict_image (4-rotation consistency)
smoke_test.py  Wiring sanity test on examples/
```

**Dependency direction**: `spec → config → index → datasets → builders →
{train, calibrate, infer, smoke_test}`. No cycles.

**Imports are flat** (`from x import y`, not `from .x import y`). Run from
project root with `venv/bin/python src/<file>.py`. There is no `__init__.py`.

## Entry points

```
venv/bin/python src/smoke_test.py
venv/bin/python src/train.py    [--epochs N] [--batch-size N] [--num-workers N] [--invalidate-cache]
venv/bin/python src/calibrate.py [--checkpoint PATH] [--batch-size N] [--num-workers N] [--invalidate-cache]
```

## User's coding preferences (firm)

Strict rules — apply without asking:

1. **No underscore-prefixed names** at module level. Reserved only for
   private class methods when the rest of the styling allows it.
2. **All imports at the top of the file.** Never `import x` inside a function.
3. **No `global x`**. If you need it, the code is wrong.
4. **Small, focused git commits.** Big commits get rejected. One change per
   commit. Always use the configured `git config user.*` — don't override.
5. **Multi-line function signatures** when there are many args/kwargs.
6. **Use `logging`, never `print`.** Module-level variable is named `logger`
   (not `log`). Setup goes through `logsetup.setup_logging()`.
7. **Use `time.monotonic()`** for elapsed-time measurements, not `time.time()`.
8. **Dataclasses, not dicts**, for structured returns.
9. **No defaults for things the caller should choose.** e.g. `mean`/`std` are
   required kwargs, not defaulted to ImageNet constants.
10. **No extension whitelists** when scanning files unless we have a concrete
    reason — let PIL fail at decode time.
11. **Centralize cross-cutting state in `config.py`**: device, paths,
    DATASETS, sizes. Don't redeclare `torch.device(...)` in modules.
12. **`help=` on every argparse argument.**
13. **Use `model.train(False)`** for inference mode (the security hook flags
    the other spelling).

### Splitting rules (learned the hard way)

- **Split files by responsibility / by the work they do**, not by line count.
- **Don't extract single-use helpers into their own files.** If only
  `train.py` calls `save_checkpoint`, it lives in `train.py`. Earlier I
  extracted `checkpoint.py`, `evaluate.py`, `schedule.py` and was told to
  put them back. They're now inlined.
- `DatasetSpec` is its own file (`spec.py`) because it's a shared shape
  with no logic, used across modules.

### Anti-patterns the user called out

- Calling `git add -A` and accidentally committing fixtures/examples.
- Re-authoring history with overrides when `git config user.*` is already
  correct — just commit.
- Factory functions like `build_model()` when the class constructor is already
  fine: `RotationModel(pretrained=True)`.
- Keeping unused dead helpers around "just in case" (`predict_path`,
  `load_model_for_inference`, `build_model` — all removed).

## Communication style

- The user expects **honest pushback**, not agreement. If a suggestion is
  wrong, say so with reasoning. They asked "is deepcopy normal or should you
  use pytorch itself" — the right answer was "yes deepcopy is normal, but
  pytorch has AveragedModel which also uses deepcopy" with the trade-off.
- The user **does not want clarifying questions** — make the reasonable call
  and continue, they'll redirect.
- Keep updates short. End-of-turn summary: 1–2 sentences max.

## Gotchas

- The user has **two GPUs** but only the 5060 Ti is useful for training. The
  1660 SUPER is just for driving the display.
- **`ImageList` was renamed to `ImageLoader`** — don't go back.
- The cache directory is `cache/` (no leading dot). Earlier it was `.cache/`
  which collided with the gitignore Python template.
- `pillow-jxl-plugin` registers on import as a side effect. The `import
  pillow_jxl` line in `datasets.py` and `infer.py` is **not unused** — keep
  the `# noqa: F401` comment.
- `RandomResizedCrop` with low scale (< ~0.7) can crop out all orientation
  cues. Keep crops mild.

## Outstanding work (as of last session)

- PyTorch is being installed manually by the user (CUDA 12.8 wheels for
  Blackwell). When done, run `smoke_test.py` to verify the pipeline before
  launching a real training run.
- `timm` is not installed yet (waits on torch). `model.py` imports it
  top-level, so smoke test will fail until timm is installed.
- No checkpoint exists. First real run goes through `train.py` end-to-end.
- ONNX export is planned but deliberately deferred.
- Ambiguity-validation set (DTD textures, COCO circular subjects, Perlin
  noise) is not yet built. Needed for tuning the 4-rotation consistency
  threshold + temperature scaling.
