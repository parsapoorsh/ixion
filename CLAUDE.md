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

- **Training + display**: RTX 5060 Ti 16GB (Blackwell, sm_120). Same card
  drives the desktop, so expect ~0.5–1 GB of VRAM permanently reserved by
  the compositor — leave headroom when sizing batches. Requires PyTorch
  with CUDA 12.8+ wheels. BF16 mixed precision is the native fast path.
- **Second GPU**: a GTX 1660 SUPER 6GB is physically installed alongside
  the 5060 Ti but is currently not used for anything. **Do not** wire it
  into training/DDP — Turing has no BF16 tensor cores and the 6GB would
  bottleneck the per-GPU batch size on the faster card. The realistic
  future use is as a display-offload card so the 5060 Ti's VRAM is
  freed up for compute.
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
- **Format**: JPEG-XL (`.jxl`). All image opening goes through
  `utils.image_io.open_rgb`, which registers the `pillow_jxl` handler and
  enables truncated-image tolerance in one place. Don't call
  `PIL.Image.open` directly — use `open_rgb` so new modules can't silently
  miss either side effect. Directories may also contain other formats; do
  **not** hard-code `.jxl` anywhere, let PIL fail at decode if anything's
  wrong.
- **File index**: scanned once per (split, dataset). The result is collapsed
  into a path *trie* (nested dict, directories as subdicts, files as null
  leaves) and written through `lzma` to `cache/index_<name>.json.xz` — shared
  prefixes deduplicate and the file shrinks by roughly an order of magnitude
  vs. the flat list. `index_files` returns a `PathList`, a sequence view
  that walks the trie on `__getitem__` so the full path list never has to be
  materialised in memory. `ImageLoader.files` is typed `Sequence[Path]`, so
  a `PathList` (or any other sequence) drops in. CLI flag
  `--invalidate-cache` deletes the cache files (and stats cache too).
- **Rotation pairs**: each image yields 4 samples deterministically.
  `len(dataset) = 4 * n_images`, sample `i` → `(img_idx, rot_idx) = divmod(i, 4)`.
  Guarantees balanced classes and full coverage every epoch.
- **Per-dataset normalization**: each `DatasetSpec` can carry an explicit
  `mean`/`std`, or leave them as `None` (the default). When `None`,
  `stats.get_stats` samples `config.STATS_SAMPLE_FRACTION` (5% default) of
  the train split with seed `config.STATS_SAMPLE_SEED`, computes per-channel
  mean/std, and caches the result as JSON under `cache/stats_<name>.json`.
  Default `DATASETS` in `config.py` pass `None` for both shipped datasets;
  `IMAGENET_MEAN`/`IMAGENET_STD` remain in `config.py` so a spec can be
  pinned to fixed constants when wanted. `build_train_dataset` /
  `build_val_dataset` (in `datasets.py`) call `get_stats(spec)` per spec,
  then `ConcatDataset` the per-spec parts.
- **Stats computation** is GPU-accelerated: a `StatsDataset` decodes images
  via the shared `build_eval_preprocess()` pipeline (Resize → CenterCrop
  → ToTensor, no Normalize — we're computing the normaliser), workers
  parallelise decode, the per-batch sum/sum-of-squares reduces on
  `config.DEVICE` in float32, and accumulation happens on CPU in float64
  (consumer GPUs are ~1/64 speed for float64). The sample is sorted by
  path *after* sampling so HDD-backed datasets read in on-disk locality
  order — without this, random scatter across a 1.2M-file tree thrashes
  seeks once the OS page cache runs out (~700 img/s → ~150 img/s in the
  wild). `prefetch_factor=4` keeps each worker's queue deep enough to mask
  remaining I/O latency. tqdm shows progress in images/s with a
  per-image `pbar.update(b)`. Changing `STATS_SAMPLE_FRACTION` needs
  `--invalidate-cache`; the cache file doesn't track which fraction
  produced it. Stats are computed *after* the resize, so the std comes
  out higher than canonical ImageNet (~0.27 vs 0.229) — that's expected
  and consistent with the input distribution the model actually sees.
  Don't "fix" by hardcoding canonical numbers.
- **Worker count**: `config.NUM_WORKERS` (default 8) is the single source
  of truth for DataLoader workers across `train`, `calibrate`, and
  `stats`. Every CLI exposes `--num-workers` to override per-run. Too many
  workers make HDDs slower, not faster — drop the number if the bar
  stalls and CPU usage falls.

## Ambiguity detection — the key insight

Rotation is **equivariant**: if the model predicts rotation `p` for image `I`,
then on `rotate(I, r)` it should predict `(p + r) mod 360`. We use this:

For each image, run all 4 rotations through the model. From each view `i ∈
{0..3}`, the implied source rotation is `(pred_i - i) mod 4`. If all four
views agree → confident. 3 agree → weak. ≤2 agree → ambiguous, abstain.

This is a structural test, not a confidence score, and is far more reliable
than entropy thresholding for the grass/circle case. It costs 4× inference,
which is fine for batch processing 500k photos.

`Prediction.confidence_pct` carries a continuous companion to the
categorical label: the maximum of the rotation-equivariant posterior. For
each candidate source rotation `s`, average the model's probability across
views (view `i`'s probability for source `s` is `probs[i, (s + i) % 4]`),
take the max, scale to [0, 100]. Strictly more informative than the
4-bucket vote share — use it for ranking or thresholding when the
categorical confident/weak/ambiguous label is too coarse. The categorical
label still drives the abstention decision (whether `rotation` is None).

Entropy + temperature scaling exists as a calibrated tiebreaker.

## File layout (src/)

ML-domain code lives at `src/` root; general-purpose helpers live under
`src/utils/` (a real package, with `__init__.py`).

```
config.py         DEVICE, IMAGENET_MEAN/STD, DATASETS (instances),
                  INPUT_SIZE/RESIZE_SIZE, STATS_SAMPLE_FRACTION/SEED,
                  NUM_WORKERS
constants.py      ROTATIONS, NUM_CLASSES, PROJECT_ROOT, CHECKPOINT_DIR,
                  INDEX_CACHE_DIR — domain constants and static paths
                  depended on by datasets/model/train/infer/utils without
                  going through config
datasets.py       DatasetSpec dataclass (name, train, val, mean=None, std=None),
                  ImageLoader (PIL by index), RotationPairDataset (4× wrapper),
                  build_train_dataset / build_val_dataset (call get_stats + the
                  transforms from builders, then ConcatDataset the per-spec parts)
stats.py          get_stats(spec) + compute_stats (GPU-accelerated via a
                  DataLoader + StatsDataset) + sample_files (sorted by path)
                  + CLI for ad-hoc stats runs. Sampled mean/std cached as
                  cache/stats_<name>.json; explicit spec.mean/spec.std
                  bypasses the scan entirely.
builders.py       build_train_transform, build_eval_preprocess (Resize →
                  CenterCrop → ToTensor; shared with StatsDataset),
                  build_eval_transform (preprocess + Normalize). Transforms
                  only — no stats / index / DatasetSpec dependency.
model.py          RotationModel (ConvNeXt-V2-Tiny + head + temperature)
ema.py            ModelEMA — deepcopy + per-step weight averaging
infer.py          Prediction dataclass (rotation, confidence categorical,
                  confidence_pct continuous, votes, entropy) +
                  predict_image (4-rotation consistency + posterior averaging)
train.py          TrainConfig, ValidationResult, run_validation,
                  save_checkpoint, cosine_with_warmup, set_lr, train_one_run, main
calibrate.py      gather_logits + fit_temperature (LBFGS) +
                  expected_calibration_error + CLI (load checkpoint → fit T → save)
smoke_test.py     Wiring sanity test on examples/
utils/
  image_io.py     open_rgb(path) — single entry point for opening images;
                  registers the JXL handler and enables truncated-image tolerance
  index.py        index_files(name, roots) + PathList (sequence view over a
                  path trie) + clear_cache() — filesystem scan cached as
                  lzma-compressed trie JSON
  logsetup.py     setup_logging() — shared logger config
```

**Dependency direction**: `utils.* + constants → builders →
{stats, datasets} → config → {train, calibrate, infer, smoke_test}`.
builders has no project-local deps beyond `config`, which lets stats
import `build_eval_preprocess` and datasets import the rest of the
transforms without recreating the old builders↔stats cycle. No cycles.

**Import style**: `src/` is on `sys.path` when a CLI is run from project
root (`venv/bin/python src/<file>.py`), so absolute imports work — e.g.
`from utils.image_io import open_rgb`, `import config`, `from datasets
import DatasetSpec`. No leading-dot relative imports.

## Entry points

```
venv/bin/python src/smoke_test.py
venv/bin/python src/train.py     [--epochs N] [--batch-size N] [--num-workers N] [--invalidate-cache]
venv/bin/python src/calibrate.py [--checkpoint PATH] [--batch-size N] [--num-workers N] [--invalidate-cache]
venv/bin/python src/stats.py     [--num-workers N] [--invalidate-cache]
```

`stats.py` runs `get_stats` over every entry in `config.DATASETS` and
prints/cached results. Useful for pre-computing stats before a training
run so the first epoch isn't waiting on the scan.

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
   (not `log`). Setup goes through `utils.logsetup.setup_logging()`.
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
14. **Type-hint-only imports go under `if TYPE_CHECKING:`** with
    `from __future__ import annotations` at the top. Anything used only in
    annotations (e.g. `Path`, `Sequence`, `DatasetSpec`) belongs there so
    the runtime import graph stays minimal.
15. **Prefer `import module` over `from module import name` when there's a
    cycle risk.** `import module` only needs `module` to exist in
    `sys.modules` (even partially loaded), so it survives cycles that
    `from`-imports break.

### Splitting rules (learned the hard way)

- **Split files by responsibility / by the work they do**, not by line count.
- **Don't extract single-use helpers into their own files.** If only
  `train.py` calls `save_checkpoint`, it lives in `train.py`. Earlier I
  extracted `checkpoint.py`, `evaluate.py`, `schedule.py` and was told to
  put them back. They're now inlined.
- `DatasetSpec` lives in `datasets.py` alongside the dataset classes that
  consume it — it used to be in `spec.py` but the standalone file was
  one line of dataclass with no other purpose.
- `ROTATIONS` / `NUM_CLASSES` / `PROJECT_ROOT` / `CHECKPOINT_DIR` /
  `INDEX_CACHE_DIR` live in `constants.py`, not `config.py`, so data and
  model code can import them without dragging in the rest of config (and
  so config can import data-side types without creating a cycle).
- `build_train_dataset` / `build_val_dataset` live in `datasets.py`, not
  `builders.py`. They depend on `stats.get_stats`, and putting them in
  builders re-created the old builders↔stats cycle. Keeping `builders.py`
  as transforms-only and letting `datasets.py` own the wiring breaks the
  cycle while keeping each module honest.
- Transforms (`build_train_transform`, `build_eval_preprocess`,
  `build_eval_transform`) live in `builders.py`. Earlier they were in
  `config.py`; that was wrong — they're functions, not configuration.

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

- The user has **two GPUs** but only the 5060 Ti is in use; it drives both
  the desktop and training. The 1660 SUPER sits idle in the box.
- **`ImageList` was renamed to `ImageLoader`** — don't go back.
- The cache directory is `cache/` (no leading dot). Earlier it was `.cache/`
  which collided with the gitignore Python template.
- `pillow-jxl-plugin` registers on import as a side effect. That import now
  lives **only** in `utils/image_io.py` (with a `# noqa: F401`). Don't add
  it back to other modules — use `utils.image_io.open_rgb` instead, which
  is the whole reason that file exists.
- `RandomResizedCrop` with low scale (< ~0.7) can crop out all orientation
  cues. Keep crops mild.
- `index_files` returns a `PathList`, not a real list. It supports `len()`
  and `[idx]` and is iterable, but it is *not* a `list` — don't call
  list-only methods on it. `random.sample` and torch's `Dataset` both
  work because they only need the sequence protocol.
- Our computed dataset stats have higher std than the canonical ImageNet
  constants (~0.27 vs 0.229). Reason: we compute after the resize used
  at train time; canonical values are computed on raw full-res pixels.
  Don't "fix" this by hardcoding the canonical numbers — the computed
  ones match the actual input distribution.
- The transform ordering (PIL augmentations → ToTensor → Normalize →
  RandomErasing) is intentional. PIL ops can't take tensors, ColorJitter
  needs values in [0,1] not zero-centred, and RandomErasing's default
  fill of 0 means "dataset mean" *after* Normalize (a grey patch instead
  of pure black). Don't reorder.
- ConvNeXt-V2-Tiny ends in `AdaptiveAvgPool`, so `INPUT_SIZE` is a free
  knob, not an architectural requirement. 224 is the convention from the
  pretrained checkpoint we're *not* using; from-scratch training can
  pick 192/224/256/288 — bigger trades compute for more orientation
  cues. `RESIZE_SIZE` is just slightly larger than `INPUT_SIZE` so
  `RandomCrop` has wiggle room (~1.03–1.14× ratio).
- `stats.py` needs `tqdm`. It's not in the project's torch family deps so
  add it explicitly to requirements if a fresh venv is being set up.
- `infer.py` does NOT need to import `image_io` — `predict_image`
  receives an already-opened `PIL.Image` from the caller, and the caller
  is responsible for going through `open_rgb` themselves (which registers
  the JXL handler). Earlier we kept a defensive `import image_io` here
  for "safety"; it was paranoia and got removed.

## Outstanding work (as of last session)

- PyTorch CUDA 12.8 wheels for Blackwell installed by the user manually
  (not via `pip install torch`). Verify with `smoke_test.py` before
  kicking off a real training run.
- `timm` likewise installed manually after torch.
- `tqdm` is a new dep introduced by `stats.py`. Add to requirements if
  setting up a fresh venv.
- No checkpoint exists yet. First real run goes through `train.py`
  end-to-end.
- ImageNet stats have been computed once on this machine — wall-clock is
  ~5 minutes for 5% of the train split at ~215 img/s (HDD-bound after
  the page cache is exhausted; the path-sort + prefetch tweaks help but
  there's a hard disk-bandwidth ceiling).
- ONNX export is planned but deliberately deferred.
- Ambiguity-validation set (DTD textures, COCO circular subjects, Perlin
  noise) is not yet built. Needed for tuning the 4-rotation consistency
  threshold + temperature scaling.
