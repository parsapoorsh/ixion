# CLAUDE.md

Project context, design decisions, and the user's coding preferences. Read
this first.

## What this project is

An image rotation detector that classifies each input as one of 4 rotations
(0°/90°/180°/270°) and abstains on genuinely ambiguous images (uniform
textures, circles, symmetric content). Built to pre-sort ~500k personal
photos for a self-hosted Immich instance - so it's batch-processing, not
real-time.

## Hardware target

The codebase itself is hardware-agnostic — every CLI takes `--device`,
`--batch-size`, `--num-workers`, and `--prefetch-factor` as required
flags. The user's dev machine is an RTX 5060 Ti 16GB (Blackwell,
sm_120) that also drives the desktop, so a fresh run there leaves
~0.5–1 GB of VRAM permanently reserved by the compositor — size
batches with that in mind. Requires PyTorch with CUDA 12.8+ wheels for
this card. BF16 autocast is the only mixed-precision path in
`train.py`; the validator runs fp32.

**Inference VRAM budget**: ≤ 250 MB. Comfortable via ONNX Runtime at
deployment time; PyTorch's CUDA context alone eats ~180 MB so dev
measurements look higher than the eventual deployment.

## Model architecture

- **Backbone**: EfficientNet-B0 via timm (~5.3M params, ~0.4 GFLOPs
  at 224²), built directly inside `RotationModel.__init__` with
  `global_pool="avg"` and `num_classes=0`. Trained from scratch
  (`pretrained=False`). There used to be a `Backbone` ABC under
  `models/backbones/` so the arch was swappable with
  ConvNeXt-V2-Tiny; both the abstraction and the second arch were
  ripped out once the project settled on B0.
- **Head**: `LayerNorm(1280) → Dropout(0.2) → Linear(1280, 4)`. The
  global average pool is provided by timm; the head consumes the
  pooled features directly.
- **No temperature buffer**: temperature scaling was wired up but never
  fit (no checkpoints exist), so it was pulled. Reintroduce when a
  calibration step actually matters.
- **Single LR**: backbone and head learn at the same rate (random init,
  no reason for a layer-wise split).

## Training recipe

Single phase, from random init. No frozen warm-up phase, no
noise-filtering pass - both were over-engineered.

Recipe knobs (`epochs`, `lr`, `weight_decay`, `label_smoothing`,
`warmup_epochs`, `grad_clip`, `ema_decay`) are plain kwargs on
`train_one_run` with the project defaults baked in as parameter
defaults — no `TrainRecipe`/`TrainDefaults` dataclass and no
`dataclasses.replace` plumbing. `train.main()` parses `--epochs` (its
default pulled from the function signature via `inspect` so there's
a single source of truth) and passes overrides through as kwargs.
Preprocess sizes (`input_size`, `resize_size`) are constants —
`constants.INPUT_SIZE` / `constants.RESIZE_SIZE`. `batch_size` is
**not** a recipe knob — it's a hardware decision and lives only on
the CLI. `RunConfig` carries the process/IO knobs (`device`,
`num_workers`, `prefetch_factor`, `seed`, `log_every`) — independent
of recipe.

Project defaults:
- BF16 autocast, AdamW (β=0.9/0.999), weight_decay 0.05
- LR 1e-3, cosine schedule with 5-epoch linear warmup
- Label smoothing 0.1 - handles the ~1% wrong-orientation images in ImageNet
- EMA decay 0.9999, evaluated and checkpointed off the EMA weights
- Gradient clip at norm=1.0
- 100 epochs
- input_size=224, resize_size=232

### Augmentations - be careful here

**Safe** (preserve rotation label):
- ColorJitter, RandomGrayscale, RandomErasing, Normalize, mild RandomCrop

**Forbidden** (corrupt the label):
- RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
  RandomPerspective, RandomAffine - any spatial op that re-orients pixels.
  Pay attention if you add new augmentations.

## Data pipeline

- **Sources**: ImageNet ILSVRC2017 (`.../CLS-LOC/{train,val}`) + COCO 2017
  (`.../coco/{train2017,val2017}`).
- **Format**: JPEG-XL (`.jxl`). All image opening goes through
  `utils.image_io.open_rgb`, which registers the `pillow_jxl` handler and
  enables truncated-image tolerance in one place. Don't call
  `PIL.Image.open` directly - use `open_rgb` so new modules can't silently
  miss either side effect. Directories may also contain other formats; do
  **not** hard-code `.jxl` anywhere, let PIL fail at decode if anything's
  wrong.
- **File index**: scanned once per (split, dataset). The result is collapsed
  into a path *trie* (nested dict, directories as subdicts, files as null
  leaves) and written through `lzma` to `cache/index_<name>.json.xz` - shared
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
  `stats.get_stats(spec, input_size, resize_size, num_workers, device)`
  samples `config.STATS_SAMPLE_FRACTION` (5% default) of the train split
  with seed `config.STATS_SAMPLE_SEED`, computes per-channel mean/std,
  and caches the result as JSON under
  `cache/stats_<name>_<resize>x<input>.json`. The cache filename includes
  both sizes because stats depend on the actual pixels after
  Resize→CenterCrop - a different preprocess silently reusing stale
  numbers would be a bug. Default `DATASETS` in `config.py` pass `None`
  for both shipped datasets; `IMAGENET_MEAN`/`IMAGENET_STD` remain in
  `config.py` so a spec can be pinned to fixed constants when wanted.
  `build_train_dataset` / `build_val_dataset` (in `datasets.py`) take
  `input_size`/`resize_size`/`num_workers`/`device` and thread them
  into `get_stats`, then `ConcatDataset` the per-spec parts. Callers
  pass `constants.INPUT_SIZE` / `constants.RESIZE_SIZE` — they used
  to live in `config.py`, then briefly on the Backbone class while
  the project had multiple archs, and landed in `constants.py` when
  the Backbone abstraction was removed.
- **Stats computation** is device-accelerated: images are decoded via
  the shared `build_eval_preprocess()` pipeline (Resize → CenterCrop →
  ToTensor, no Normalize — we're computing the normaliser), workers
  parallelise decode, the per-batch sum/sum-of-squares reduces on the
  caller-supplied device in float32, and accumulation happens on CPU in
  float64 (consumer GPUs are ~1/64 speed for float64). The sample is
  sorted by path *after* sampling so reads happen in on-disk locality
  order — a cheap win regardless of storage medium. The stats CLI takes
  `--device`, `--num-workers` (both required) so the loader matches the
  machine it runs on. tqdm shows progress in images/s with a per-image
  `pbar.update(b)`. Changing `STATS_SAMPLE_FRACTION` needs
  `--invalidate-cache`; the cache file doesn't track which fraction
  produced it. Stats are computed *after* the resize, so the std comes
  out higher than canonical ImageNet (~0.27 vs 0.229) — that's expected
  and consistent with the input distribution the model actually sees.
  Don't "fix" by hardcoding canonical numbers.
- **DataLoader knobs are caller-supplied**: there's no
  `constants.NUM_WORKERS` / `constants.PREFETCH_FACTOR` default. Every
  CLI requires `--num-workers` and `--prefetch-factor` and passes them
  down through `RunConfig`/builders so the library never silently picks
  a value sized for one machine.
- **Checkpointing**: `train.py` dumps `checkpoints/epoch_NNN.pt` after every
  epoch (both raw and EMA state dicts, plus `epoch` and the latest val
  `acc`). Earlier we kept only `best.pt` and `last.pt`; switched to
  per-epoch because long runs benefit from being able to resume or bisect
  from any point, and a "best" by val acc is too easy to spike on noise.
- **Logging**: `utils.logsetup.setup_logging()` attaches *two* handlers to
  the root logger - the existing tqdm-aware console handler and a
  `FileHandler` writing to `logs/YYYY-MM-DD_HH-MM-SS.txt`
  (`constants.LOGS_DIR`). Both share one `Formatter` with full date+time
  so a multi-day run's lines stay self-dating. `logs/` is gitignored.

## Ambiguity detection - the key insight

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
4-bucket vote share - use it for ranking or thresholding when the
categorical confident/weak/ambiguous label is too coarse. The categorical
label still drives the abstention decision (whether `rotation` is None).

Entropy is still reported per-prediction. Temperature scaling existed
as a calibrated tiebreaker but is gone for now - bring it back when an
actual checkpoint exists to fit T against.

## File layout (src/)

ML-domain code lives at `src/` root; general-purpose helpers live under
`src/utils/` (a real package, with `__init__.py`).

```
config.py         IMAGENET_MEAN/STD, DATASETS (instances),
                  STATS_SAMPLE_FRACTION/SEED. No DEVICE — every CLI
                  takes --device explicitly. No DEFAULT_BACKBONE — the
                  arch is fixed and lives inside RotationModel.
constants.py      ROTATIONS, NUM_CLASSES, PROJECT_ROOT, CHECKPOINT_DIR,
                  INDEX_CACHE_DIR, LOGS_DIR, LOG_STEP, INPUT_SIZE,
                  RESIZE_SIZE — domain constants and static paths
                  depended on by datasets/model/train/infer/utils
                  without going through config. No hardware knobs
                  (NUM_WORKERS/PREFETCH_FACTOR) live here.
datasets.py       DatasetSpec dataclass (name, train, val, mean=None, std=None),
                  ImageLoader (PIL by index), RotationPairDataset (4× wrapper),
                  build_train_dataset / build_val_dataset (take input_size,
                  resize_size, num_workers, device; call get_stats + the
                  transforms from builders, then ConcatDataset the per-spec
                  parts)
stats.py          get_stats(spec, input_size, resize_size, num_workers,
                  device) + compute_stats (device-accelerated via a
                  DataLoader + ImageLoader) + sample_files (sorted by
                  path) + CLI for ad-hoc stats runs (--device, --num-workers
                  required). Sampled mean/std cached as
                  cache/stats_<name>_<resize>x<input>.json - sizes in
                  the filename so a different preprocess can't reuse
                  stale numbers. Explicit spec.mean/spec.std bypasses
                  the scan entirely.
builders.py       build_train_transform, build_eval_preprocess (Resize →
                  CenterCrop → ToTensor; shared with StatsDataset),
                  build_eval_transform (preprocess + Normalize). Each
                  takes input_size + resize_size explicitly - transforms
                  no longer read globals from config. No project-local
                  deps; transforms only.
models/
  __init__.py     re-exports RotationModel, ModelEMA
  rotation.py     RotationModel - builds efficientnet_b0 via timm
                  (global_pool="avg", num_classes=0) and stacks the
                  4-class head on top. No Backbone indirection.
  ema.py          ModelEMA - shadow-weight wrapper updated via
                  bias-corrected ramp.
infer.py          Prediction dataclass (rotation, confidence categorical,
                  confidence_pct continuous, votes, entropy) +
                  predict_image (takes input_size/resize_size from the
                  caller, runs 4-rotation consistency + posterior averaging)
train.py          RunConfig (device, num_workers, prefetch_factor,
                  seed, log_every), save_checkpoint,
                  cosine_with_warmup, set_lr, train_one_run (run,
                  batch_size, + optimisation kwargs with defaults
                  baked in), main. main pulls the --epochs argparse
                  default from train_one_run's signature via inspect
                  so there's a single source of truth. No validation
                  here; see validator.py.
validator.py      Standalone checkpoint validator CLI. run_validation
                  (fp32, no autocast) + validate_checkpoint loader.
                  Required flags: --checkpoint, --device, --batch-size,
                  --num-workers, --prefetch-factor. No async machinery,
                  no ambiguity scan, no dtype menu.
smoke_test.py     Wiring sanity test on examples/ (device auto-detected;
                  exercises run_validation but no scan)
utils/
  image_io.py     open_rgb(path) - single entry point for opening images;
                  registers the JXL handler and enables truncated-image tolerance
  index.py        index_files(name, roots) + PathList (sequence view over a
                  path trie) + clear_cache() - filesystem scan cached as
                  lzma-compressed trie JSON
  logsetup.py     setup_logging() - shared logger config; installs a
                  TqdmLoggingHandler so log records route through
                  tqdm.write and don't clobber an active progress bar,
                  and a FileHandler that mirrors everything to
                  logs/YYYY-MM-DD_HH-MM-SS.txt
```

**Dependency direction**: `utils.* + constants → builders →
{models.*, datasets, stats} → config → {train, infer, smoke_test}`.
config no longer imports anything from `models` (the Backbone
instance is gone). builders has no project-local deps beyond
torchvision, which lets stats import `build_eval_preprocess` and
datasets import the rest of the transforms without recreating the
old builders↔stats cycle. No cycles.

**Import style**: `src/` is on `sys.path` when a CLI is run from project
root (`venv/bin/python src/<file>.py`), so absolute imports work - e.g.
`from utils.image_io import open_rgb`, `import config`, `from datasets
import DatasetSpec`. No leading-dot relative imports.

## Entry points

```
venv/bin/python src/smoke_test.py

venv/bin/python src/train.py \
    --device cuda --batch-size N --num-workers N --prefetch-factor N \
    [--epochs N] [--seed N] [--resume PATH] [--invalidate-cache]

venv/bin/python src/validator.py \
    --checkpoint checkpoints/epoch_NNN.pt \
    --device cuda --batch-size N --num-workers N --prefetch-factor N \
    [--weights ema|raw] [--max-batches N]

venv/bin/python src/stats.py \
    --device cuda --num-workers N \
    [--input-size N] [--resize-size N] [--invalidate-cache]
```

`--device`, `--batch-size`, `--num-workers`, and `--prefetch-factor`
are **required** on every CLI that builds a DataLoader. There are no
hardware defaults baked into the library — pick values for the
machine the command is running on.

`train.py`'s recipe defaults are kwargs on `train_one_run`; the
`--epochs` argparse default is read from that signature via
`inspect`, so the value lives in exactly one place. There is no
`--backbone` flag — the arch is fixed.

`stats.py` runs `get_stats` over every entry in `config.DATASETS` at
`constants.INPUT_SIZE` / `constants.RESIZE_SIZE` and prints/caches
the result. Useful for pre-computing stats before a training run so
the first epoch isn't waiting on the scan. Pass `--input-size`/
`--resize-size` to compute stats at a different size - they'll land
in a separately keyed cache file.

`validator.py` is independent of `train.py` — point it at any
`epoch_NNN.pt` and it runs one fp32 validation pass on the device
you specify. No async, no IPC, no second-GPU assumption.

## User's coding preferences (firm)

Strict rules - apply without asking:

1. **No underscore-prefixed names** at module level. Reserved only for
   private class methods when the rest of the styling allows it.
2. **All imports at the top of the file.** Never `import x` inside a function.
3. **No `global x`**. If you need it, the code is wrong.
4. **Small, focused git commits.** Big commits get rejected. One change per
   commit. Always use the configured `git config user.*` - don't override.
5. **Multi-line function signatures** when there are many args/kwargs.
6. **Use `logging`, never `print`.** Module-level variable is named `logger`
   (not `log`). Setup goes through `utils.logsetup.setup_logging()`.
7. **Use `time.monotonic()`** for elapsed-time measurements, not `time.time()`.
8. **Dataclasses, not dicts**, for structured returns.
9. **No defaults for things the caller should choose.** e.g. `mean`/`std` are
   required kwargs, not defaulted to ImageNet constants.
10. **No extension whitelists** when scanning files unless we have a concrete
    reason - let PIL fail at decode time.
11. **Cross-cutting static state lives in `config.py`** (DATASETS,
    stats sampling fraction/seed) or `constants.py` (paths,
    ROTATIONS, LOG_STEP, INPUT_SIZE, RESIZE_SIZE). **Hardware/runtime
    knobs (device, num_workers, prefetch_factor, batch_size) do NOT
    live in either — they come from the CLI on every entry point** so
    the library stays machine-agnostic.
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
- `ModelEMA` lives in `models/ema.py`. It's used by `train.py` only,
  but the module is non-trivial (deepcopy plumbing + bias-corrected
  update + load/save). Inlining it once made `train.py` noisier than
  it gained.
- `DatasetSpec` lives in `datasets.py` alongside the dataset classes that
  consume it - it used to be in `spec.py` but the standalone file was
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
  `config.py`; that was wrong - they're functions, not configuration.
- The arch lives inline in `RotationModel`. Earlier there was a
  `Backbone` ABC under `models/backbones/` with `EfficientNetB0` and
  `ConvNeXtV2Tiny` subclasses, a `TrainDefaults` ClassVar carrying
  recipe knobs, and `config.DEFAULT_BACKBONE` pointing at the chosen
  instance. The whole tower came down when ConvNeXt was dropped:
  with one arch, the indirection was just noise. If a second arch
  ever lands, prefer adding it inline (e.g. an `arch` arg on
  `RotationModel`) before reintroducing an ABC.
- Recipe knobs are kwargs on `train_one_run`, not a dataclass.
  Earlier shapes that got walked back: `TrainConfig(TrainDefaults)`
  bundling backbone+recipe, then a standalone `TrainRecipe` frozen
  dataclass + `dataclasses.replace` for CLI overrides. The current
  shape is plain kwargs with defaults baked in; the CLI reads its
  `--epochs` default from the signature via `inspect`.
- `RunConfig` (process/IO/seed/logging knobs) is separate from
  recipe kwargs by intent - same recipe, different machine.

### Anti-patterns the user called out

- Calling `git add -A` and accidentally committing fixtures/examples.
- Re-authoring history with overrides when `git config user.*` is already
  correct - just commit.
- Factory functions like `build_model()` when the class constructor
  is already fine: `RotationModel(pretrained=False)`.
- Keeping unused dead helpers around "just in case" (`predict_path`,
  `load_model_for_inference`, `build_model`, the temperature
  buffer/calibrated_logits/calibrate.py - all removed).
- ABCs / registries / wrapper dataclasses introduced "for
  swappability" before the second instance actually exists. The
  `Backbone` ABC, the `BACKBONES = {name: ...}` registry attempts,
  the `TrainConfig(TrainDefaults)` wrapper, and the `TrainRecipe`
  dataclass were all paid in indirection without a second concrete
  use case. Wait for the second case to land before abstracting.
- Duplicating defaults between an argparse `default=` and the
  function it calls. If `train_one_run` has `epochs: int = 100`, the
  CLI default reads from `inspect.signature(...)` rather than
  repeating the number.

## Communication style

- The user expects **honest pushback**, not agreement. If a suggestion is
  wrong, say so with reasoning. They asked "is deepcopy normal or should you
  use pytorch itself" - the right answer was "yes deepcopy is normal, but
  pytorch has AveragedModel which also uses deepcopy" with the trade-off.
- The user **does not want clarifying questions** - make the reasonable call
  and continue, they'll redirect.
- Keep updates short. End-of-turn summary: 1–2 sentences max.

## Gotchas

- The user has **two GPUs** physically (5060 Ti + 1660 SUPER) but the
  codebase no longer assumes anything about that — there's no
  `cuda:1` defaulting anywhere. Pass `--device` explicitly. The 1660
  SUPER sits idle.
- **`ImageList` was renamed to `ImageLoader`** - don't go back.
- **`async_validator.py` was renamed to `validator.py`** and the
  AsyncValidator/scan_dataset/AmbiguityScan* classes are gone. If
  something asks for them, it's stale.
- The cache directory is `cache/` (no leading dot). Earlier it was `.cache/`
  which collided with the gitignore Python template.
- `pillow-jxl-plugin` registers on import as a side effect. That import now
  lives **only** in `utils/image_io.py` (with a `# noqa: F401`). Don't add
  it back to other modules - use `utils.image_io.open_rgb` instead, which
  is the whole reason that file exists.
- `RandomResizedCrop` with low scale (< ~0.7) can crop out all orientation
  cues. Keep crops mild.
- `index_files` returns a `PathList`, not a real list. It supports `len()`
  and `[idx]` and is iterable, but it is *not* a `list` - don't call
  list-only methods on it. `random.sample` and torch's `Dataset` both
  work because they only need the sequence protocol.
- Our computed dataset stats have higher std than the canonical ImageNet
  constants (~0.27 vs 0.229). Reason: we compute after the resize used
  at train time; canonical values are computed on raw full-res pixels.
  Don't "fix" this by hardcoding the canonical numbers - the computed
  ones match the actual input distribution.
- The transform ordering (PIL augmentations → ToTensor → Normalize →
  RandomErasing) is intentional. PIL ops can't take tensors, ColorJitter
  needs values in [0,1] not zero-centred, and RandomErasing's default
  fill of 0 means "dataset mean" *after* Normalize (a grey patch instead
  of pure black). Don't reorder.
- EfficientNet-B0 ends in `AdaptiveAvgPool`, so `INPUT_SIZE` is a
  convention (224), not an architectural requirement. From-scratch
  training can pick 192/256/288 — bigger trades compute for more
  orientation cues. `RESIZE_SIZE` is just slightly larger so
  `RandomCrop` has wiggle room (~1.03× ratio at 232/224). Both live
  in `constants.py` now; they used to be on the Backbone class while
  the project had multiple archs (different timm archs want
  different native sizes — B1=240, B2=256).
- Stats are cached per `(dataset, resize_size, input_size)` triple in
  the filename. If you change `INPUT_SIZE`/`RESIZE_SIZE`, the
  existing cache silently won't match and stats recompute - that's
  the intended behavior, not a bug.
- `stats.py` needs `tqdm`. It's not in the project's torch family deps so
  add it explicitly to requirements if a fresh venv is being set up.
- `infer.py` does NOT need to import `image_io` - `predict_image`
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
  end-to-end. The arch is EfficientNet-B0 (the project briefly
  shipped a swappable Backbone ABC with ConvNeXt-V2-Tiny as a second
  option; both were removed once B0 was settled).
- Datasets live on SSD at `/media/poweruser/data/2-stuff/datasets/`.
  Earlier they were on an HDD at `/mnt/archive/...` and the loop was
  I/O-bound; the move was made to let compute be the actual ceiling.
- ImageNet stats have been computed once on this machine at 232×224.
  The earlier `cache/stats_*.json` (size-less filenames) won't match
  the new `stats_<name>_<resize>x<input>.json` and will be recomputed.
- ONNX export is planned but deliberately deferred.
- The ambiguity scan (4-rotation consistency walk over the val set,
  writing `cache/ambiguity.json`) was removed when the validator was
  decoupled from training. Reintroduce as its own file once there's
  a trained checkpoint to actually scan with.
- Temperature scaling / `calibrate.py` were removed because no
  checkpoint exists to fit T against. Reintroduce when there's a real
  trained model and ECE is worth measuring.
