"""Domain constants for the rotation detector.

These live in their own module (rather than in config.py) so dataset
shapes and model code can depend on them without depending on the rest
of config - keeping the import graph free of cycles when config grows
to import data-side types.
"""

from pathlib import Path

ROTATIONS = (0, 90, 180, 270)
# Rotated views generated per base image. The model itself has a single
# output neuron (P(upright)); this is purely a data-side count.

# Path layout. PROJECT_ROOT resolves via this file's location, so it
# stays correct as long as constants.py lives directly under src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INDEX_CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

LOG_STEP = 50

# Preprocess sizes for the EfficientNet-B0 rotation model.
# resize_size is ~1.03x input_size so RandomCrop has wiggle room.
INPUT_SIZE = 224
RESIZE_SIZE = 232

# The stats pass decodes a seeded fraction of the train split rather than
# every file, which keeps mean/std cheap and stable. These are part of the
# stats cache fingerprint, so changing them needs --invalidate-cache.
STATS_SAMPLE_FRACTION = 0.20
STATS_SAMPLE_SEED = 0

