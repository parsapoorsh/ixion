"""Domain constants for the rotation detector.

These live in their own module (rather than in config.py) so dataset
shapes and model code can depend on them without depending on the rest
of config — keeping the import graph free of cycles when config grows
to import data-side types.
"""

from pathlib import Path

ROTATIONS = (0, 90, 180, 270)
NUM_CLASSES = len(ROTATIONS)

# Path layout. PROJECT_ROOT resolves via this file's location, so it
# stays correct as long as constants.py lives directly under src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
INDEX_CACHE_DIR = PROJECT_ROOT / "cache"

# DataLoader worker processes for image decoding. Tune to the machine —
# more workers help when decode is the bottleneck, but too many concurrent
# random seeks make HDD-backed datasets slower, not faster. Each CLI also
# exposes --num-workers to override per-run.
NUM_WORKERS = 8

# Batches each DataLoader worker keeps queued ahead of the consumer. Deeper
# queues mask HDD decode latency at the cost of more RAM held by workers.
PREFETCH_FACTOR = 4
