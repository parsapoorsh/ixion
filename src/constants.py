"""Domain constants for the rotation detector.

These live in their own module (rather than in config.py) so dataset
shapes and model code can depend on them without depending on the rest
of config — keeping the import graph free of cycles when config grows
to import data-side types.
"""

ROTATIONS = (0, 90, 180, 270)
NUM_CLASSES = len(ROTATIONS)
