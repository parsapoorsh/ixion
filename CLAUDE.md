# CLAUDE.md

Project context, design decisions, and the user's coding preferences. Read
this first.

## What this project is

An image rotation detector that classifies each input as one of 4 rotations
(0°/90°/180°/270°) and abstains on genuinely ambiguous images (uniform
textures, circles, symmetric content). Built to pre-sort ~500k personal
photos for a self-hosted Immich instance - so it's batch-processing, not
real-time.
