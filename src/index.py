"""File-system indexing: scan dataset directories once, cache the result.

Scanning 1.2M files takes tens of seconds; once cached to JSON, subsequent
runs read the list straight back. The cache lives under cache/ at the project
root.
"""

import json
import logging
import time
from pathlib import Path
from typing import Sequence

import config

log = logging.getLogger(__name__)


def scan_images(roots: Sequence[Path]) -> list[Path]:
    """Recursively gather every regular file under the given roots.

    No extension filter — the dataset directories are expected to contain only
    images, and PIL will raise at load time on anything it can't decode.
    """
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for entry in root.rglob("*"):
            if entry.is_file():
                out.append(entry)
    return out


def load_or_build_index(
    name: str,
    roots: Sequence[Path],
) -> list[Path]:
    """Build (and cache to disk) the file list for a dataset split."""
    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.INDEX_CACHE_DIR / f"index_{name}.json"
    if cache.exists():
        with cache.open() as f:
            return [Path(p) for p in json.load(f)]

    t0 = time.time()
    files = scan_images(roots)
    dt = time.time() - t0
    log.info("scanned %s: %d files in %.1fs", name, len(files), dt)

    with cache.open("w") as f:
        json.dump([str(p) for p in files], f)
    return files
