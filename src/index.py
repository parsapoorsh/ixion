"""File-system indexing: scan dataset directories once, cache the result.

Scanning 1.2M files takes tens of seconds; once cached, subsequent runs read
the list straight back. To keep the cache file small we serialise the path
list as a nested trie (each directory becomes a dict, files become leaves
with a null value) and write it through lzma. Shared path prefixes collapse,
the JSON is then compressed, and on-disk size drops by an order of magnitude
vs. the flat list of full strings.
"""

import json
import logging
import lzma
import time
from pathlib import Path
from typing import Sequence

import config

logger = logging.getLogger(__name__)


def clear_cache() -> None:
    """Delete every cached index file under the index cache directory."""
    if not config.INDEX_CACHE_DIR.exists():
        return
    for f in config.INDEX_CACHE_DIR.glob("index_*.json*"):
        f.unlink()
        logger.info("removed cached index %s", f.name)


def index_files(name: str, roots: Sequence[Path]) -> list[Path]:
    """Recursively list every regular file under the given roots, cached on disk.

    No extension filter — the dataset directories are expected to contain only
    images, and PIL will raise at load time on anything it can't decode.
    """
    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.INDEX_CACHE_DIR / f"index_{name}.json.xz"
    if cache.exists():
        with lzma.open(cache, "rt") as f:
            trie = json.load(f)
        return trie_to_paths(trie)

    t0 = time.monotonic()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for entry in root.rglob("*"):
            if entry.is_file():
                files.append(entry)
    dt = time.monotonic() - t0
    logger.info("scanned %s: %d files in %.1fs", name, len(files), dt)

    trie = paths_to_trie(files)
    with lzma.open(cache, "wt") as f:
        json.dump(trie, f)
    return files


def paths_to_trie(paths: Sequence[Path]) -> dict:
    """Collapse a list of paths into a nested dict keyed by path component.

    Files become leaves with the value ``None``; directories become subdicts.
    """
    root: dict = {}
    for p in paths:
        parts = p.parts
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None
    return root


def trie_to_paths(node: dict) -> list[Path]:
    """Inverse of paths_to_trie: walk the trie and materialise the path list."""
    out: list[Path] = []
    walk_trie(node, (), out)
    return out


def walk_trie(node: dict, prefix: tuple, out: list[Path]) -> None:
    for name, child in node.items():
        new_prefix = prefix + (name,)
        if child is None:
            out.append(Path(*new_prefix))
        else:
            walk_trie(child, new_prefix, out)
