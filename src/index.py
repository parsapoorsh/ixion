"""File-system indexing: scan dataset directories once, cache the result.

Scanning 1.2M files takes tens of seconds; once cached, subsequent runs read
the list straight back. The cache stores the path list as a nested trie
(each directory becomes a dict, files become leaves with a null value) and
the JSON is written through lzma. Shared path prefixes collapse and the
file shrinks by roughly an order of magnitude vs. the flat list.

At read time we don't materialise the full path list. ``index_files``
returns a ``PathList`` — a sequence-like view over the trie that
reconstructs paths on demand. The model only ever needs ``len`` and
``[idx]``, which is what torch's ``Dataset`` API requires, so a real list
is not needed and the trie can stay collapsed in memory.
"""

import bisect
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


def index_files(name: str, roots: Sequence[Path]) -> "PathList":
    """Recursively list every regular file under the given roots, cached on disk.

    No extension filter — the dataset directories are expected to contain only
    images, and PIL will raise at load time on anything it can't decode.
    """
    config.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.INDEX_CACHE_DIR / f"index_{name}.json.xz"
    if cache.exists():
        with lzma.open(cache, "rt") as f:
            trie = json.load(f)
        return PathList(trie)

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
    return PathList(trie)


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


class PathList:
    """Sequence view over a path trie. Random access in O(depth * log branching).

    Build once: walks the trie and stores, per internal node, the ordered
    list of child names and the cumulative leaf count after each child. At
    lookup time we descend the trie, binary-searching the cum array at each
    level to find the child whose subtree contains the requested index.

    Memory cost on top of the trie itself is one (keys-list, cums-list) pair
    per *internal* node — orders of magnitude fewer than file count.
    """

    def __init__(self, trie: dict) -> None:
        self.root = trie
        self.children: dict[int, list[str]] = {}
        self.cum_counts: dict[int, list[int]] = {}
        self.length = self.build_index(trie)

    def build_index(self, node: dict) -> int:
        keys: list[str] = []
        cum: list[int] = []
        total = 0
        for name, child in node.items():
            keys.append(name)
            if child is None:
                total += 1
            else:
                total += self.build_index(child)
            cum.append(total)
        self.children[id(node)] = keys
        self.cum_counts[id(node)] = cum
        return total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Path:
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        parts: list[str] = []
        node = self.root
        remaining = idx
        while True:
            keys = self.children[id(node)]
            cum = self.cum_counts[id(node)]
            i = bisect.bisect_right(cum, remaining)
            name = keys[i]
            parts.append(name)
            child = node[name]
            if child is None:
                return Path(*parts)
            prev = cum[i - 1] if i > 0 else 0
            remaining -= prev
            node = child

    def __iter__(self):
        return (self[i] for i in range(self.length))
