"""File-system indexing: scan dataset directories once, cache the result.

Scanning 1.2M files takes tens of seconds; once cached, subsequent runs read
the trie straight back. Paths are inserted into a nested trie as the scan
walks the filesystem (each directory becomes a dict, files become leaves
with a null value) so we never hold a flat ``list[Path]`` in memory. The
JSON is written through lzma; shared path prefixes collapse and the file
shrinks by roughly an order of magnitude vs. a flat list.

At read time we don't materialise the full path list. ``index_files``
returns a ``PathList`` - a sequence-like view over the trie that
reconstructs paths on demand. The model only ever needs ``len`` and
``[idx]``, which is what torch's ``Dataset`` API requires, so a real list
is not needed and the trie can stay collapsed in memory.
"""

from __future__ import annotations

import bisect
import json
import logging
import lzma
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def index_files(
    name: str,
    roots: Sequence[Path],
    cache_dir: Path,
    invalidate_cache: bool = False,
) -> "PathList":
    """Recursively list every regular file under the given roots, cached on disk.

    No extension filter - the dataset directories are expected to contain only
    images, and PIL will raise at load time on anything it can't decode.

    When ``invalidate_cache`` is true, every cached index under ``cache_dir``
    is deleted before scanning.
    """
    if invalidate_cache and cache_dir.exists():
        for f in cache_dir.glob("index_*.json*"):
            f.unlink()
            logger.info("removed cached index %s", f.name)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"index_{name}.json.xz"
    if cache.exists():
        with lzma.open(cache, "rt") as f:
            trie = json.load(f)
        return PathList(trie)

    t0 = time.monotonic()
    trie: dict = {}
    count = 0
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        for entry in root.rglob("*"):
            if entry.is_file():
                parts = entry.parts
                node = trie
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = None
                count += 1
    dt = time.monotonic() - t0
    logger.info("scanned %s: %d files in %.1fs", name, count, dt)

    with lzma.open(cache, "wt") as f:
        json.dump(trie, f)
    return PathList(trie)


class PathList(Sequence[Path]):
    """Sequence view over a path trie. Random access in O(depth * log branching).

    Build once: walks the trie and stores, per internal node, the ordered
    list of child names and the cumulative leaf count after each child. At
    lookup time we descend the trie, binary-searching the cum array at each
    level to find the child whose subtree contains the requested index.

    Memory cost on top of the trie itself is one (keys-list, cums-list) pair
    per *internal* node - orders of magnitude fewer than file count.
    """

    def __init__(self, trie: dict) -> None:
        self.root = trie
        self.children: dict[int, list[str]] = {}
        self.cum_counts: dict[int, list[int]] = {}
        self.length = self.build_index(trie)

    def __getstate__(self) -> dict:
        # children/cum_counts key on id(node), which is not stable across
        # serialization - when a DataLoader spawn-worker reconstructs this,
        # the trie dicts get fresh memory addresses and the cached keys no
        # longer match. Ship only the trie and rebuild the index in the
        # recipient process.
        return {"root": self.root}

    def __setstate__(self, state: dict) -> None:
        self.root = state["root"]
        self.children = {}
        self.cum_counts = {}
        self.length = self.build_index(self.root)

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

    def __iter__(self) -> Iterator[Path]:
        return (self[i] for i in range(self.length))
