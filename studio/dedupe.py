"""Advisory near-duplicate detection (perceptual dHash, numpy-only).

Like the sharpness check, this is **advisory**: it groups images that look
near-identical so you can spot an over-weighted duplicate before packaging — it
never deletes or blocks anything. A difference hash (dHash) is cheap, rotation-
/exact-crop-insensitive enough for "same shot twice", and needs no new dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path: Path, hash_size: int = 8) -> int:
    """Difference hash: compare each pixel to its right neighbour on a tiny grey
    thumbnail. Returns a `hash_size * hash_size`-bit integer."""
    with Image.open(path) as im:
        small = im.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    grey = np.asarray(small, dtype=np.int16)
    diff = grey[:, 1:] > grey[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def find_near_duplicate_groups(
    paths: list[Path], max_distance: int = 5
) -> list[list[Path]]:
    """Group images whose dHashes are within `max_distance` bits of each other.

    Order-preserving single-link grouping. Only groups of 2+ are returned; images
    that fail to open are skipped (advisory, never fatal).
    """
    hashed: list[tuple[Path, int]] = []
    for p in paths:
        try:
            hashed.append((p, dhash(p)))
        except Exception:
            continue  # advisory — an unreadable image must not break the scan
    groups: list[list[Path]] = []
    used: set[Path] = set()
    for i, (p, h) in enumerate(hashed):
        if p in used:
            continue
        group = [p]
        for q, hq in hashed[i + 1:]:
            if q not in used and hamming(h, hq) <= max_distance:
                group.append(q)
                used.add(q)
        if len(group) > 1:
            used.add(p)
            groups.append(group)
    return groups
