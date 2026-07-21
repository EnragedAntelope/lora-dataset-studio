"""Tests for advisory helpers: near-duplicate detection + composition flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from studio.dedupe import dhash, find_near_duplicate_groups, hamming
from studio.quality import composition_flags


def _noise(path: Path, seed: int) -> Path:
    rng = np.random.RandomState(seed)
    Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    return path


# ---------- dedupe ----------

def test_dhash_identical_images_match(tmp_path: Path) -> None:
    a = _noise(tmp_path / "a.png", 1)
    b = _noise(tmp_path / "b.png", 1)  # same seed -> identical pixels
    assert dhash(a) == dhash(b)
    assert hamming(dhash(a), dhash(b)) == 0


def test_dhash_different_images_differ(tmp_path: Path) -> None:
    a = _noise(tmp_path / "a.png", 1)
    b = _noise(tmp_path / "b.png", 999)
    assert hamming(dhash(a), dhash(b)) > 5


def test_find_near_duplicate_groups(tmp_path: Path) -> None:
    a1 = _noise(tmp_path / "a1.png", 1)
    a2 = _noise(tmp_path / "a2.png", 1)   # dup of a1
    b = _noise(tmp_path / "b.png", 42)    # unique
    groups = find_near_duplicate_groups([a1, a2, b])
    assert len(groups) == 1
    assert set(groups[0]) == {a1, a2}


def test_find_near_duplicate_groups_distance_is_configurable(tmp_path: Path) -> None:
    a = _noise(tmp_path / "a.png", 1)
    b = _noise(tmp_path / "b.png", 999)   # differs by > 5 bits (see test above)
    # Strict distance keeps them apart; a very loose distance groups everything.
    assert find_near_duplicate_groups([a, b], max_distance=0) == []
    assert len(find_near_duplicate_groups([a, b], max_distance=64)) == 1


# ---------- composition ----------

def test_composition_flags_dark(tmp_path: Path) -> None:
    p = tmp_path / "dark.png"
    Image.new("RGB", (32, 32), (5, 5, 5)).save(p)
    flags = composition_flags(p)
    assert "dark" in flags and "low contrast" in flags


def test_composition_flags_bright(tmp_path: Path) -> None:
    p = tmp_path / "bright.png"
    Image.new("RGB", (32, 32), (250, 250, 250)).save(p)
    assert "bright" in composition_flags(p)


def test_composition_flags_normal_image_is_clean(tmp_path: Path) -> None:
    # Checkerboard: mid mean, high contrast -> nothing flagged.
    board = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    Image.fromarray(np.stack([board] * 3, axis=-1)).save(tmp_path / "n.png")
    assert composition_flags(tmp_path / "n.png") == []
