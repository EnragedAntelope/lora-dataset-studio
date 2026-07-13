"""Tests for the advisory sharpness check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from studio.quality import is_blurry, sharpness


def _save(arr: np.ndarray, path: Path) -> Path:
    Image.fromarray(arr.astype("uint8")).save(path)
    return path


def _checkerboard(size: int = 64, cell: int = 4) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    return np.where(((xs // cell) + (ys // cell)) % 2 == 0, 255, 0)


def _blur(arr: np.ndarray, passes: int = 6) -> np.ndarray:
    out = arr.astype("float64")
    for _ in range(passes):  # cheap box blur, no scipy
        out = (out
               + np.roll(out, 1, 0) + np.roll(out, -1, 0)
               + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5.0
    return out


def test_sharp_scores_higher_than_blurred(tmp_path: Path) -> None:
    sharp = _save(_checkerboard(), tmp_path / "sharp.png")
    blurred = _save(_blur(_checkerboard()), tmp_path / "blur.png")
    assert sharpness(sharp) > sharpness(blurred)


def test_is_blurry_respects_threshold(tmp_path: Path) -> None:
    img = _save(_checkerboard(), tmp_path / "img.png")
    score = sharpness(img)
    high_blurry, _ = is_blurry(img, threshold=score + 1)
    low_blurry, _ = is_blurry(img, threshold=score - 1)
    assert high_blurry is True
    assert low_blurry is False


def test_tiny_image_is_safe(tmp_path: Path) -> None:
    tiny = _save(np.zeros((2, 2)), tmp_path / "tiny.png")
    assert sharpness(tiny) == 0.0
