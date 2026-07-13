"""Advisory image-quality checks.

Currently a single, cheap sharpness estimate (variance of the Laplacian) used to
flag blurry shots in the curate/export views. It is deliberately advisory: it
labels images, it never deletes or blocks them. numpy-only — no cv2/scipy — so
it adds no heavy dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from studio.config import settings

# 3x3 discrete Laplacian; high response on edges, so a focused image has a much
# higher variance of the filtered signal than a blurred one.
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _convolve3x3(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid-region 3x3 convolution using shifted slices (no scipy)."""
    out = np.zeros(
        (gray.shape[0] - 2, gray.shape[1] - 2), dtype=np.float64
    )
    for dy in range(3):
        for dx in range(3):
            k = kernel[dy, dx]
            if k:
                out += k * gray[dy : dy + out.shape[0], dx : dx + out.shape[1]]
    return out


def sharpness(path: Path) -> float:
    """Variance of the Laplacian of the grayscale image. Higher = sharper."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    return float(_convolve3x3(gray, _LAPLACIAN).var())


def is_blurry(path: Path, threshold: float | None = None) -> tuple[bool, float]:
    """Return (blurry?, score). Uses the configured threshold when none given."""
    thr = settings.sharpness_blur_threshold if threshold is None else threshold
    score = sharpness(path)
    return score < thr, score
