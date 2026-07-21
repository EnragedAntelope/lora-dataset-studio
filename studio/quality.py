"""Advisory image-quality checks.

Cheap, numpy-only estimates used to *label* shots in the curate/export views —
never to delete or block them:

- **sharpness** (variance of the Laplacian) -> "blurry"
- **exposure/contrast** (mean and spread of luminance) -> "dark"/"bright"/"low contrast"

These are honest heuristics, not a semantic framing model: they catch the common
"something's off with this shot" cases without a heavy dependency. Semantic
framing (face/bust/body/back) would need a detector and is intentionally left out.
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


def exposure(path: Path) -> tuple[float, float]:
    """(mean luminance, luminance std) on 0-255. Low mean = dark, low std = flat."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float64)
    return float(gray.mean()), float(gray.std())


def composition_flags(path: Path) -> list[str]:
    """Advisory exposure/contrast labels for a shot (empty = nothing notable).

    Thresholds are tunable via `LDS_DARK_LUMA_THRESHOLD` / `LDS_BRIGHT_LUMA_
    THRESHOLD` / `LDS_LOW_CONTRAST_THRESHOLD`.
    """
    mean, std = exposure(path)
    flags: list[str] = []
    if mean <= settings.dark_luma_threshold:
        flags.append("dark")
    elif mean >= settings.bright_luma_threshold:
        flags.append("bright")
    if std <= settings.low_contrast_threshold:
        flags.append("low contrast")
    return flags
