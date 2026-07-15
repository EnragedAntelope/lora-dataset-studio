"""Inspect a finished dataset folder so training configs can be derived from it.

A flat 2000 steps is wrong for both an 8-image and a 60-image dataset, and a
single 1024 bucket wastes any non-square images. Reading the folder costs one
Pillow header parse per file and makes both choices explainable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from studio.config import list_images

# Character LoRAs converge around ~75 steps per image on these trainers. Clamped
# because the ratio stops holding at the extremes: a 4-image set still needs a
# floor to learn anything, and a 100-image set does not need 7,500 steps.
STEPS_PER_IMAGE = 75
MIN_STEPS = 1000
MAX_STEPS = 4000

# ai-toolkit's established multi-resolution idiom: list the buckets and it sorts
# images into them. Only keys attested in ai-toolkit's own examples are emitted.
BUCKET_LADDER = [512, 768, 1024]


@dataclass
class DatasetStats:
    n_images: int
    n_captioned: int
    sizes: list[tuple[int, int]] = field(default_factory=list)
    aspect_counts: Counter = field(default_factory=Counter)

    @property
    def min_long_side(self) -> int:
        return min((max(s) for s in self.sizes), default=0)

    @property
    def max_long_side(self) -> int:
        return max((max(s) for s in self.sizes), default=0)

    @property
    def suggested_steps(self) -> int:
        if not self.n_images:
            return MIN_STEPS
        return max(MIN_STEPS, min(MAX_STEPS, self.n_images * STEPS_PER_IMAGE))

    def buckets_for(self, resolution: int) -> list[int]:
        """Bucket ladder capped at `resolution` — never upscale past the
        training resolution, and never past what the images actually contain."""
        ceiling = min(resolution, self.max_long_side or resolution)
        ladder = [b for b in BUCKET_LADDER if b <= ceiling]
        if resolution <= ceiling and resolution not in ladder:
            ladder.append(resolution)
        return sorted(set(ladder)) or [resolution]

    def summary(self) -> str:
        if not self.n_images:
            return "No images found in that folder."
        shapes = ", ".join(f"{a} ×{n}" for a, n in self.aspect_counts.most_common(4))
        lines = [
            f"**{self.n_images} images** ({self.n_captioned} with captions)",
            f"Long side: {self.min_long_side}–{self.max_long_side}px",
            f"Aspect ratios: {shapes}",
            f"Suggested steps: **{self.suggested_steps}** "
            f"({self.n_images} images × {STEPS_PER_IMAGE}, clamped to "
            f"{MIN_STEPS}–{MAX_STEPS})",
        ]
        if self.n_captioned < self.n_images:
            lines.append(f"⚠️ {self.n_images - self.n_captioned} image(s) have no "
                         f"`.txt` caption and will be trained with an empty caption.")
        return "  \n".join(lines)


def _aspect_label(w: int, h: int) -> str:
    if w == h:
        return "square"
    ratio = w / h
    known = {"3:2": 1.5, "4:3": 4 / 3, "16:9": 16 / 9, "2:3": 2 / 3,
             "3:4": 0.75, "9:16": 9 / 16}
    label, _ = min(known.items(), key=lambda kv: abs(kv[1] - ratio))
    return label


def inspect(dataset_dir: Path) -> DatasetStats:
    """Read image dimensions + caption coverage for `dataset_dir`.

    Pillow parses headers lazily, so this never decodes pixel data. Unreadable
    files are skipped rather than raising — a stray non-image shouldn't break
    config generation.
    """
    images = list_images(dataset_dir)
    stats = DatasetStats(n_images=0, n_captioned=0)
    for path in images:
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception:
            continue
        stats.n_images += 1
        stats.sizes.append(size)
        stats.aspect_counts[_aspect_label(*size)] += 1
        if path.with_suffix(".txt").exists():
            stats.n_captioned += 1
    return stats
