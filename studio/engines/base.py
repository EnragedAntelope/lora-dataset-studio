"""Engine protocol shared by the local ComfyUI and Nano Banana engines."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from studio.shotplan import Shot


class GenerationError(Exception):
    """A shot failed permanently (after retry)."""


class Engine(Protocol):
    name: str

    def generate(self, sources: list[Path], shot: Shot, out_path: Path, seed: int) -> Path:
        """Generate one image for `shot` using `sources` as identity references.

        Writes the result to `out_path` and returns it. Raises GenerationError
        on permanent failure (e.g. content refusal).
        """
        ...
