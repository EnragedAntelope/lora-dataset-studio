"""Tests for the preprocess stage (I/O-light paths: no restore, no isolation)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from studio.preprocess import preprocess


def _img(path: Path, size: tuple[int, int] = (64, 48),
         color: tuple[int, int, int] = (120, 120, 120)) -> None:
    Image.new("RGB", size, color).save(path)


def test_preprocess_same_stem_different_ext_no_clobber(tmp_path: Path) -> None:
    """cat.jpg + cat.png both map to `cat_prepped.png` naively; the second must
    not silently overwrite the first — both images have to survive."""
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    a, b = src / "cat.jpg", src / "cat.png"
    _img(a, color=(200, 50, 50))
    _img(b, color=(50, 50, 200))

    r1 = preprocess(a, out, target=32, force_restore=False, isolate=False)
    r2 = preprocess(b, out, target=32, force_restore=False, isolate=False)

    assert r1.output != r2.output
    assert r1.output.exists() and r2.output.exists()
    assert r1.output.name == "cat_prepped.png"
    assert r2.output.name == "cat_prepped_2.png"
    assert {p.name for p in out.glob("*.png")} == {"cat_prepped.png", "cat_prepped_2.png"}
