"""Tests for the optional ZIP export (Item 4)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from studio.package import zip_dataset


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "sy-dataset"
    ds.mkdir()
    (ds / "01.png").write_bytes(b"fakepng")
    (ds / "01.txt").write_text("sysnootles, standing", encoding="utf-8")
    (ds / "metadata.json").write_text("{}", encoding="utf-8")
    return ds


def test_zip_dataset_contains_files_under_named_folder(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    zp = zip_dataset(ds)
    assert zp.exists() and zp.name == "sy-dataset.zip"
    with zipfile.ZipFile(zp) as z:
        names = set(z.namelist())
    # Entries extract into a tidy folder named after the dataset.
    assert "sy-dataset/01.png" in names
    assert "sy-dataset/01.txt" in names
    assert "sy-dataset/metadata.json" in names


def test_zip_dataset_never_clobbers(tmp_path: Path) -> None:
    ds = _make_dataset(tmp_path)
    first = zip_dataset(ds)
    second = zip_dataset(ds)
    assert first.name == "sy-dataset.zip"
    assert second.name == "sy-dataset-2.zip"


def test_zip_dataset_rejects_non_folder(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(RuntimeError):
        zip_dataset(tmp_path / "does-not-exist")


def test_package_dataset_writes_hf_metadata_jsonl(tmp_path: Path) -> None:
    import json

    from PIL import Image

    from studio.package import package_dataset

    src = tmp_path / "src.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src)
    ds = package_dataset([(src, "trig, standing outdoors")], tmp_path / "out",
                         "Sy", "trig", {})
    lines = (ds / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0]) == {"file_name": "01.png", "text": "trig, standing outdoors"}
