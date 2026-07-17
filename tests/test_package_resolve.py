from pathlib import Path

from studio.package import ExportResolution, resolve_export_items


def _img(folder: Path, name: str, caption: str | None) -> Path:
    p = folder / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n")  # header bytes; not decoded here
    if caption is not None:
        p.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return p


def test_all_captioned(tmp_path):
    a = _img(tmp_path, "01.png", "sysnootles, front view")
    b = _img(tmp_path, "02.png", "sysnootles, side view")
    res = resolve_export_items([a, b])
    assert isinstance(res, ExportResolution)
    assert res.items == [(a, "sysnootles, front view"), (b, "sysnootles, side view")]
    assert res.empties == []
    assert res.missing == []


def test_missing_sidecar_excluded(tmp_path):
    a = _img(tmp_path, "01.png", "sysnootles")
    b = _img(tmp_path, "02.png", None)  # no .txt
    res = resolve_export_items([a, b])
    assert res.items == [(a, "sysnootles")]
    assert res.missing == ["02.png"]
    assert res.empties == []


def test_blank_sidecar_is_empty(tmp_path):
    a = _img(tmp_path, "01.png", "   \n ")  # whitespace only
    res = resolve_export_items([a])
    assert res.items == []
    assert res.empties == ["01.png"]
    assert res.missing == []


def test_same_name_two_folders_no_collision(tmp_path):
    f1 = tmp_path / "prepped"
    f1.mkdir()
    f2 = tmp_path / "generated"
    f2.mkdir()
    a = _img(f1, "01.png", "from prepped")
    b = _img(f2, "01.png", "from generated")
    res = resolve_export_items([a, b])
    assert res.items == [(a, "from prepped"), (b, "from generated")]


def test_order_preserved(tmp_path):
    imgs = [_img(tmp_path, f"{i:02d}.png", f"cap {i}") for i in (3, 1, 2)]
    res = resolve_export_items(imgs)
    assert [c for _, c in res.items] == ["cap 3", "cap 1", "cap 2"]
