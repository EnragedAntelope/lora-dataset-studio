"""Tests for the WD tagger backend (pure logic + glue, no ONNX/download)."""

from __future__ import annotations

from pathlib import Path

from studio.captioner import Captioner
from studio.config import CAPTIONERS_BY_KEY
from studio.tagger import (
    CATEGORY_CHARACTER,
    CATEGORY_GENERAL,
    CATEGORY_RATING,
    WDTagger,
    _read_selected_tags,
    format_tag,
    select_tag_names,
)


# ---------- pure selection / formatting ----------

def test_select_tag_names_thresholds_and_order() -> None:
    labels = [
        ("long_hair", CATEGORY_GENERAL, 0.9),
        ("smile", CATEGORY_GENERAL, 0.5),
        ("blurry", CATEGORY_GENERAL, 0.1),        # below general threshold
        ("hatsune_miku", CATEGORY_CHARACTER, 0.95),
        ("some_char", CATEGORY_CHARACTER, 0.5),   # below character threshold
        ("explicit", CATEGORY_RATING, 0.8),       # ratings dropped by default
    ]
    out = select_tag_names(labels, general_threshold=0.35, character_threshold=0.85)
    # General (by confidence desc) then character; ratings excluded.
    assert out == ["long_hair", "smile", "hatsune_miku"]


def test_select_tag_names_can_include_top_rating() -> None:
    labels = [
        ("safe", CATEGORY_RATING, 0.2),
        ("explicit", CATEGORY_RATING, 0.7),
        ("solo", CATEGORY_GENERAL, 0.9),
    ]
    out = select_tag_names(labels, 0.35, 0.85, include_ratings=True)
    assert out == ["solo", "explicit"]  # only the most-likely rating appended


def test_format_tag_underscores_to_spaces_but_keeps_kaomoji() -> None:
    assert format_tag("long_hair") == "long hair"
    assert format_tag("^_^") == "^_^"


def test_read_selected_tags(tmp_path: Path) -> None:
    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "tag_id,name,category,count\n1,long_hair,0,100\n2,miku,4,50\n",
        encoding="utf-8")
    names, cats = _read_selected_tags(csv_path)
    assert names == ["long_hair", "miku"]
    assert cats == [0, 4]


# ---------- registry ----------

def test_wd_tagger_specs_registered() -> None:
    for key in ("wd-eva02", "wd-vit"):
        spec = CAPTIONERS_BY_KEY[key]
        assert spec.backend == "wd_tagger"
        assert spec.hf_id.startswith("SmilingWolf/")
        assert spec.nsfw_capable  # WD taggers include explicit tags


# ---------- glue (fake session, no onnxruntime import) ----------

class _FakeInput:
    name = "input"
    shape = [1, 448, 448, 3]


class _FakeSession:
    def __init__(self, preds: list[float]) -> None:
        self._preds = preds

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, _inputs):
        return [[self._preds]]  # shape (1, n_tags)


def test_wdtagger_tag_pipeline(monkeypatch) -> None:
    wd = WDTagger("fake/repo")
    # Pre-seed the session so .load() short-circuits (never imports onnxruntime).
    wd._session = _FakeSession([0.9, 0.5, 0.95, 0.1])
    wd._tag_names = ["long_hair", "smile", "hatsune_miku", "blurry"]
    wd._categories = [0, 0, 4, 0]
    monkeypatch.setattr(wd, "_preprocess", lambda _p: None)
    assert wd.tag(Path("x.png")) == ["long hair", "smile", "hatsune miku"]


def test_captioner_wd_tagger_returns_joined_tags(monkeypatch) -> None:
    cap = Captioner("wd-eva02")

    class FakeTagger:
        def tag(self, _p):
            return ["long hair", "smile"]

    monkeypatch.setattr(cap, "_load_tagger", lambda: FakeTagger())
    # Backend is a tagger: caption() ignores subject/style and emits tags.
    assert cap.caption(Path("x.png"), subject="whatever", style="prose") == "long hair, smile"
