"""Tests for the prose vs booru-tag caption style."""

from __future__ import annotations

from pathlib import Path

from studio import captioner as C
from studio.captioner import (
    SUBJECT_ALIASES,
    _normalize_tags,
    apply_affixes,
    caption_images,
    finalize_caption,
)
from studio.config import CAPTIONERS_BY_KEY


# ---------- template selection ----------

def test_prompt_for_selects_template_per_style() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    assert spec.prompt_for("tags") == spec.tags_template
    assert spec.prompt_for("e621") == spec.e621_template
    assert spec.prompt_for("prose") == spec.prompt_template
    # Unknown style must not crash — fall back to prose.
    assert spec.prompt_for("nonsense") == spec.prompt_template


def test_tags_and_e621_templates_are_distinct() -> None:
    spec = CAPTIONERS_BY_KEY["qwen3vl"]
    # e621 is a different vocabulary, not an alias for the Danbooru option.
    assert spec.e621_template != spec.tags_template
    assert "e621" in spec.e621_template.lower()


def test_every_captioner_has_a_subject_slot_in_all_templates() -> None:
    for spec in CAPTIONERS_BY_KEY.values():
        assert "{subject}" in spec.prompt_template
        assert "{subject}" in spec.tags_template
        assert "{subject}" in spec.e621_template


def test_joycaption_and_nsfw_have_distinct_tag_templates() -> None:
    joy = CAPTIONERS_BY_KEY["joycaption"]
    nsfw = CAPTIONERS_BY_KEY["qwen3vl-nsfw"]
    generic = CAPTIONERS_BY_KEY["qwen3vl"]
    # Overrides applied, not the generic default — for both tag vocabularies.
    assert joy.tags_template != generic.tags_template
    assert joy.e621_template != generic.e621_template
    assert "explicit" in nsfw.tags_template.lower()
    assert "explicit" in nsfw.e621_template.lower()


# ---------- tag normalization ----------

def test_normalize_tags_dedupes_lowercases_and_trims() -> None:
    raw = "From Side,  full body , From Side\noutdoor scene,"
    assert _normalize_tags(raw) == ["from side", "full body", "outdoor scene"]


def test_normalize_tags_drops_empties_and_trailing_periods() -> None:
    assert _normalize_tags("a, , b., ,c") == ["a", "b", "c"]


# ---------- finalize (tags) ----------

def test_finalize_tags_puts_trigger_first() -> None:
    out = finalize_caption("standing, from above, park", "sysnootles",
                           "Sy Snootles", SUBJECT_ALIASES, style="tags")
    assert out == "sysnootles, standing, from above, park"


def test_finalize_tags_dedupes_model_emitted_trigger() -> None:
    # If the model happens to emit the trigger, it appears once, first.
    out = finalize_caption("SysNootles, standing, from above", "sysnootles",
                           "", SUBJECT_ALIASES, style="tags")
    assert out == "sysnootles, standing, from above"


def test_finalize_tags_without_trigger() -> None:
    out = finalize_caption("standing, from above", "", "", SUBJECT_ALIASES, style="tags")
    assert out == "standing, from above"


def test_finalize_e621_uses_the_same_tag_formatting_as_tags() -> None:
    # e621 differs only in the vocabulary requested; the output shape (dedupe,
    # lowercase, trigger-first, comma-joined) is identical to the Danbooru path.
    raw = "Anthro, From Side, anthro, forest"
    tags = finalize_caption(raw, "trig", "", SUBJECT_ALIASES, style="tags")
    e621 = finalize_caption(raw, "trig", "", SUBJECT_ALIASES, style="e621")
    assert tags == e621 == "trig, anthro, from side, forest"


def test_finalize_tags_does_not_inject_character_name() -> None:
    # Identity is the trigger; the name is not added as a tag (unlike prose).
    out = finalize_caption("standing, from above", "sks", "Sy Snootles",
                           SUBJECT_ALIASES, style="tags")
    assert "Sy Snootles" not in out
    assert out.startswith("sks, ")


# ---------- prose path unchanged ----------

def test_finalize_prose_still_prepends_trigger_and_names() -> None:
    out = finalize_caption("The woman stands in a park.", "sysnootles",
                           "Sy Snootles", SUBJECT_ALIASES)
    assert out.startswith("sysnootles, ")
    assert "Sy Snootles" in out  # alias replacement still runs in prose mode


def test_finalize_default_style_is_prose() -> None:
    # No style arg → prose behaviour: the sentence's first letter is lowercased
    # after the trigger. The tags path would instead lowercase the whole thing.
    raw = "A Person Standing Outside."
    prose = finalize_caption(raw, "trig", "", SUBJECT_ALIASES)
    tags = finalize_caption(raw, "trig", "", SUBJECT_ALIASES, style="tags")
    assert prose == "trig, a Person Standing Outside."
    assert tags == "trig, a person standing outside"


# ---------- prefix / suffix affixes ----------

def test_apply_affixes_tags_uses_comma() -> None:
    out = apply_affixes("trig, standing", "score_9, score_8_up", "", "tags")
    assert out == "score_9, score_8_up, trig, standing"


def test_apply_affixes_prose_uses_space_and_suffix() -> None:
    out = apply_affixes("trig, a person", "photo of", "high quality", "prose")
    assert out == "photo of trig, a person high quality"


def test_apply_affixes_empty_is_noop() -> None:
    assert apply_affixes("trig, standing", "", "  ", "tags") == "trig, standing"


# ---------- caption_images: skip_existing + affixes (no real model) ----------

def _stub_model(monkeypatch, text: str) -> None:
    monkeypatch.setattr(C.Captioner, "caption",
                        lambda self, image_path, subject="the character", style="prose": text)
    monkeypatch.setattr(C.Captioner, "load", lambda self: None)
    monkeypatch.setattr(C.Captioner, "unload", lambda self: None)


def test_caption_images_skip_existing(tmp_path: Path, monkeypatch) -> None:
    imgs = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        p.write_bytes(b"x")
        imgs.append(p)
    (tmp_path / "0.txt").write_text("already captioned", encoding="utf-8")
    _stub_model(monkeypatch, "trig, tag")  # gemini backend -> no real load
    items = caption_images(imgs, "gemini-flash", trigger="trig", skip_existing=True)
    done = {p.name for p, _ in items}
    assert done == {"1.png", "2.png"}  # the already-captioned 0.png is skipped


def test_caption_images_applies_prefix(tmp_path: Path, monkeypatch) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(b"x")
    _stub_model(monkeypatch, "a, b")
    items = caption_images([p], "gemini-flash", trigger="trig", style="tags",
                           prefix="score_9")
    assert items[0][1] == "score_9, trig, a, b"
