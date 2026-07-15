"""Tests for the random unisex wardrobe pool."""

from __future__ import annotations

from studio import wardrobe


def test_outfits_are_distinct() -> None:
    # Repeats would partly defeat the point: the feature exists to stop the LoRA
    # binding one outfit to the character's identity.
    outfits = wardrobe.random_outfits(24, seed=1)
    assert len(outfits) == 24
    assert len(set(outfits)) == 24


def test_seed_is_reproducible() -> None:
    assert wardrobe.random_outfits(10, seed=7) == wardrobe.random_outfits(10, seed=7)


def test_different_seeds_differ() -> None:
    assert wardrobe.random_outfits(10, seed=1) != wardrobe.random_outfits(10, seed=2)


def test_zero_and_negative_return_empty() -> None:
    assert wardrobe.random_outfits(0) == []
    assert wardrobe.random_outfits(-3) == []


def test_outfit_reads_as_a_phrase() -> None:
    outfit = wardrobe.random_outfits(1, seed=3)[0]
    assert outfit.startswith(("a ", "an "))
    assert " and " in outfit


def test_article_agrees_with_the_colour() -> None:
    """These phrases are injected verbatim into prompts as 'wearing {outfit}',
    so 'a olive utility jacket' would be visible in the output."""
    for outfit in wardrobe.random_outfits(200, seed=13):
        first_word = outfit.split()[1]
        expected = "an" if first_word[0].lower() in "aeiou" else "a"
        assert outfit.startswith(f"{expected} "), outfit


def test_vowel_colours_get_an() -> None:
    assert wardrobe._article("olive") == "an"
    assert wardrobe._article("off-white") == "an"
    assert wardrobe._article("charcoal") == "a"
    assert wardrobe._article("navy") == "a"


def test_top_and_bottom_colors_differ() -> None:
    # Head-to-toe single colour reads as a costume, not clothing.
    for outfit in wardrobe.random_outfits(40, seed=5):
        top, _, bottom = outfit.partition(" and ")
        top_color = top.removeprefix("a ").split()[0]
        assert not bottom.startswith(top_color)


def test_vocabulary_is_gender_neutral() -> None:
    """One pool serves any character, so the UI needs no masc/femme picker."""
    gendered = {"dress", "skirt", "blouse", "gown", "heels", "suit and tie",
                "bra", "lingerie", "tuxedo", "menswear", "womenswear"}
    vocab = " ".join(wardrobe.TOPS + wardrobe.BOTTOMS).lower()
    assert not [g for g in gendered if g in vocab]


def test_large_request_degrades_to_repeats_rather_than_hanging() -> None:
    outfits = wardrobe.random_outfits(500, seed=11)
    assert len(outfits) == 500


def test_close_ups_are_excluded_from_wardrobe() -> None:
    # Emotion shots frame the face/upper shoulders; a full outfit description
    # there pulls the framing wider.
    assert "emotion" not in wardrobe.OUTFIT_SHOT_KINDS
    assert set(wardrobe.OUTFIT_SHOT_KINDS) == {"angle", "pose"}
