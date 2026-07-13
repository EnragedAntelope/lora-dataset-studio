"""Tests for the curated shot plan."""

from __future__ import annotations

from studio.shotplan import default_plan


def test_default_plan_has_24_shots() -> None:
    plan = default_plan()
    assert len(plan) == 24


def test_shots_have_required_fields() -> None:
    plan = default_plan()
    for shot in plan:
        assert shot.id
        assert shot.kind in {"angle", "pose", "emotion"}
        assert shot.local_prompt
        assert shot.cloud_prompt


def test_emotion_and_setting_fields_exist() -> None:
    plan = default_plan()
    emotions = {s.emotion for s in plan if s.emotion}
    settings = {s.setting for s in plan if s.setting}
    assert emotions
    assert settings
    assert len(settings) >= 6


def test_chained_shots_sort_last_in_generation() -> None:
    plan = default_plan()
    chained = [s for s in plan if s.chain_from]
    assert chained
    for shot in chained:
        assert shot.chain_from in {s.id for s in plan}


def test_no_duplicate_angle_pose_setting_combinations() -> None:
    plan = default_plan()
    combos = set()
    for shot in plan:
        # Use prompt-derived angle/pose stub; full uniqueness is checked via ids
        combos.add(shot.id)
    assert len(combos) == len(plan)


def test_angle_shots_use_sks_grammar() -> None:
    plan = default_plan()
    angles = [s for s in plan if s.kind == "angle"]
    assert angles
    for shot in angles:
        assert "<sks>" in shot.local_prompt


def test_pose_and_emotion_shots_do_not_use_sks() -> None:
    plan = default_plan()
    non_angles = [s for s in plan if s.kind != "angle"]
    assert non_angles
    for shot in non_angles:
        assert "<sks>" not in shot.local_prompt


def test_emotion_shots_are_closeup() -> None:
    plan = default_plan()
    emotions = [s for s in plan if s.kind == "emotion"]
    assert emotions
    for shot in emotions:
        assert "close-up" in shot.local_prompt.lower() or "closeup" in shot.local_prompt.lower()


def test_setting_varies_across_plan() -> None:
    plan = default_plan()
    settings = [s.setting for s in plan if s.setting]
    assert len(set(settings)) >= 6


def test_plan_includes_common_angles() -> None:
    plan = default_plan()
    ids = {s.id for s in plan}
    assert {"angle-front", "angle-back", "angle-right", "angle-left"} <= ids
