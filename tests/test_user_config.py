"""Tests for user_config persistence (paths + last training settings)."""

from __future__ import annotations

from pathlib import Path

import pytest

from studio import user_config


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / ".cache"
    monkeypatch.setattr(user_config, "CACHE_DIR", cache)
    monkeypatch.setattr(user_config, "USER_SETTINGS_FILE", cache / "user_settings.json")


def test_missing_file_returns_empty() -> None:
    assert user_config.load_user_config() == {}


def test_trainer_path_round_trip() -> None:
    user_config.set_trainer_path("ai-toolkit", r"C:\ai-toolkit")
    assert user_config.get_trainer_path("ai-toolkit") == r"C:\ai-toolkit"
    assert user_config.get_trainer_path("musubi") == ""


def test_saves_merge_not_overwrite() -> None:
    user_config.set_trainer_path("musubi", "/opt/musubi")
    user_config.set_last_train_settings({"steps": 1234})
    assert user_config.get_trainer_path("musubi") == "/opt/musubi"
    assert user_config.get_last_train_settings()["steps"] == 1234


def test_corrupt_file_is_safe() -> None:
    user_config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    user_config.USER_SETTINGS_FILE.write_text("{ not json", encoding="utf-8")
    assert user_config.load_user_config() == {}
