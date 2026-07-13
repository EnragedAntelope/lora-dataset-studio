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


def test_custom_captioner_round_trip() -> None:
    user_config.set_custom_captioner(
        "https://openrouter.ai/api/v1/", "qwen/qwen2.5-vl-72b-instruct",
        "OPENROUTER_API_KEY", 2.5)
    cfg = user_config.get_custom_captioner()
    assert cfg["base_url"] == "https://openrouter.ai/api/v1"  # trailing slash stripped
    assert cfg["model"] == "qwen/qwen2.5-vl-72b-instruct"
    assert cfg["api_key_env"] == "OPENROUTER_API_KEY"
    assert cfg["min_interval_s"] == 2.5


def test_custom_captioner_never_stores_a_key() -> None:
    # Only the env-var NAME is persisted; the secret itself must never be here.
    user_config.set_custom_captioner("https://x/v1", "m", "MY_KEY_ENV", 0)
    raw = user_config.USER_SETTINGS_FILE.read_text(encoding="utf-8")
    assert "MY_KEY_ENV" in raw
    assert "api_key" in raw  # the env-var-name field
    # Sanity: the persisted dict has exactly the four non-secret fields.
    assert set(user_config.get_custom_captioner()) == set(user_config._CUSTOM_KEYS)
