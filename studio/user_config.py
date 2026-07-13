"""Persist non-secret user preferences (trainer install paths, last-used
training settings) to `.cache/user_settings.json`.

Deliberately narrow: this file holds filesystem paths and hyperparameter
numbers only. API keys never go here — they stay in `.env`/environment. The
cache dir is gitignored, so nothing written here is ever committed.
"""

from __future__ import annotations

import json
from typing import Any

from studio.config import CACHE_DIR

USER_SETTINGS_FILE = CACHE_DIR / "user_settings.json"


def load_user_config() -> dict[str, Any]:
    """Return the saved settings, or {} if missing/corrupt (never raises)."""
    if not USER_SETTINGS_FILE.exists():
        return {}
    try:
        data = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_user_config(data: dict[str, Any]) -> None:
    """Merge `data` into the saved settings and write atomically-ish."""
    current = load_user_config()
    current.update(data)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    USER_SETTINGS_FILE.write_text(
        json.dumps(current, indent=2, default=str), encoding="utf-8"
    )


def get_trainer_path(trainer: str) -> str:
    return load_user_config().get("trainer_paths", {}).get(trainer, "")


def set_trainer_path(trainer: str, path: str) -> None:
    cfg = load_user_config()
    paths = cfg.get("trainer_paths", {})
    paths[trainer] = path
    save_user_config({"trainer_paths": paths})


def get_last_train_settings() -> dict[str, Any]:
    return load_user_config().get("last_train", {})


def set_last_train_settings(settings: dict[str, Any]) -> None:
    save_user_config({"last_train": settings})
