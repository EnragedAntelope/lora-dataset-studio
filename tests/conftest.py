"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / ".cache"
    cache.mkdir()
    return cache
