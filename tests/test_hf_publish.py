"""Tests for HF dataset publishing validation/guards (Item 5).

No network: the real upload path is exercised only on a live publish. These
cover the pure validation and the pre-upload guards that must fail fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studio.hf_publish import (
    HFPublishError,
    normalize_repo_id,
    publish_dataset,
    resolve_token,
)


def test_normalize_repo_id_trims_and_accepts_valid() -> None:
    assert normalize_repo_id("  my-character-lora ") == "my-character-lora"
    assert normalize_repo_id("user/name") == "user/name"
    assert normalize_repo_id("/user/name/") == "user/name"


def test_normalize_repo_id_rejects_empty() -> None:
    with pytest.raises(HFPublishError):
        normalize_repo_id("   ")


@pytest.mark.parametrize("bad", ["a/b/c", "has space", "bad$char", "-leading"])
def test_normalize_repo_id_rejects_malformed(bad: str) -> None:
    with pytest.raises(HFPublishError):
        normalize_repo_id(bad)


def test_resolve_token_prefers_explicit(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert resolve_token("explicit-token") == "explicit-token"


def test_resolve_token_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert resolve_token() == "env-token"


def test_publish_requires_a_token(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ds = tmp_path / "d"
    ds.mkdir()
    with pytest.raises(HFPublishError, match="token"):
        publish_dataset(ds, "my-lora")


def test_publish_requires_existing_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "env-token")
    with pytest.raises(HFPublishError, match="not found"):
        publish_dataset(tmp_path / "nope", "my-lora")
