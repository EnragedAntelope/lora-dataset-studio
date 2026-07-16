"""Tests for the best-effort GitHub-release update check."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from studio import update_check


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_file = tmp_path / "update_check.json"
    monkeypatch.setattr(update_check, "UPDATE_CACHE_FILE", cache_file)
    monkeypatch.setattr(update_check.settings, "update_check_enabled", True)
    monkeypatch.setattr(update_check, "__version__", "0.4.0")


def _write_cache(latest: str, *, hours_old: float = 0) -> None:
    cached_at = datetime.now(tz=timezone.utc) - timedelta(hours=hours_old)
    update_check.UPDATE_CACHE_FILE.write_text(
        json.dumps({"cached_at": cached_at.isoformat(), "latest": latest}),
        encoding="utf-8",
    )


class _FakeResponse:
    def __init__(self, tag: str) -> None:
        self._tag = tag

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"tag_name": self._tag}


def test_parse_version() -> None:
    assert update_check._parse_version("v0.4.0") == (0, 4, 0)
    assert update_check._parse_version("0.10.2") == (0, 10, 2)


def test_is_newer() -> None:
    assert update_check._is_newer("v0.5.0", "0.4.0")
    assert not update_check._is_newer("v0.4.0", "0.4.0")
    assert not update_check._is_newer("v0.3.9", "0.4.0")


def test_no_notice_when_up_to_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: "v0.4.0")
    assert update_check.check_for_update() is None
    assert update_check.update_banner_markdown() == ""


def test_notice_when_newer_release_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: "v0.5.0")
    info = update_check.check_for_update()
    assert info == {"current": "0.4.0", "latest": "0.5.0",
                    "url": update_check.RELEASES_PAGE_URL}
    banner = update_check.update_banner_markdown()
    assert "0.4.0" in banner and "0.5.0" in banner


def test_disabled_via_settings_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_check.settings, "update_check_enabled", False)

    def _boom() -> None:
        raise AssertionError("should not fetch when disabled")

    monkeypatch.setattr(update_check, "_fetch_latest_tag", _boom)
    assert update_check.check_for_update() is None


def test_network_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: None)
    assert update_check.check_for_update() is None
    assert update_check.update_banner_markdown() == ""


def test_fresh_cache_skips_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cache("v0.9.0", hours_old=1)

    def _boom() -> None:
        raise AssertionError("should not fetch when cache is fresh")

    monkeypatch.setattr(update_check, "_fetch_latest_tag", _boom)
    assert update_check.latest_version() == "v0.9.0"


def test_expired_cache_triggers_refetch(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cache("v0.4.0", hours_old=999)
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: "v0.6.0")
    assert update_check.latest_version() == "v0.6.0"


def test_expired_cache_falls_back_to_stale_on_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cache("v0.4.5", hours_old=999)
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: None)
    assert update_check.latest_version() == "v0.4.5"


def test_force_refresh_bypasses_fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cache("v0.4.0", hours_old=1)
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: "v0.7.0")
    assert update_check.latest_version(force_refresh=True) == "v0.7.0"


def test_corrupt_cache_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    update_check.UPDATE_CACHE_FILE.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(update_check, "_fetch_latest_tag", lambda: None)
    assert update_check.latest_version() is None
    assert update_check.check_for_update() is None


def test_fetch_latest_tag_uses_httpx_get(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def _fake_get(url, timeout=None, headers=None):
        calls["url"] = url
        calls["timeout"] = timeout
        calls["headers"] = headers
        return _FakeResponse("v1.2.3")

    monkeypatch.setattr(update_check.httpx, "get", _fake_get)
    assert update_check._fetch_latest_tag() == "v1.2.3"
    assert calls["url"] == update_check.RELEASES_API_URL
    assert calls["headers"]["User-Agent"]
