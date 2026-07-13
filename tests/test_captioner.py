"""Tests for captioner registry configuration."""

from __future__ import annotations

from studio.config import CAPTIONERS_BY_KEY


def test_groq_qwen3_6_spec_exists() -> None:
    spec = CAPTIONERS_BY_KEY["groq-qwen3.6"]
    assert spec.model == "qwen/qwen3.6-27b"
    assert spec.backend == "openai"
    assert spec.base_url == "https://api.groq.com/openai/v1"
    assert spec.api_key_env == "GROQ_API_KEY"


def test_groq_qwen3_6_rate_interval_is_conservative() -> None:
    spec = CAPTIONERS_BY_KEY["groq-qwen3.6"]
    assert spec.min_interval_s >= 3.0


def test_groq_scout_rate_interval_exists() -> None:
    spec = CAPTIONERS_BY_KEY["groq-llama4-scout"]
    assert spec.min_interval_s > 0
