"""Tests for the expanded trainer registry: SDXL presets (Item 3)."""

from __future__ import annotations

from pathlib import Path

import yaml

from studio.trainer_configs import TRAINER_MODELS, TrainConfig, render_aitoolkit_yaml


def _ai_preset(key: str):
    return next(p for p in TRAINER_MODELS["ai-toolkit"] if p.key == key)


def _cfg(preset, tmp_path: Path) -> TrainConfig:
    return TrainConfig(trainer="ai-toolkit", model=preset, dataset_dir=tmp_path,
                       name="x-lora", trigger="trig")


def test_sdxl_preset_exists_and_is_honest() -> None:
    sdxl = _ai_preset("sdxl")
    assert sdxl.arch == "sdxl"
    assert sdxl.quantize is False
    assert sdxl.name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
    # The family placeholder is honest about needing a path.
    assert "<<FILL" in _ai_preset("sdxl-custom").name_or_path


def test_sdxl_renders_ddpm_and_higher_cfg(tmp_path: Path) -> None:
    doc = yaml.safe_load(render_aitoolkit_yaml(_cfg(_ai_preset("sdxl"), tmp_path)))
    proc = doc["config"]["process"][0]
    assert proc["model"]["arch"] == "sdxl"
    assert proc["model"]["quantize"] is False
    assert proc["train"]["noise_scheduler"] == "ddpm"
    assert proc["sample"]["guidance_scale"] == 7
    assert proc["sample"]["sample_steps"] == 25


def test_flow_matching_defaults_unchanged(tmp_path: Path) -> None:
    # Existing flow-matching presets must render exactly as before the refactor.
    yaml_text = render_aitoolkit_yaml(_cfg(_ai_preset("flux-dev"), tmp_path))
    assert "noise_scheduler: flowmatch" in yaml_text
    assert "guidance_scale: 4" in yaml_text
    assert "sample_steps: 20" in yaml_text
