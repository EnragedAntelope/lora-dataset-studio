"""Tests for trainer config generation."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from studio.trainer_configs import (
    TRAINER_MODELS,
    TrainConfig,
    render_aitoolkit_yaml,
    render_musubi_toml,
    write_configs,
)


def _cfg(trainer: str, tmp_path: Path) -> TrainConfig:
    return TrainConfig(trainer=trainer, model=TRAINER_MODELS[trainer][0],
                       dataset_dir=tmp_path, trigger="sysnootles", name="sy-lora",
                       resolution=1024, rank=16, alpha=16, steps=1500, lr=1e-4)


def test_aitoolkit_yaml_parses_and_has_keys(tmp_path: Path) -> None:
    doc = yaml.safe_load(render_aitoolkit_yaml(_cfg("ai-toolkit", tmp_path)))
    proc = doc["config"]["process"][0]
    assert proc["type"] == "sd_trainer"
    assert proc["network"]["linear"] == 16
    assert proc["train"]["steps"] == 1500
    assert proc["datasets"][0]["folder_path"] == tmp_path.as_posix()
    assert proc["model"]["name_or_path"]  # non-empty


def test_musubi_toml_parses_and_has_dataset(tmp_path: Path) -> None:
    doc = tomllib.loads(render_musubi_toml(_cfg("musubi", tmp_path)))
    assert doc["general"]["caption_extension"] == ".txt"
    assert doc["datasets"][0]["image_directory"] == tmp_path.as_posix()


def test_write_configs_writes_file_and_command(tmp_path: Path) -> None:
    written, command = write_configs(_cfg("ai-toolkit", tmp_path), install_path=r"C:\ai-toolkit")
    assert written[0].exists()
    assert written[0].name == "ai-toolkit.yaml"
    assert "ai-toolkit" in command and "run.py" in command


def test_write_configs_never_clobbers(tmp_path: Path) -> None:
    write_configs(_cfg("musubi", tmp_path))
    written2, _ = write_configs(_cfg("musubi", tmp_path))
    assert written2[0].name == "dataset.2.toml"


def test_musubi_command_flags_model_paths(tmp_path: Path) -> None:
    _, command = write_configs(_cfg("musubi", tmp_path), install_path="/opt/musubi")
    assert "<<FILL:" in command  # honest about needing model paths
    assert "/opt/musubi" in command


def test_unknown_trainer_raises(tmp_path: Path) -> None:
    cfg = _cfg("ai-toolkit", tmp_path)
    cfg.trainer = "nope"
    with pytest.raises(ValueError):
        write_configs(cfg)
