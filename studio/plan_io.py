"""Save/load shot plans as YAML — user-editable prompt libraries.

A plan is just a list of `Shot` rows. Persisting it lets users keep and share
curated shot sets (community prompt libraries) without editing code. YAML is
chosen over JSON for hand-editability; `Shot` validation still runs on load, so
a malformed file fails loudly rather than producing garbage shots.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from studio.shotplan import Shot


def save_plan(shots: list[Shot], path: Path) -> Path:
    """Write `shots` to `path` as YAML. Returns the path written."""
    path = path.with_suffix(".yaml") if path.suffix.lower() not in {".yaml", ".yml"} else path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.model_dump() for s in shots]
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def load_plan(path: Path) -> list[Shot]:
    """Parse a YAML shot plan back into validated `Shot` objects."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} is not a shot-plan list (got {type(raw).__name__}).")
    return [Shot(**row) for row in raw]
