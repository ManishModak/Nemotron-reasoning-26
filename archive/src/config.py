"""Minimal config loading helpers for JSON-compatible YAML files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path_like: str | Path) -> dict[str, Any]:
    """Load JSON or JSON-compatible YAML without forcing a PyYAML dependency."""

    path = Path(path_like)
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must deserialize into a mapping: {path}")
    return payload

