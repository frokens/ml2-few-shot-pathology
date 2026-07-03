"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal local smoke envs
    yaml = None

from ml_final.utils.paths import project_root


def resolve_project_path(path: str | Path | None) -> Path | None:
    """Resolve a path relative to the project root unless it is absolute."""
    if path is None:
        return None
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return project_root() / path_obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file using project-relative path semantics."""
    resolved = resolve_project_path(path)
    if resolved is None:
        raise ValueError("config path cannot be None")
    if not resolved.exists():
        raise FileNotFoundError(f"config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        if resolved.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            if yaml is None:
                raise ImportError(
                    "pyyaml is required for YAML configs. Use a .json smoke config "
                    "locally or install dependencies with `pip install -e .`."
                )
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {resolved}")
    extends = data.pop("_extends", None)
    if extends:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            parent_path = resolved.parent / parent_path
        parent = load_yaml(parent_path)
        data = deep_merge(parent, data)
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating inputs."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
