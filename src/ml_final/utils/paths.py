"""Path utilities — project root resolution and artifact discovery."""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the absolute path to the project root.

    Resolves from this file's location: src/ml_final/utils/paths.py
    -> project root is four levels up: ../../../
    """
    return Path(__file__).resolve().parent.parent.parent.parent


def model_registry_dir() -> Path:
    return project_root() / "artifacts" / "model_registry"


def configs_dir() -> Path:
    return project_root() / "configs"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_relative(path: str | Path) -> str:
    """Return a POSIX path relative to the project root when possible."""
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    try:
        return path_obj.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path_obj.as_posix()
