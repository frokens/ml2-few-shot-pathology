"""Helpers for locked Hugging Face Hub model loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ml_final.utils.config import load_yaml, resolve_project_path


def prepare_locked_hf_model_name(
    model_name: str,
    *,
    lock_key: str | None,
    lock_path: str | Path = "artifacts/model_registry/models.lock.yaml",
) -> str:
    """Return an `hf-hub:` model name pinned to the lockfile revision.

    The project stores model files under `external_models/model_store` and keeps
    the Hub cache separate.  timm's `hf-hub:` loader needs the revision in the
    model name when `HF_HUB_OFFLINE=1`; otherwise it tries to resolve `main`.
    When a lock entry exists, also expose the local files through the expected
    Hugging Face cache snapshot path using symlinks.
    """

    if not model_name.startswith("hf-hub:") or not lock_key:
        return model_name
    repo_part = model_name.removeprefix("hf-hub:")
    if "@" in repo_part:
        return model_name

    resolved_lock = resolve_project_path(lock_path)
    if resolved_lock is None or not resolved_lock.exists():
        return model_name
    lock = load_yaml(resolved_lock)
    entry: dict[str, Any] | None = (lock.get("models") or {}).get(lock_key)
    if not entry:
        return model_name

    revision = str(entry.get("revision") or "").strip()
    local_path = resolve_project_path(entry.get("local_path"))
    if not revision:
        return model_name
    if local_path and local_path.exists():
        cache_path = _resolve_cache_path(entry)
        _enable_local_hf_cache(cache_path)
        _link_snapshot_files(repo_part, revision, local_path, cache_path)
    return f"hf-hub:{repo_part}@{revision}"


def resolve_locked_hf_cache_dir(
    model_name: str,
    *,
    lock_key: str | None,
    lock_path: str | Path = "artifacts/model_registry/models.lock.yaml",
) -> str | None:
    """Return the lockfile HF cache directory for timm's explicit cache_dir."""

    if not model_name.startswith("hf-hub:") or not lock_key:
        return None
    resolved_lock = resolve_project_path(lock_path)
    if resolved_lock is None or not resolved_lock.exists():
        return None
    lock = load_yaml(resolved_lock)
    entry: dict[str, Any] | None = (lock.get("models") or {}).get(lock_key)
    if not entry:
        return None
    cache_path = _resolve_cache_path(entry)
    _enable_local_hf_cache(cache_path)
    return str(cache_path)


def _enable_local_hf_cache(cache_path: Path) -> None:
    """Prefer the locked local cache during formal offline experiment runs."""

    os.environ.setdefault("HF_HUB_CACHE", str(cache_path))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_cache_path(entry: dict[str, Any]) -> Path:
    cache_path = entry.get("cache_path") or os.environ.get("HF_HUB_CACHE")
    if cache_path:
        return Path(str(cache_path)).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    resolved = resolve_project_path("hf_home/hub")
    if resolved is None:
        raise ValueError("could not resolve HF cache path")
    return resolved


def _link_snapshot_files(repo_id: str, revision: str, local_path: Path, cache_path: Path) -> None:
    repo_cache = cache_path / ("models--" + repo_id.replace("/", "--"))
    snapshot_dir = repo_cache / "snapshots" / revision
    refs_dir = repo_cache / "refs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision + "\n", encoding="utf-8")

    for src in local_path.iterdir():
        if not src.is_file():
            continue
        dst = snapshot_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src)
        except OSError:
            # Files are large; prefer not to copy.  If symlink creation is not
            # supported, leave the original error surface to the timm loader.
            pass
