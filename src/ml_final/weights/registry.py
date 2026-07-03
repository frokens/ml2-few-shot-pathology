"""Model registry — parse, validate, and generate YAML manifests."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MODEL_KEYS = {"uni2_h", "virchow2", "conch", "h_optimus_0"}
VALID_SOURCES = {"official", "hf-mirror", "modelscope", "offline"}
VALID_HUBS = {"huggingface", "modelscope"}


def _validate_requested_entry(key: str, entry: dict[str, Any]) -> None:
    """Raise ValueError if a requested model entry is invalid."""
    required_fields = {"hub", "repo_id", "required", "gated", "license", "usage"}
    missing = required_fields - set(entry.keys())
    if missing:
        raise ValueError(f"Model '{key}' missing required fields: {missing}")
    if entry["hub"] not in VALID_HUBS:
        raise ValueError(
            f"Model '{key}': hub must be one of {VALID_HUBS}, got '{entry['hub']}'"
        )


def load_requested(path: str | Path) -> dict[str, Any]:
    """Load and validate models.requested.yaml.

    Returns the parsed document with a top-level 'models' key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model registry config not found: {path}")
    with open(path, "r") as f:
        doc = yaml.safe_load(f)
    if "models" not in doc:
        raise ValueError(f"Expected top-level 'models' key in {path}")
    for key, entry in doc["models"].items():
        _validate_requested_entry(key, entry)
    return doc


def load_lock(path: str | Path) -> dict[str, Any] | None:
    """Load models.lock.yaml if it exists, return None otherwise."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_lock(lock_data: dict[str, Any], path: str | Path) -> None:
    """Write models.lock.yaml."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(lock_data, f, sort_keys=False, default_flow_style=False)


def generate_lock_template(
    requested: dict[str, Any],
    source: str,
    store: str,
) -> dict[str, Any]:
    """Generate a lockfile template from requested models.

    This is used for --dry-run previews — it fills in placeholder values
    for fields that require actual downloads.
    """
    lock_models = {}
    for key, entry in requested["models"].items():
        repo_id = entry["repo_id"]
        safe_name = repo_id.replace("/", "__")
        lock_models[key] = {
            "repo_id": repo_id,
            "revision": "<resolved_after_download>",
            "source_endpoint": _endpoint_for_source(source, entry["hub"]),
            "local_path": os.path.join(store, "hf", safe_name, "<revision>"),
            "cache_path": "<auto_managed_by_hf>",
            "downloaded_at": datetime.datetime.now().isoformat(),
            "sha256_file": "artifacts/model_registry/SHA256SUMS",
            "license": entry["license"],
            "terms_accepted": True,
        }
    return {"models": lock_models}


def _endpoint_for_source(source: str, hub: str) -> str:
    if hub == "modelscope" or source == "modelscope":
        return "https://modelscope.cn"
    if source == "hf-mirror":
        return "https://hf-mirror.com"
    return "https://huggingface.co"
