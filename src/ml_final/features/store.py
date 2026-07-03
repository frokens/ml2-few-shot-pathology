"""Feature store helpers based on compressed NumPy archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize feature rows."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def save_feature_bundle(
    path: str | Path,
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_filenames: np.ndarray,
    class_names: list[str],
    metadata: dict[str, Any],
    test_features: np.ndarray | None = None,
    test_filenames: np.ndarray | None = None,
    train_origin_filenames: np.ndarray | None = None,
    train_origin_indices: np.ndarray | None = None,
    train_eval_features: np.ndarray | None = None,
) -> Path:
    """Save a train/test feature bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "train_features": train_features,
        "train_labels": train_labels,
        "train_filenames": train_filenames,
        "class_names": np.asarray(class_names, dtype=object),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
    }
    if test_features is not None:
        arrays["test_features"] = test_features
    if test_filenames is not None:
        arrays["test_filenames"] = test_filenames
    if train_origin_filenames is not None:
        arrays["train_origin_filenames"] = train_origin_filenames
    if train_origin_indices is not None:
        arrays["train_origin_indices"] = train_origin_indices
    if train_eval_features is not None:
        arrays["train_eval_features"] = train_eval_features
    np.savez_compressed(path, **arrays)
    return path


def load_feature_bundle(path: str | Path) -> dict[str, Any]:
    """Load a feature bundle into memory."""
    with np.load(Path(path), allow_pickle=True) as data:
        bundle = {key: data[key] for key in data.files}
    if "metadata_json" in bundle:
        bundle["metadata"] = json.loads(str(bundle["metadata_json"].item()))
    return bundle
