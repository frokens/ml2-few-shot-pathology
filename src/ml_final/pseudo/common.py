"""Shared helpers for Scheme 03 pseudo-label workflows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ml_final.utils.config import resolve_project_path
from ml_final.utils.paths import project_relative


DEFAULT_CLASS_NAMES = [f"Class_{idx}" for idx in range(5)]


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a two-dimensional array."""

    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def probabilities_to_logits(probs: np.ndarray) -> np.ndarray:
    """Convert probabilities to pseudo-logits for post-hoc temperature scaling."""

    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return np.log(clipped)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probability vectors."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return stable_softmax(probabilities_to_logits(probs) / float(temperature))


def nll_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean multiclass negative log likelihood."""

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    picked = probs[np.arange(len(y_true)), y_true]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def entropy(probs: np.ndarray) -> np.ndarray:
    """Row-wise predictive entropy."""

    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Expected calibration error using equal-width confidence bins."""

    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    ece = 0.0
    for start, end in zip(np.linspace(0.0, 1.0, n_bins, endpoint=False), np.linspace(1 / n_bins, 1.0, n_bins)):
        if math.isclose(end, 1.0):
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)
        if not np.any(mask):
            continue
        bin_conf = float(confidence[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += float(mask.mean()) * abs(bin_conf - bin_acc)
    return float(ece)


def top2_stats(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Return top-1/top-2 labels and probabilities."""

    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1] if probs.shape[1] > 1 else order[:, 0]
    prob_top1 = probs[np.arange(len(probs)), top1]
    prob_top2 = probs[np.arange(len(probs)), top2]
    return {
        "top1": top1,
        "top2": top2,
        "prob_top1": prob_top1,
        "prob_top2": prob_top2,
        "margin": prob_top1 - prob_top2,
        "entropy": entropy(probs),
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dictionaries as a stable UTF-8 CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_existing_paths(paths: list[str] | None) -> list[Path]:
    """Resolve project-relative paths and keep existing files only."""

    resolved: list[Path] = []
    for item in paths or []:
        path = resolve_project_path(item)
        if path is not None and path.exists():
            resolved.append(path)
    return resolved


def discover_npz_files(
    *,
    explicit_files: list[str] | None,
    search_dir: str | None,
    pattern: str,
    allow_search: bool = False,
) -> list[Path]:
    """Resolve explicit prediction files or discover them under a directory."""

    files = resolve_existing_paths(explicit_files)
    if files:
        return sorted(files)
    if not allow_search:
        return []
    root = resolve_project_path(search_dir) if search_dir else None
    if root is None or not root.exists():
        return []
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def project_rel_list(paths: list[Path]) -> list[str]:
    """Convert paths to project-relative POSIX strings."""

    return [project_relative(path) for path in paths]
