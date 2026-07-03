"""OOF probability ensemble search utilities."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ml_final.metrics.classification import compute_classification_metrics


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Clip and row-normalize probability arrays."""

    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def average_probs(prob_arrays: list[np.ndarray]) -> np.ndarray:
    """Average already aligned probability arrays."""

    if not prob_arrays:
        raise ValueError("prob_arrays cannot be empty")
    return normalize_probabilities(np.mean([normalize_probabilities(item) for item in prob_arrays], axis=0))


def weighted_probs(prob_arrays: list[np.ndarray], weights: list[float] | np.ndarray) -> np.ndarray:
    """Weighted average for aligned probability arrays."""

    if not prob_arrays:
        raise ValueError("prob_arrays cannot be empty")
    weight_arr = np.asarray(weights, dtype=np.float64)
    if weight_arr.ndim != 1 or weight_arr.size != len(prob_arrays):
        raise ValueError("weights must be a 1D array matching prob_arrays")
    if np.any(weight_arr < 0):
        raise ValueError("weights must be non-negative")
    weight_arr = weight_arr / np.maximum(weight_arr.sum(), 1e-12)
    stacked = np.stack([normalize_probabilities(item) for item in prob_arrays], axis=0)
    probs = np.tensordot(weight_arr, stacked, axes=(0, 0))
    return normalize_probabilities(probs)


def grid_weight_search(
    prob_arrays: list[np.ndarray],
    y_true: np.ndarray,
    class_names: list[str],
    *,
    step: float = 0.1,
    max_sources: int = 5,
) -> dict[str, Any]:
    """Search non-negative weights on OOF predictions.

    The search is intentionally coarse.  With 250 labeled examples, a high
    capacity stacker can overfit the validation folds; a simplex grid keeps the
    choice auditable and bounded.
    """

    if not prob_arrays:
        raise ValueError("prob_arrays cannot be empty")
    if len(prob_arrays) > max_sources:
        prob_arrays = prob_arrays[:max_sources]
    n_sources = len(prob_arrays)
    units = int(round(1.0 / step))
    best: dict[str, Any] | None = None
    for weights in simplex_grid(n_sources, units):
        probs = weighted_probs(prob_arrays, weights)
        pred = probs.argmax(axis=1)
        metrics = compute_classification_metrics(y_true, pred, class_names)
        candidate = {
            "weights": [float(item) for item in weights],
            "metrics": metrics,
            "probs": probs,
        }
        if best is None or metrics["selection_score"] > best["metrics"]["selection_score"]:
            best = candidate
    if best is None:
        raise ValueError("weight grid produced no candidates")
    return best


def simplex_grid(n_sources: int, units: int):
    """Yield simplex weights with integer grid units summing to one."""

    if n_sources <= 0:
        return
    if n_sources == 1:
        yield np.asarray([1.0], dtype=np.float64)
        return
    for prefix in product(range(units + 1), repeat=n_sources - 1):
        used = sum(prefix)
        if used > units:
            continue
        values = list(prefix) + [units - used]
        yield np.asarray(values, dtype=np.float64) / float(units)
