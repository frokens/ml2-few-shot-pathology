"""Class-bias calibration for one saved prediction file."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative


def calibrate_class_bias(config_path: str | Path) -> dict[str, Any]:
    """Fit and nested-validate additive class log-probability biases."""

    config = load_yaml(config_path)
    oof_path = resolve_project_path(config["oof_prediction"])
    out_dir = resolve_project_path(config.get("out_dir", "artifacts/analysis/class_bias"))
    if oof_path is None or not oof_path.exists():
        raise FileNotFoundError(f"OOF prediction not found: {config['oof_prediction']}")
    if out_dir is None:
        raise ValueError("out_dir cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = np.load(oof_path, allow_pickle=True)
    try:
        probs = np.asarray(payload["probs"], dtype=np.float64)
        y_true = np.asarray(payload["y_true"], dtype=np.int64)
        class_names = [str(value) for value in payload["class_names"].tolist()]
        filenames = np.asarray(payload.get("train_filenames", np.arange(len(y_true))), dtype=object)
    finally:
        payload.close()

    bias_grid = build_bias_grid(config.get("bias_grid", {}), class_names)
    temperature_grid = build_temperature_grid(config.get("temperature_grid", [1.0]))
    base_pred = probs.argmax(axis=1)
    base_metrics = prediction_metrics(y_true, base_pred, class_names)
    nested = nested_bias_predictions(
        probs,
        y_true,
        bias_grid=bias_grid,
        temperature_grid=temperature_grid,
        n_splits=int(config.get("n_splits", 5)),
        seed=int(config.get("seed", 2026)),
    )
    nested_metrics = prediction_metrics(y_true, nested["pred"], class_names)
    fit = fit_bias(probs, y_true, bias_grid=bias_grid, temperature_grid=temperature_grid)
    all_oof_probs = apply_bias_to_probs(probs, fit["bias"], temperature=fit["temperature"])
    all_oof_pred = all_oof_probs.argmax(axis=1)
    all_oof_metrics = prediction_metrics(y_true, all_oof_pred, class_names)

    np.savez_compressed(
        out_dir / "oof_biased_nested.npz",
        probs=nested["probs"],
        y_true=y_true,
        y_pred=nested["pred"],
        class_names=np.asarray(class_names, dtype=object),
        train_filenames=filenames,
        source_oof=np.asarray(project_relative(oof_path), dtype=object),
        prediction_kind=np.asarray("nested_class_bias_oof", dtype=object),
    )
    np.savez_compressed(
        out_dir / "oof_biased_all_oof_fit.npz",
        probs=all_oof_probs,
        y_true=y_true,
        y_pred=all_oof_pred,
        class_names=np.asarray(class_names, dtype=object),
        train_filenames=filenames,
        source_oof=np.asarray(project_relative(oof_path), dtype=object),
        prediction_kind=np.asarray("all_oof_class_bias_diagnostic", dtype=object),
    )
    bias_payload = {
        "class_names": class_names,
        "bias": fit["bias"].tolist(),
        "temperature": fit["temperature"],
        "source_oof": project_relative(oof_path),
        "selection_score": fit["selection_score"],
        "note": "Promotion decisions should use nested metrics; all-OOF bias is for future single-refit test post-processing.",
    }
    (out_dir / "class_bias.json").write_text(json.dumps(bias_payload, indent=2, sort_keys=True) + "\n")
    report = {
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "source_oof": project_relative(oof_path),
        "out_dir": project_relative(out_dir),
        "num_rows": int(len(y_true)),
        "class_names": class_names,
        "grid_size": int(len(bias_grid)),
        "temperature_grid": temperature_grid.tolist(),
        "base": base_metrics,
        "nested": {
            **nested_metrics,
            "fold_biases": [bias.tolist() for bias in nested["fold_biases"]],
            "fold_temperatures": nested["fold_temperatures"],
        },
        "all_oof_fit": {**all_oof_metrics, "bias": fit["bias"].tolist(), "temperature": fit["temperature"]},
        "artifacts": {
            "nested_oof": project_relative(out_dir / "oof_biased_nested.npz"),
            "all_oof_fit_oof": project_relative(out_dir / "oof_biased_all_oof_fit.npz"),
            "bias": project_relative(out_dir / "class_bias.json"),
        },
    }
    (out_dir / "class_bias_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out_dir / "class_bias_report.md").write_text(render_report(report), encoding="utf-8")
    return report


def apply_class_bias(prediction: str | Path, *, bias: str | Path, out: str | Path) -> dict[str, Any]:
    """Apply a saved class bias to one prediction NPZ."""

    pred_path = resolve_project_path(prediction)
    bias_path = resolve_project_path(bias)
    out_path = resolve_project_path(out)
    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"prediction not found: {prediction}")
    if bias_path is None or not bias_path.exists():
        raise FileNotFoundError(f"bias not found: {bias}")
    if out_path is None:
        raise ValueError("out cannot be None")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bias_payload = json.loads(bias_path.read_text(encoding="utf-8"))
    class_names = [str(value) for value in bias_payload["class_names"]]
    bias_values = np.asarray(bias_payload["bias"], dtype=np.float64)
    temperature = float(bias_payload.get("temperature", 1.0))

    payload = np.load(pred_path, allow_pickle=True)
    try:
        pred_classes = [str(value) for value in payload["class_names"].tolist()]
        if pred_classes != class_names:
            raise ValueError("class_names do not match bias file")
        probs = np.asarray(payload["probs"], dtype=np.float64)
        output = {key: payload[key] for key in payload.files if key != "probs"}
    finally:
        payload.close()
    output["probs"] = apply_bias_to_probs(probs, bias_values, temperature=temperature)
    output["class_bias"] = bias_values
    output["class_bias_temperature"] = np.asarray(temperature, dtype=np.float64)
    output["class_bias_source"] = np.asarray(project_relative(bias_path), dtype=object)
    np.savez_compressed(out_path, **output)
    return {"prediction": project_relative(pred_path), "bias": project_relative(bias_path), "out": project_relative(out_path)}


def build_bias_grid(raw_grid: dict[str, Any], class_names: list[str]) -> np.ndarray:
    """Return zero-mean additive bias candidates in class order."""

    if not raw_grid:
        raw_grid = {name: [-0.4, -0.2, 0.0, 0.2, 0.4] for name in class_names}
    values = []
    for name in class_names:
        if name not in raw_grid:
            raise ValueError(f"bias_grid missing class: {name}")
        values.append([float(value) for value in raw_grid[name]])
    grid = np.asarray(list(itertools.product(*values)), dtype=np.float64)
    return grid - grid.mean(axis=1, keepdims=True)


def build_temperature_grid(raw_grid: Any) -> np.ndarray:
    """Return positive temperature candidates for optional log-probability scaling."""

    temperatures = np.asarray([float(value) for value in (raw_grid or [1.0])], dtype=np.float64)
    if temperatures.ndim != 1 or len(temperatures) == 0:
        raise ValueError("temperature_grid must be a non-empty 1D list")
    if np.any(temperatures <= 0):
        raise ValueError("temperature_grid values must be positive")
    return temperatures


def fit_bias(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    bias_grid: np.ndarray,
    temperature_grid: np.ndarray,
) -> dict[str, Any]:
    """Fit the best class bias on a labeled probability matrix."""

    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    best_score = -1.0
    best_bias = bias_grid[0]
    best_temperature = float(temperature_grid[0])
    for temperature in temperature_grid:
        scaled_log_probs = log_probs / float(temperature)
        for bias in bias_grid:
            pred = (scaled_log_probs + bias).argmax(axis=1)
            score = selection_score(y_true, pred)
            if score > best_score:
                best_score = score
                best_bias = bias.copy()
                best_temperature = float(temperature)
    return {"bias": best_bias, "temperature": best_temperature, "selection_score": float(best_score)}


def nested_bias_predictions(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    bias_grid: np.ndarray,
    temperature_grid: np.ndarray,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    """Fit class bias on training folds and apply it to held-out folds."""

    nested_probs = np.zeros_like(probs, dtype=np.float64)
    nested_pred = np.zeros_like(y_true, dtype=np.int64)
    fold_biases = []
    fold_temperatures = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(np.zeros(len(y_true)), y_true):
        fit = fit_bias(
            probs[train_idx],
            y_true[train_idx],
            bias_grid=bias_grid,
            temperature_grid=temperature_grid,
        )
        bias = fit["bias"]
        temperature = fit["temperature"]
        fold_biases.append(bias)
        fold_temperatures.append(temperature)
        nested_probs[val_idx] = apply_bias_to_probs(probs[val_idx], bias, temperature=temperature)
        nested_pred[val_idx] = nested_probs[val_idx].argmax(axis=1)
    return {"probs": nested_probs, "pred": nested_pred, "fold_biases": fold_biases, "fold_temperatures": fold_temperatures}


def apply_bias_to_probs(probs: np.ndarray, bias: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Apply an additive log-probability bias and renormalize."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_probs = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature) + bias.reshape(1, -1)
    log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
    exp = np.exp(log_probs)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def selection_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Project selection score: mean of macro-F1 and balanced accuracy."""

    macro = f1_score(y_true, y_pred, average="macro")
    balanced = balanced_accuracy_score(y_true, y_pred)
    return float((macro + balanced) / 2.0)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Compute aggregate and per-class metrics."""

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro = float(f1_score(y_true, y_pred, average="macro"))
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "macro_f1": macro,
        "balanced_accuracy": balanced,
        "selection_score": float((macro + balanced) / 2.0),
        "pred_counts": np.bincount(y_pred, minlength=len(class_names)).astype(int).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, name in enumerate(class_names)
        },
    }


def render_report(report: dict[str, Any]) -> str:
    """Render a compact Markdown report."""

    lines = [
        "# Class Bias Calibration Report",
        "",
        f"- Source OOF: `{report['source_oof']}`",
        f"- Grid size: `{report['grid_size']}`",
        f"- Temperature grid: `{report.get('temperature_grid', [1.0])}`",
        "",
        "| Candidate | Macro-F1 | Balanced Acc | Selection |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [("base", "base"), ("nested", "nested_bias"), ("all_oof_fit", "all_oof_fit")]:
        row = report[key]
        lines.append(f"| {label} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | {row['selection_score']:.6f} |")
    if "temperature" in report["all_oof_fit"]:
        lines.extend(["", f"- All-OOF fit temperature: `{report['all_oof_fit']['temperature']:.6f}`"])
    if "fold_temperatures" in report["nested"]:
        lines.append(f"- Nested fold temperatures: `{report['nested']['fold_temperatures']}`")
    lines.extend(["", "## Per-Class Nested Bias", "", "| Class | Precision | Recall | F1 |", "|---|---:|---:|---:|"])
    for name, row in report["nested"]["per_class"].items():
        lines.append(f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |")
    lines.append("")
    return "\n".join(lines)
