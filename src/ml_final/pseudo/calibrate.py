"""Teacher ensemble construction and OOF temperature scaling."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from ml_final.metrics.classification import compute_classification_metrics
from ml_final.pseudo.common import (
    DEFAULT_CLASS_NAMES,
    apply_temperature,
    discover_npz_files,
    expected_calibration_error,
    nll_loss,
    project_rel_list,
    read_csv_rows,
    top2_stats,
    write_csv_rows,
    write_json,
)
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


def build_teacher(
    config_path: str | Path,
    *,
    out: str | Path,
) -> dict[str, Any]:
    """Build a teacher ensemble from OOF/test prediction files."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme03 teacher build")
    set_seed(int(config.get("seed", 2026)))
    out_dir = resolve_project_path(out)
    if out_dir is None:
        raise ValueError("out cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)

    source_files = read_teacher_source_files(config.get("teacher_sources_file"))
    source_files.extend(read_teacher_source_files(config.get("optional_teacher_sources_file")))
    source_oof_files = [path for path in source_files if path.name.startswith("oof_")]
    source_test_files = [path for path in source_files if path.name.startswith("test_")]
    oof_files = discover_npz_files(
        explicit_files=config.get("oof_prediction_files"),
        search_dir=config.get("oof_prediction_search_dir"),
        pattern=str(config.get("oof_pattern", "oof_*.npz")),
        allow_search=bool(config.get("allow_prediction_search", False)),
    )
    oof_files = sorted(set(source_oof_files + oof_files))
    test_files = discover_npz_files(
        explicit_files=config.get("test_prediction_files"),
        search_dir=config.get("test_prediction_search_dir"),
        pattern=str(config.get("test_pattern", "test_*.npz")),
        allow_search=bool(config.get("allow_prediction_search", False)),
    )
    test_files = sorted(set(source_test_files + test_files))
    if not oof_files and config.get("synthetic_smoke", {}).get("enabled", False):
        synthetic = make_synthetic_predictions(config)
        oof_files = [synthetic["oof_path"]]
        test_files = [synthetic["test_path"]]
    if not oof_files:
        raise FileNotFoundError("no OOF prediction files found for teacher build")

    oof = load_oof_ensemble(oof_files)
    raw_oof_probs = oof["probs"]
    y_true = oof["y_true"]
    class_names = oof["class_names"]
    temperature_grid = [float(item) for item in config.get("temperature_grid", [1.0])]
    if not temperature_grid:
        raise ValueError("temperature_grid cannot be empty")
    temperatures = []
    for temp in temperature_grid:
        scaled = apply_temperature(raw_oof_probs, temp)
        temperatures.append(
            {
                "temperature": temp,
                "nll": nll_loss(scaled, y_true),
                "ece": expected_calibration_error(scaled, y_true),
            }
        )
    best = min(temperatures, key=lambda row: row["nll"])
    best_temperature = float(best["temperature"])
    oof_probs = apply_temperature(raw_oof_probs, best_temperature)
    y_pred = oof_probs.argmax(axis=1)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["nll"] = nll_loss(oof_probs, y_true)
    metrics["ece"] = expected_calibration_error(oof_probs, y_true)

    oof_csv = out_dir / "teacher_oof_predictions.csv"
    write_prediction_csv(
        oof_csv,
        filenames=oof["filenames"],
        probs=oof_probs,
        class_names=class_names,
        y_true=y_true,
        source_agreement=oof["agreement"],
    )

    test_csv = None
    test_rows = 0
    if test_files:
        test = load_test_ensemble(test_files, class_names=class_names)
        test_probs = apply_temperature(test["probs"], best_temperature)
        test_csv = out_dir / "teacher_test_predictions.csv"
        write_prediction_csv(
            test_csv,
            filenames=test["filenames"],
            probs=test_probs,
            class_names=class_names,
            y_true=None,
            source_agreement=test["agreement"],
        )
        test_rows = len(test["filenames"])

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "oof_prediction_files": project_rel_list(oof_files),
        "test_prediction_files": project_rel_list(test_files),
        "class_names": class_names,
        "temperature": best_temperature,
        "temperature_grid": temperatures,
        "metrics": metrics,
        "teacher_oof_predictions": project_relative(oof_csv),
        "teacher_test_predictions": project_relative(test_csv) if test_csv else None,
        "num_oof_rows": int(len(y_true)),
        "num_test_rows": int(test_rows),
    }
    write_json(out_dir / "teacher_manifest.json", manifest)
    write_calibration_report(out_dir / "calibration_report.md", manifest)
    return manifest


def load_oof_ensemble(files: list[Path]) -> dict[str, Any]:
    """Load and average aligned OOF prediction files."""

    loaded = [np.load(path, allow_pickle=True) for path in files]
    try:
        ref = loaded[0]
        y_true = np.asarray(ref["y_true"], dtype=np.int64)
        filenames = np.asarray(ref["train_filenames"]).astype(str)
        class_names = [str(item) for item in ref["class_names"].tolist()]
        probs = []
        votes = []
        for item, path in zip(loaded, files):
            current_y = np.asarray(item["y_true"], dtype=np.int64)
            current_names = np.asarray(item["train_filenames"]).astype(str)
            current_classes = [str(cls) for cls in item["class_names"].tolist()]
            if not np.array_equal(current_y, y_true):
                raise ValueError(f"OOF labels do not align: {path}")
            if not np.array_equal(current_names, filenames):
                raise ValueError(f"OOF filenames do not align: {path}")
            if current_classes != class_names:
                raise ValueError(f"OOF class_names do not align: {path}")
            current_probs = normalize_probs(np.asarray(item["probs"], dtype=np.float64))
            probs.append(current_probs)
            votes.append(current_probs.argmax(axis=1))
        stacked = np.stack(probs, axis=0)
        averaged = stacked.mean(axis=0)
        agreement = compute_agreement(np.stack(votes, axis=0), averaged.argmax(axis=1))
        return {
            "probs": averaged,
            "y_true": y_true,
            "filenames": filenames,
            "class_names": class_names,
            "agreement": agreement,
        }
    finally:
        for item in loaded:
            item.close()


def read_teacher_source_files(path_value: str | Path | None) -> list[Path]:
    """Read OOF/test npz source paths from a selection text file."""

    if not path_value:
        return []
    path = resolve_project_path(path_value)
    if path is None or not path.exists():
        return []
    files: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        resolved = resolve_project_path(value)
        if resolved is not None and resolved.exists():
            files.append(resolved)
    return files


def load_test_ensemble(files: list[Path], *, class_names: list[str]) -> dict[str, Any]:
    """Load and average aligned test prediction files."""

    loaded = [np.load(path, allow_pickle=True) for path in files]
    try:
        ref = loaded[0]
        filenames = np.asarray(ref["test_filenames"]).astype(str)
        probs = []
        votes = []
        for item, path in zip(loaded, files):
            current_names = np.asarray(item["test_filenames"]).astype(str)
            current_classes = [str(cls) for cls in item["class_names"].tolist()]
            if not np.array_equal(current_names, filenames):
                raise ValueError(f"test filenames do not align: {path}")
            if current_classes != class_names:
                raise ValueError(f"test class_names do not align: {path}")
            current_probs = normalize_probs(np.asarray(item["probs"], dtype=np.float64))
            probs.append(current_probs)
            votes.append(current_probs.argmax(axis=1))
        stacked = np.stack(probs, axis=0)
        averaged = stacked.mean(axis=0)
        agreement = compute_agreement(np.stack(votes, axis=0), averaged.argmax(axis=1))
        return {"probs": averaged, "filenames": filenames, "agreement": agreement}
    finally:
        for item in loaded:
            item.close()


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Normalize probabilities defensively."""

    probs = np.clip(probs, 1e-12, 1.0)
    return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)


def compute_agreement(votes: np.ndarray, ensemble_pred: np.ndarray) -> np.ndarray:
    """Fraction of source models agreeing with the ensemble top-1 class."""

    return (votes == ensemble_pred[None, :]).mean(axis=0)


def write_prediction_csv(
    path: Path,
    *,
    filenames: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    y_true: np.ndarray | None,
    source_agreement: np.ndarray,
) -> None:
    """Write teacher predictions to a CSV consumable by selector code."""

    stats = top2_stats(probs)
    rows: list[dict[str, Any]] = []
    for idx, filename in enumerate(filenames):
        row: dict[str, Any] = {
            "filename": str(filename),
            "pred_label": class_names[int(stats["top1"][idx])],
            "prob_top1": f"{stats['prob_top1'][idx]:.10f}",
            "prob_top2": f"{stats['prob_top2'][idx]:.10f}",
            "margin": f"{stats['margin'][idx]:.10f}",
            "entropy": f"{stats['entropy'][idx]:.10f}",
            "teacher_agreement": f"{source_agreement[idx]:.10f}",
        }
        if y_true is not None:
            row["true_label"] = class_names[int(y_true[idx])]
            row["y_true"] = int(y_true[idx])
            row["correct"] = int(int(stats["top1"][idx]) == int(y_true[idx]))
        for cls_idx, _class_name in enumerate(class_names):
            row[f"prob_{cls_idx}"] = f"{probs[idx, cls_idx]:.10f}"
        rows.append(row)

    fieldnames = [
        "filename",
        "true_label",
        "y_true",
        "pred_label",
        "correct",
        "prob_top1",
        "prob_top2",
        "margin",
        "entropy",
        "teacher_agreement",
    ] + [f"prob_{idx}" for idx in range(len(class_names))]
    write_csv_rows(path, rows, fieldnames)


def write_calibration_report(path: Path, manifest: dict[str, Any]) -> None:
    """Write a concise calibration report."""

    lines = [
        "# Scheme 03 Teacher Calibration Report",
        "",
        "## Inputs",
        "",
        f"- OOF prediction files: {len(manifest['oof_prediction_files'])}",
        f"- Test prediction files: {len(manifest['test_prediction_files'])}",
        f"- Classes: {', '.join(manifest['class_names'])}",
        "",
        "## Selected Temperature",
        "",
        f"- Temperature: `{manifest['temperature']}`",
        "",
        "## OOF Metrics",
        "",
        f"- macro-F1: `{manifest['metrics']['macro_f1']:.6f}`",
        f"- balanced accuracy: `{manifest['metrics']['balanced_accuracy']:.6f}`",
        f"- NLL: `{manifest['metrics']['nll']:.6f}`",
        f"- ECE: `{manifest['metrics']['ece']:.6f}`",
        "",
        "## Temperature Grid",
        "",
    ]
    for row in manifest["temperature_grid"]:
        lines.append(
            f"- T={row['temperature']}: nll={row['nll']:.6f}, ece={row['ece']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_synthetic_predictions(config: dict[str, Any]) -> dict[str, Path]:
    """Create deterministic synthetic predictions for smoke tests only."""

    smoke = config.get("synthetic_smoke", {})
    out_dir = resolve_project_path(smoke.get("dir", "artifacts/smoke/scheme_03_teacher_inputs"))
    if out_dir is None:
        raise ValueError("synthetic smoke dir cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    rng = np.random.default_rng(seed)
    class_names = config.get("class_names", DEFAULT_CLASS_NAMES)
    n_classes = len(class_names)
    n_train = int(smoke.get("num_train", max(25, n_classes * 5)))
    n_test = int(smoke.get("num_test", 15))
    y_true = np.arange(n_train, dtype=np.int64) % n_classes
    logits = rng.normal(0, 0.35, size=(n_train, n_classes))
    logits[np.arange(n_train), y_true] += 2.5
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    train_filenames = np.asarray([f"train_smoke_{idx:05d}.png" for idx in range(n_train)], dtype=object)
    test_logits = rng.normal(0, 0.6, size=(n_test, n_classes))
    test_logits[np.arange(n_test), np.arange(n_test) % n_classes] += 2.3
    test_probs = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
    test_probs = test_probs / test_probs.sum(axis=1, keepdims=True)
    test_filenames = np.asarray([f"test_smoke_{idx:05d}.png" for idx in range(n_test)], dtype=object)
    oof_path = out_dir / "oof_synthetic_teacher.npz"
    test_path = out_dir / "test_synthetic_teacher.npz"
    np.savez_compressed(
        oof_path,
        probs=probs,
        y_true=y_true,
        y_pred=probs.argmax(axis=1),
        class_names=np.asarray(class_names, dtype=object),
        train_filenames=train_filenames,
    )
    np.savez_compressed(
        test_path,
        probs=test_probs,
        class_names=np.asarray(class_names, dtype=object),
        test_filenames=test_filenames,
    )
    return {"oof_path": oof_path, "test_path": test_path}
