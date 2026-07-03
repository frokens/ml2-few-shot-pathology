"""Submission generation and validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths


def infer_final(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a submission from saved test prediction files."""
    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="final inference")
    run_name = run_name or config.get("run_name", "scheme01_seed_frozen")
    prediction_files = [
        resolve_project_path(path) for path in config.get("test_prediction_files", [])
    ]
    prediction_files.extend(read_prediction_source_files(config.get("test_prediction_sources_file")))
    prediction_files = [path for path in prediction_files if path is not None and path.exists()]
    if not prediction_files:
        search_dir = resolve_project_path(config.get("prediction_search_dir"))
        if search_dir and search_dir.exists():
            if not bool(config.get("allow_prediction_search", False)):
                raise FileNotFoundError(
                    "final inference requires explicit test_prediction_files or "
                    "test_prediction_sources_file; set allow_prediction_search=true only for debug"
                )
            prediction_files = [path for path in sorted(search_dir.rglob("test_*.npz")) if is_debug_search_allowed(path)]
    if not prediction_files:
        raise FileNotFoundError("no test prediction files found for final inference")

    loaded = [np.load(path, allow_pickle=True) for path in prediction_files]
    try:
        reference_classes = [str(x) for x in loaded[0]["class_names"].tolist()]
        reference_filenames = np.asarray(loaded[0]["test_filenames"]).astype(str)
        for item, path in zip(loaded[1:], prediction_files[1:]):
            current_classes = [str(x) for x in item["class_names"].tolist()]
            current_filenames = np.asarray(item["test_filenames"]).astype(str)
            if current_classes != reference_classes:
                raise ValueError(f"test prediction class_names do not align: {path}")
            if not np.array_equal(current_filenames, reference_filenames):
                raise ValueError(f"test prediction filenames do not align: {path}")
        probs = np.mean([item["probs"] for item in loaded], axis=0)
        class_names = reference_classes
        filenames = reference_filenames.tolist()
        labels = [class_names[idx] for idx in probs.argmax(axis=1)]
    finally:
        for item in loaded:
            item.close()

    out_dir = resolve_project_path(out or config.get("submission_dir", "artifacts/submissions/scheme_01_seed_frozen"))
    if out_dir is None:
        raise ValueError("submission_dir cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_path = out_dir / "submission.csv"
    with submission_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        writer.writerows(zip(filenames, labels))
    metadata = {
        "run_name": run_name,
        "prediction_files": [project_relative(path) for path in prediction_files],
        "num_rows": len(filenames),
        "class_names": class_names,
    }
    metadata_path = out_dir / "submission_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "submission": project_relative(submission_path),
        "metadata": project_relative(metadata_path),
        **metadata,
    }


def read_prediction_source_files(path_value: str | Path | None) -> list[Path]:
    """Read test prediction paths from a selection/fusion source file."""

    if not path_value:
        return []
    path = resolve_project_path(path_value)
    if path is None or not path.exists():
        return []
    files: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or not Path(value).name.startswith("test_"):
            continue
        resolved = resolve_project_path(value)
        if resolved is not None and resolved.exists():
            files.append(resolved)
    return files


def is_debug_search_allowed(path: Path) -> bool:
    """Filter obvious non-final prediction files when debug search is enabled."""

    text = path.as_posix().lower()
    blocked = ("/smoke", "smoke_", "/tmp", "/temp", "incomplete")
    return not any(marker in text for marker in blocked)


def validate_submission(
    submission: str | Path,
    *,
    test_manifest: str | Path,
    test_prediction: str | Path | None = None,
    compare_submission: str | Path | None = None,
    expected_rows: int | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Validate submission CSV schema, labels, row alignment, and optional prediction diagnostics."""
    submission_path = resolve_project_path(submission)
    manifest_path = resolve_project_path(test_manifest)
    if submission_path is None or manifest_path is None:
        raise ValueError("submission and test_manifest are required")
    if not submission_path.exists():
        raise FileNotFoundError(f"submission not found: {submission_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"test manifest not found: {manifest_path}")
    class_names = class_names or [f"Class_{idx}" for idx in range(5)]

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        expected = [row["filename"] for row in csv.DictReader(handle)]
    with submission_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "label"]:
            raise ValueError(f"submission header must be filename,label; got {reader.fieldnames}")
        rows = list(reader)
    labels = [row["label"] for row in rows]
    filenames = [row["filename"] for row in rows]
    bad_labels = sorted(set(labels) - set(class_names))
    if bad_labels:
        raise ValueError(f"invalid labels in submission: {bad_labels}")
    if filenames != expected:
        raise ValueError("submission filenames do not exactly match test manifest order")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"submission row count mismatch: expected {expected_rows}, got {len(rows)}")
    report: dict[str, Any] = {
        "valid": True,
        "rows": len(rows),
        "submission": project_relative(submission_path),
        "class_distribution": {name: int(labels.count(name)) for name in class_names},
    }
    if test_prediction:
        report["prediction"] = validate_submission_prediction(
            test_prediction,
            filenames=filenames,
            labels=labels,
            class_names=class_names,
        )
    if compare_submission:
        report["comparison"] = compare_submission_labels(
            compare_submission,
            filenames=filenames,
            labels=labels,
        )
    return report


def validate_submission_prediction(
    test_prediction: str | Path,
    *,
    filenames: list[str],
    labels: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    """Validate optional test prediction NPZ and summarize confidence."""

    prediction_path = resolve_project_path(test_prediction)
    if prediction_path is None or not prediction_path.exists():
        raise FileNotFoundError(f"test prediction not found: {test_prediction}")
    payload = np.load(prediction_path, allow_pickle=True)
    try:
        probs = np.asarray(payload["probs"], dtype=np.float64)
        pred_filenames = np.asarray(payload["test_filenames"]).astype(str).tolist()
        pred_classes = [str(value) for value in payload["class_names"].tolist()]
        if pred_classes != class_names:
            raise ValueError(f"test prediction class_names do not match submission classes: {prediction_path}")
        if pred_filenames != filenames:
            raise ValueError(f"test prediction filenames do not match submission order: {prediction_path}")
        pred_labels = [class_names[idx] for idx in probs.argmax(axis=1)]
        if pred_labels != labels:
            raise ValueError("submission labels do not match test prediction argmax labels")
        confidence = probs.max(axis=1)
        result: dict[str, Any] = {
            "path": project_relative(prediction_path),
            "rows": int(probs.shape[0]),
            "top1_confidence": {
                "min": float(confidence.min()) if confidence.size else None,
                "mean": float(confidence.mean()) if confidence.size else None,
                "median": float(np.median(confidence)) if confidence.size else None,
                "p05": float(np.quantile(confidence, 0.05)) if confidence.size else None,
                "p95": float(np.quantile(confidence, 0.95)) if confidence.size else None,
                "max": float(confidence.max()) if confidence.size else None,
            },
        }
        if "prediction_kind" in payload.files:
            result["prediction_kind"] = str(payload["prediction_kind"])
        if "model_count" in payload.files:
            result["model_count"] = int(payload["model_count"])
        return result
    finally:
        payload.close()


def compare_submission_labels(
    compare_submission: str | Path,
    *,
    filenames: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Compare labels against a backup/reference submission with matching filenames."""

    compare_path = resolve_project_path(compare_submission)
    if compare_path is None or not compare_path.exists():
        raise FileNotFoundError(f"compare submission not found: {compare_submission}")
    with compare_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "label"]:
            raise ValueError(f"compare submission header must be filename,label; got {reader.fieldnames}")
        compare_rows = list(reader)
    compare_filenames = [row["filename"] for row in compare_rows]
    if compare_filenames != filenames:
        raise ValueError("compare submission filenames do not exactly match target submission order")
    compare_labels = [row["label"] for row in compare_rows]
    changed = sum(int(left != right) for left, right in zip(compare_labels, labels))
    return {"path": project_relative(compare_path), "changed_labels": int(changed)}
