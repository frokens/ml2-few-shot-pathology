"""Final prediction fusion based on OOF validation."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np

from ml_final.heads.ensemble_search import average_probs, grid_weight_search, weighted_probs
from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.utils.config import resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths


def select_final_blend(
    *,
    scheme01_runs: str | Path,
    scheme02_runs: str | Path | None,
    scheme03_runs: str | Path | None,
    out: str | Path,
    top_k: int = 5,
    weight_step: float = 0.1,
) -> dict[str, Any]:
    """Select final blend weights from aligned OOF predictions."""

    assert_no_forbidden_reference_paths(
        {"scheme01_runs": scheme01_runs, "scheme02_runs": scheme02_runs, "scheme03_runs": scheme03_runs},
        context="final blend selection",
    )
    out_dir = resolve_project_path(out)
    if out_dir is None:
        raise ValueError("out is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = resolve_project_path("artifacts/submissions")
    if submission_dir is None:
        raise ValueError("submission dir cannot be None")
    submission_dir.mkdir(parents=True, exist_ok=True)

    oof_files = collect_oof_files([scheme01_runs, scheme02_runs, scheme03_runs])
    candidates = load_aligned_oof_candidates(oof_files)
    if not candidates:
        raise FileNotFoundError("no aligned OOF prediction files found for final blend")
    candidates = [item for item in candidates if item.get("test_path")]
    if not candidates:
        raise FileNotFoundError("no final-blend OOF candidate has a matching test prediction file")
    candidates = sorted(candidates, key=lambda item: item["metrics"]["selection_score"], reverse=True)
    selected = candidates[: max(1, min(top_k, len(candidates)))]
    y_true = selected[0]["y_true"]
    class_names = selected[0]["class_names"]
    simple_probs = average_probs([item["probs"] for item in selected])
    simple_metrics = compute_classification_metrics(y_true, simple_probs.argmax(axis=1), class_names)
    weighted = (
        grid_weight_search(
            [item["probs"] for item in selected],
            y_true,
            class_names,
            step=weight_step,
            max_sources=top_k,
        )
        if len(selected) > 1
        else {"weights": [1.0], "metrics": selected[0]["metrics"], "probs": selected[0]["probs"]}
    )
    chosen_method = "weighted" if weighted["metrics"]["selection_score"] >= simple_metrics["selection_score"] else "simple"
    chosen_probs = weighted["probs"] if chosen_method == "weighted" else simple_probs
    chosen_weights = weighted["weights"] if chosen_method == "weighted" else [1.0 / len(selected)] * len(selected)
    oof_path = out_dir / "final_blend_oof.npz"
    np.savez_compressed(
        oof_path,
        probs=chosen_probs,
        y_true=y_true,
        y_pred=chosen_probs.argmax(axis=1),
        class_names=np.asarray(class_names, dtype=object),
        source_files=np.asarray([item["oof_path"] for item in selected], dtype=object),
        source_weights=np.asarray(chosen_weights, dtype=np.float64),
    )
    test_output = write_final_test_submission(selected, chosen_weights, submission_dir=submission_dir)
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_count": len(candidates),
        "selected_count": len(selected),
        "selected_oof_files": [item["oof_path"] for item in selected],
        "selected_test_files": [item.get("test_path") for item in selected],
        "weights": [float(item) for item in chosen_weights],
        "chosen_method": chosen_method,
        "simple_metrics": simple_metrics,
        "weighted_metrics": weighted["metrics"],
        "final_metrics": compute_classification_metrics(y_true, chosen_probs.argmax(axis=1), class_names),
        "oof_path": project_relative(oof_path),
        "submission": test_output.get("submission"),
        "candidate_ranking": [compact_candidate(item) for item in candidates],
    }
    write_json(report, out_dir / "final_blend_selection.json")
    write_final_blend_report(out_dir / "final_blend_report.md", report)
    return report


def collect_oof_files(roots: list[str | Path | None]) -> list[Path]:
    """Collect OOF files from provided roots."""

    files: list[Path] = []
    for root in roots:
        if root is None:
            continue
        resolved = resolve_project_path(root)
        if resolved is None or not resolved.exists():
            continue
        if resolved.is_file() and resolved.name.startswith("oof_") and resolved.suffix == ".npz":
            if is_candidate_prediction_file(resolved):
                files.append(resolved)
        elif resolved.is_dir():
            files.extend(path for path in sorted(resolved.rglob("oof_*.npz")) if is_candidate_prediction_file(path))
    return sorted(set(files))


def is_candidate_prediction_file(path: Path) -> bool:
    """Reject smoke/tmp/incomplete prediction files from final blend discovery."""

    text = path.as_posix().lower()
    blocked = ("/smoke", "smoke_", "/tmp", "/temp", "incomplete")
    return not any(marker in text for marker in blocked)


def load_aligned_oof_candidates(files: list[Path]) -> list[dict[str, Any]]:
    """Load candidates whose OOF rows align with the first valid candidate."""

    candidates: list[dict[str, Any]] = []
    ref_y = None
    ref_names = None
    ref_classes = None
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            if "probs" not in data or "y_true" not in data:
                continue
            probs = np.asarray(data["probs"], dtype=np.float64)
            y_true = np.asarray(data["y_true"], dtype=np.int64)
            names = (
                np.asarray(data["train_filenames"]).astype(str)
                if "train_filenames" in data
                else np.asarray(np.arange(len(y_true))).astype(str)
            )
            class_names = [str(item) for item in data["class_names"].tolist()]
        if ref_y is None:
            ref_y = y_true
            ref_names = names
            ref_classes = class_names
        if not np.array_equal(y_true, ref_y):
            continue
        if ref_names is not None and names.shape == ref_names.shape and not np.array_equal(names, ref_names):
            continue
        if class_names != ref_classes:
            continue
        metrics = compute_classification_metrics(y_true, probs.argmax(axis=1), class_names)
        candidates.append(
            {
                "oof_path": project_relative(path),
                "test_path": infer_test_path(path),
                "probs": probs,
                "y_true": y_true,
                "class_names": class_names,
                "metrics": metrics,
            }
        )
    return candidates


def infer_test_path(oof_path: Path) -> str | None:
    """Infer matching test prediction path."""

    candidate = oof_path.with_name(oof_path.name.replace("oof_", "test_", 1))
    return project_relative(candidate) if candidate.exists() else None


def write_final_test_submission(
    selected: list[dict[str, Any]],
    weights: list[float],
    *,
    submission_dir: Path,
) -> dict[str, Any]:
    """Write final submission if all selected sources have test predictions."""

    test_paths = [resolve_project_path(item.get("test_path")) for item in selected]
    if any(path is None or not path.exists() for path in test_paths):
        return {"submission": None, "reason": "missing test prediction files"}
    loaded = [np.load(path, allow_pickle=True) for path in test_paths if path is not None]
    try:
        reference_classes = [str(item) for item in loaded[0]["class_names"].tolist()]
        reference_filenames = np.asarray(loaded[0]["test_filenames"]).astype(str)
        for item, path in zip(loaded[1:], test_paths[1:]):
            current_classes = [str(value) for value in item["class_names"].tolist()]
            current_filenames = np.asarray(item["test_filenames"]).astype(str)
            if current_classes != reference_classes:
                raise ValueError(f"final blend test class_names do not align: {path}")
            if not np.array_equal(current_filenames, reference_filenames):
                raise ValueError(f"final blend test filenames do not align: {path}")
        probs = weighted_probs([item["probs"] for item in loaded], weights)
        class_names = reference_classes
        filenames = reference_filenames.tolist()
        labels = [class_names[idx] for idx in probs.argmax(axis=1)]
        submission_path = submission_dir / "submission_final_blend.csv"
        with submission_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "label"])
            writer.writerows(zip(filenames, labels))
        np.savez_compressed(
            submission_dir / "submission_final_blend_probs.npz",
            probs=probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=np.asarray(filenames, dtype=object),
            source_files=np.asarray([project_relative(path) for path in test_paths if path is not None], dtype=object),
            source_weights=np.asarray(weights, dtype=np.float64),
        )
        return {"submission": project_relative(submission_path)}
    finally:
        for item in loaded:
            item.close()


def compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    """Compact candidate for JSON reporting."""

    metrics = item["metrics"]
    return {
        "oof_path": item["oof_path"],
        "test_path": item.get("test_path"),
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "selection_score": metrics["selection_score"],
    }


def write_final_blend_report(path: Path, report: dict[str, Any]) -> None:
    """Write a markdown report for final blend selection."""

    lines = [
        "# Final Blend Selection",
        "",
        f"- Chosen method: `{report['chosen_method']}`",
        f"- Selected sources: `{report['selected_count']}`",
        f"- Submission: `{report['submission']}`",
        f"- Final macro-F1: `{report['final_metrics']['macro_f1']:.6f}`",
        f"- Final balanced accuracy: `{report['final_metrics']['balanced_accuracy']:.6f}`",
        "",
        "## Selected Sources",
        "",
    ]
    for path_value, weight in zip(report["selected_oof_files"], report["weights"]):
        lines.append(f"- `{path_value}` weight={weight:.4f}")
    lines.extend(["", "## Candidate Ranking", ""])
    for idx, row in enumerate(report["candidate_ranking"][:20], start=1):
        lines.append(
            f"- {idx}. `{row['oof_path']}` selection={row['selection_score']:.6f}, "
            f"macro_f1={row['macro_f1']:.6f}, balanced={row['balanced_accuracy']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
