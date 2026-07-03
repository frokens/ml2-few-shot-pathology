"""Tests for sample-level OOF error audits."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ml_final.analysis.oof_audit import audit_oof_errors, compare_oof_audits


def test_oof_audit_writes_hard_class_report_and_montage(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    class_names = [f"Class_{idx}" for idx in range(5)]
    filenames = []
    y_true = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    probs = np.asarray(
        [
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.05, 0.80, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.20, 0.65, 0.05],
            [0.05, 0.05, 0.70, 0.10, 0.10],
            [0.05, 0.60, 0.20, 0.05, 0.10],
        ],
        dtype=np.float64,
    )
    for idx, label in enumerate(y_true):
        filename = f"Class_{label}_{idx}.png"
        path = image_dir / filename
        Image.new("RGB", (16, 16), color=(idx * 30, 20, 100)).save(path)
        filenames.append(filename)
        rows.append(
            {
                "filename": filename,
                "rel_path": filename,
                "abs_path": str(path),
                "label": class_names[int(label)],
            }
        )
    oof_path = tmp_path / "oof_model.npz"
    np.savez_compressed(
        oof_path,
        probs=probs,
        y_true=y_true,
        class_names=np.asarray(class_names, dtype=object),
        train_filenames=np.asarray(filenames, dtype=object),
    )
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "rel_path", "abs_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    result = audit_oof_errors(
        oof_path=oof_path,
        manifest_path=manifest_path,
        out_dir=tmp_path / "audit",
        top_k=3,
    )

    assert result["num_hard_errors"] == 3
    assert (tmp_path / "audit" / "hard_class_oof_audit.md").exists()
    assert (tmp_path / "audit" / "hard_class_error_rows.csv").exists()
    assert (tmp_path / "audit" / "hard_class_error_montage.png").exists()
    summary = json.loads((tmp_path / "audit" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["confusion_matrix"][2][3] == 1


def test_compare_oof_audits_finds_persistent_errors(tmp_path: Path):
    audit_a = tmp_path / "audit_a"
    audit_b = tmp_path / "audit_b"
    image_dir = tmp_path / "images"
    audit_a.mkdir()
    audit_b.mkdir()
    image_dir.mkdir()
    image_path = image_dir / "a.png"
    Image.new("RGB", (16, 16), color=(100, 20, 150)).save(image_path)
    fieldnames = [
        "row_index",
        "filename",
        "rel_path",
        "abs_path",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
        "margin",
        "true_probability",
        "hard_related",
    ]
    rows_a = [
        make_audit_row("a.png", "Class_2", "Class_3", False, 0.9, 0.8, 0.01, abs_path=str(image_path)),
        make_audit_row("b.png", "Class_3", "Class_3", True, 0.8, 0.5, 0.8),
    ]
    rows_b = [
        make_audit_row("a.png", "Class_2", "Class_4", False, 0.95, 0.9, 0.02, abs_path=str(image_path)),
        make_audit_row("b.png", "Class_3", "Class_2", False, 0.7, 0.4, 0.3),
    ]
    for path, rows in [(audit_a / "oof_sample_rows.csv", rows_a), (audit_b / "oof_sample_rows.csv", rows_b)]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    result = compare_oof_audits(audit_dirs=[audit_a, audit_b], out_dir=tmp_path / "compare")

    assert result["num_persistent_errors"] == 1
    report = (tmp_path / "compare" / "persistent_hard_errors.md").read_text(encoding="utf-8")
    assert "a.png" in report
    assert "b.png" not in report
    assert (tmp_path / "compare" / "persistent_pair_summary.csv").exists()
    assert (tmp_path / "compare" / "persistent_pair_review.md").exists()
    assert list((tmp_path / "compare" / "pair_montages").glob("*.png"))


def make_audit_row(
    filename: str,
    true_label: str,
    pred_label: str,
    correct: bool,
    confidence: float,
    margin: float,
    true_probability: float,
    abs_path: str | None = None,
) -> dict[str, object]:
    return {
        "row_index": 0,
        "filename": filename,
        "rel_path": filename,
        "abs_path": abs_path or filename,
        "true_label": true_label,
        "pred_label": pred_label,
        "correct": correct,
        "confidence": confidence,
        "margin": margin,
        "true_probability": true_probability,
        "hard_related": True,
    }
