"""Tests for offline dataset audit and manifest generation."""

import csv
import json
import tempfile
from pathlib import Path

from PIL import Image

from ml_final.data.audit import DEFAULT_CLASSES, audit_dataset, scan_train_images


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _make_toy_train(root: Path) -> Path:
    train_dir = root / "train_few_shot"
    for class_idx, label in enumerate(DEFAULT_CLASSES):
        for img_idx in range(2):
            _write_png(
                train_dir / label / f"{label}_{img_idx + 1:03d}.png",
                color=(class_idx * 20, img_idx * 30, 100),
            )
    return train_dir


def test_scan_train_images_schema_and_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = _make_toy_train(Path(tmp))
        rows = scan_train_images(train_dir)

    assert len(rows) == 10
    assert {row["label"] for row in rows} == set(DEFAULT_CLASSES)
    assert rows[0]["split"] == "train"
    assert rows[0]["width"] == 32
    assert rows[0]["height"] == 32
    assert rows[0]["mode"] == "RGB"
    assert len(rows[0]["sha256"]) == 64


def test_audit_dataset_writes_expected_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_dir = _make_toy_train(root)
        test_dir = root / "test"
        _write_png(test_dir / "test_00001.png", color=(1, 2, 3))

        out_dir = root / "artifacts" / "dataset_audit"
        outputs = audit_dataset(train_dir=train_dir, test_dir=test_dir, out_dir=out_dir)

        assert outputs.train_manifest.exists()
        assert outputs.test_manifest is not None
        assert outputs.test_manifest.exists()
        assert outputs.summary_json.exists()
        assert outputs.image_stats_json.exists()
        assert outputs.label_map_json.exists()
        assert outputs.manifest_hashes_json.exists()

        with outputs.train_manifest.open(newline="", encoding="utf-8") as handle:
            train_rows = list(csv.DictReader(handle))
        assert len(train_rows) == 10

        summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
        assert summary["train_image_count"] == 10
        assert summary["test_image_count"] == 1
        assert summary["class_counts"] == {label: 2 for label in DEFAULT_CLASSES}


def test_audit_dataset_allows_missing_test_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_dir = _make_toy_train(root)
        outputs = audit_dataset(
            train_dir=train_dir,
            test_dir=root / "missing_test",
            out_dir=root / "audit",
        )

        assert outputs.train_manifest.exists()
        assert outputs.test_manifest is None
        summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
        assert summary["has_test_manifest"] is False
