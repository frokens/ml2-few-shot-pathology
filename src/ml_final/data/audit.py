"""Offline dataset audit and manifest generation.

This module intentionally avoids model imports and network access. It only
reads image metadata, computes file hashes, and writes deterministic manifests
for later feature extraction, validation, and submission checks.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from ml_final.utils.paths import project_relative

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_CLASSES = tuple(f"Class_{idx}" for idx in range(5))


@dataclass(frozen=True)
class AuditOutputs:
    """Paths written by a dataset audit run."""

    train_manifest: Path
    test_manifest: Path | None
    summary_json: Path
    image_stats_json: Path
    label_map_json: Path
    manifest_hashes_json: Path


def audit_dataset(
    train_dir: str | Path,
    test_dir: str | Path | None,
    out_dir: str | Path,
    expected_classes: tuple[str, ...] = DEFAULT_CLASSES,
    strict_train: bool = True,
) -> AuditOutputs:
    """Audit train/test image folders and write manifest artifacts.

    Args:
        train_dir: Directory containing class subdirectories.
        test_dir: Optional directory containing unlabeled test images.
        out_dir: Directory for audit artifacts.
        expected_classes: Expected class names and ordering.
        strict_train: If true, reject missing/extra classes or invalid train images.

    Returns:
        Paths to all generated artifacts.
    """
    train_root = Path(train_dir).expanduser().resolve()
    test_root = Path(test_dir).expanduser().resolve() if test_dir else None
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = scan_train_images(train_root, expected_classes=expected_classes)
    test_rows = (
        scan_test_images(test_root)
        if test_root is not None and test_root.exists()
        else []
    )

    _validate_train_rows(
        train_root=train_root,
        train_rows=train_rows,
        expected_classes=expected_classes,
        strict=strict_train,
    )

    train_manifest = output_root / "train_manifest.csv"
    test_manifest = output_root / "test_manifest.csv" if test_rows else None
    summary_json = output_root / "summary.json"
    image_stats_json = output_root / "image_stats.json"
    label_map_json = output_root / "label_map.json"
    manifest_hashes_json = output_root / "dataset_manifest_hashes.json"

    write_manifest(train_rows, train_manifest)
    if test_manifest is not None:
        write_manifest(test_rows, test_manifest)

    label_map = {label: idx for idx, label in enumerate(expected_classes)}
    label_map_json.write_text(
        json.dumps(label_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stats = build_image_stats(train_rows=train_rows, test_rows=test_rows)
    image_stats_json.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = build_summary(
        train_root=train_root,
        test_root=test_root,
        output_root=output_root,
        train_rows=train_rows,
        test_rows=test_rows,
        expected_classes=expected_classes,
        stats=stats,
    )
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hash_targets = {
        "train_manifest": train_manifest,
        "summary_json": summary_json,
        "image_stats_json": image_stats_json,
        "label_map_json": label_map_json,
    }
    if test_manifest is not None:
        hash_targets["test_manifest"] = test_manifest

    manifest_hashes = {
        key: {
            "path": project_relative(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in sorted(hash_targets.items())
    }
    manifest_hashes_json.write_text(
        json.dumps(manifest_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return AuditOutputs(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        summary_json=summary_json,
        image_stats_json=image_stats_json,
        label_map_json=label_map_json,
        manifest_hashes_json=manifest_hashes_json,
    )


def scan_train_images(
    train_root: Path,
    expected_classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> list[dict[str, Any]]:
    """Return deterministic manifest rows for a labeled training directory."""
    if not train_root.exists():
        raise FileNotFoundError(f"train directory not found: {train_root}")
    if not train_root.is_dir():
        raise NotADirectoryError(f"train path is not a directory: {train_root}")

    label_to_idx = {label: idx for idx, label in enumerate(expected_classes)}
    rows: list[dict[str, Any]] = []
    for class_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        label = class_dir.name
        class_index = label_to_idx.get(label, "")
        for image_path in sorted_image_files(class_dir):
            rows.append(
                build_image_row(
                    path=image_path,
                    root=train_root,
                    split="train",
                    label=label,
                    class_index=class_index,
                )
            )
    return rows


def scan_test_images(test_root: Path) -> list[dict[str, Any]]:
    """Return deterministic manifest rows for an unlabeled test directory."""
    if not test_root.exists():
        raise FileNotFoundError(f"test directory not found: {test_root}")
    if not test_root.is_dir():
        raise NotADirectoryError(f"test path is not a directory: {test_root}")

    return [
        build_image_row(
            path=image_path,
            root=test_root,
            split="test",
            label="",
            class_index="",
        )
        for image_path in sorted_image_files(test_root)
    ]


def sorted_image_files(root: Path) -> list[Path]:
    """List image files recursively with stable POSIX-style ordering."""
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def build_image_row(
    path: Path,
    root: Path,
    split: str,
    label: str,
    class_index: int | str,
) -> dict[str, Any]:
    """Build a manifest row for one image path."""
    rel_path = path.relative_to(root).as_posix()
    width: int | str = ""
    height: int | str = ""
    mode = ""
    channels: int | str = ""
    valid = True
    error = ""

    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            channels = len(image.getbands())
            image.verify()
    except Exception as exc:  # pragma: no cover - exact PIL errors vary
        valid = False
        error = f"{type(exc).__name__}: {exc}"

    return {
        "split": split,
        "filename": path.name,
        "rel_path": rel_path,
        "abs_path": project_relative(path.resolve()),
        "label": label,
        "class_index": class_index,
        "width": width,
        "height": height,
        "mode": mode,
        "channels": channels,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "is_valid": valid,
        "error": error,
    }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    """Write manifest rows to CSV with a fixed schema."""
    fieldnames = [
        "split",
        "filename",
        "rel_path",
        "abs_path",
        "label",
        "class_index",
        "width",
        "height",
        "mode",
        "channels",
        "file_size_bytes",
        "sha256",
        "is_valid",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_image_stats(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate image and duplicate statistics from manifest rows."""
    rows = train_rows + test_rows
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        duplicate_groups[str(row["sha256"])].append(str(row["rel_path"]))

    duplicates = {
        digest: sorted(paths)
        for digest, paths in sorted(duplicate_groups.items())
        if len(paths) > 1
    }

    return {
        "total_images": len(rows),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "valid_images": sum(1 for row in rows if row["is_valid"]),
        "invalid_images": sum(1 for row in rows if not row["is_valid"]),
        "width_counts": _counter_to_dict(row["width"] for row in rows),
        "height_counts": _counter_to_dict(row["height"] for row in rows),
        "mode_counts": _counter_to_dict(row["mode"] for row in rows),
        "channel_counts": _counter_to_dict(row["channels"] for row in rows),
        "duplicate_sha256_groups": duplicates,
        "duplicate_image_count": sum(len(paths) for paths in duplicates.values()),
    }


def build_summary(
    train_root: Path,
    test_root: Path | None,
    output_root: Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    expected_classes: tuple[str, ...],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable audit summary."""
    class_counts = Counter(str(row["label"]) for row in train_rows)
    observed_classes = sorted(class_counts)
    expected_set = set(expected_classes)
    observed_set = set(observed_classes)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_dir": project_relative(train_root),
        "test_dir": project_relative(test_root) if test_root is not None else None,
        "out_dir": project_relative(output_root),
        "expected_classes": list(expected_classes),
        "observed_classes": observed_classes,
        "missing_classes": sorted(expected_set - observed_set),
        "extra_classes": sorted(observed_set - expected_set),
        "class_counts": {label: class_counts.get(label, 0) for label in expected_classes},
        "train_image_count": len(train_rows),
        "test_image_count": len(test_rows),
        "has_test_manifest": bool(test_rows),
        "stats": stats,
    }


def _validate_train_rows(
    train_root: Path,
    train_rows: list[dict[str, Any]],
    expected_classes: tuple[str, ...],
    strict: bool,
) -> None:
    class_counts = Counter(str(row["label"]) for row in train_rows)
    observed = set(class_counts)
    expected = set(expected_classes)

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    invalid = [row for row in train_rows if not row["is_valid"]]

    if strict and missing:
        raise ValueError(f"missing expected class directories in {train_root}: {missing}")
    if strict and extra:
        raise ValueError(f"unexpected class directories in {train_root}: {extra}")
    if strict and invalid:
        bad = [row["rel_path"] for row in invalid[:10]]
        raise ValueError(f"invalid training images found: {bad}")


def _counter_to_dict(values: Any) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
