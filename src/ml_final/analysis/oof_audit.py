"""Sample-level OOF error audits for Scheme01 optimization."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.utils.config import resolve_project_path
from ml_final.utils.paths import project_relative


def audit_oof_errors(
    *,
    oof_path: str | Path,
    out_dir: str | Path,
    manifest_path: str | Path | None = None,
    hard_classes: list[str] | None = None,
    top_k: int = 40,
    montage_cols: int = 5,
) -> dict[str, Any]:
    """Write sample-level OOF error CSV/Markdown and optional image montage."""

    resolved_oof = resolve_project_path(oof_path)
    resolved_out = resolve_project_path(out_dir)
    resolved_manifest = resolve_project_path(manifest_path) if manifest_path else None
    if resolved_oof is None or resolved_out is None:
        raise ValueError("oof_path and out_dir are required")
    resolved_out.mkdir(parents=True, exist_ok=True)

    data = np.load(resolved_oof, allow_pickle=True)
    probs = np.asarray(data["probs"], dtype=np.float64)
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    class_names = [str(item) for item in data["class_names"].tolist()]
    filenames = load_filenames(data, len(y_true))
    manifest = read_manifest_by_filename(resolved_manifest) if resolved_manifest else {}
    hard_set = set(hard_classes or ["Class_2", "Class_3", "Class_4"])

    rows = build_error_rows(probs, y_true, filenames, class_names, manifest, hard_set)
    error_rows = [row for row in rows if not row["correct"]]
    hard_error_rows = [
        row
        for row in error_rows
        if row["true_label"] in hard_set or row["pred_label"] in hard_set
    ]

    metrics = compute_classification_metrics(y_true, probs.argmax(axis=1), class_names)
    all_csv = resolved_out / "oof_sample_rows.csv"
    hard_csv = resolved_out / "hard_class_error_rows.csv"
    write_rows_csv(all_csv, rows)
    write_rows_csv(hard_csv, hard_error_rows)
    report_path = resolved_out / "hard_class_oof_audit.md"
    write_report(
        report_path,
        oof_path=project_relative(resolved_oof),
        manifest_path=project_relative(resolved_manifest) if resolved_manifest else None,
        metrics=metrics,
        class_names=class_names,
        hard_classes=sorted(hard_set),
        hard_error_rows=hard_error_rows,
        top_k=top_k,
    )
    montage_path = write_montage(
        resolved_out / "hard_class_error_montage.png",
        hard_error_rows[:top_k],
        cols=montage_cols,
    )

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "oof_path": project_relative(resolved_oof),
        "manifest_path": project_relative(resolved_manifest) if resolved_manifest else None,
        "num_rows": int(len(rows)),
        "num_errors": int(len(error_rows)),
        "num_hard_errors": int(len(hard_error_rows)),
        "hard_classes": sorted(hard_set),
        "metrics": metrics,
        "all_rows_csv": project_relative(all_csv),
        "hard_errors_csv": project_relative(hard_csv),
        "report_path": project_relative(report_path),
        "montage_path": project_relative(montage_path) if montage_path else None,
    }
    write_json(summary, resolved_out / "summary.json")
    return summary


def compare_oof_audits(
    *,
    audit_dirs: list[str | Path],
    out_dir: str | Path,
    min_wrong_count: int = 2,
) -> dict[str, Any]:
    """Find samples that are repeatedly wrong across OOF audit outputs."""

    resolved_out = resolve_project_path(out_dir)
    if resolved_out is None:
        raise ValueError("out_dir is required")
    resolved_out.mkdir(parents=True, exist_ok=True)
    tables = []
    for audit_dir in audit_dirs:
        resolved = resolve_project_path(audit_dir)
        if resolved is None:
            raise ValueError("audit_dirs cannot contain None")
        rows_path = resolved / "oof_sample_rows.csv"
        if not rows_path.exists():
            raise FileNotFoundError(f"missing audit rows: {rows_path}")
        tables.append((project_relative(resolved), read_audit_rows(rows_path)))

    by_filename: dict[str, list[dict[str, Any]]] = {}
    for audit_name, rows in tables:
        for row in rows:
            item = dict(row)
            item["audit"] = audit_name
            by_filename.setdefault(str(row["filename"]), []).append(item)

    persistent_rows = []
    for filename, rows in sorted(by_filename.items()):
        wrong = [row for row in rows if str(row["correct"]) == "False"]
        if len(wrong) < min_wrong_count:
            continue
        true_labels = sorted({str(row["true_label"]) for row in wrong})
        pred_labels = sorted({str(row["pred_label"]) for row in wrong})
        persistent_rows.append(
            {
                "filename": filename,
                "rel_path": wrong[0].get("rel_path", ""),
                "abs_path": wrong[0].get("abs_path", ""),
                "true_labels": ";".join(true_labels),
                "pred_labels": ";".join(pred_labels),
                "wrong_count": len(wrong),
                "audit_count": len(rows),
                "mean_confidence": float(np.mean([float(row["confidence"]) for row in wrong])),
                "mean_margin": float(np.mean([float(row["margin"]) for row in wrong])),
                "min_true_probability": float(np.min([float(row["true_probability"]) for row in wrong])),
                "audits": ";".join(str(row["audit"]) for row in wrong),
            }
        )
    persistent_rows = sorted(
        persistent_rows,
        key=lambda row: (row["wrong_count"], row["mean_confidence"], row["mean_margin"]),
        reverse=True,
    )
    csv_path = resolved_out / "persistent_hard_errors.csv"
    write_rows_csv_custom(csv_path, persistent_rows)
    pair_summary_path, pair_report_path, pair_montages = write_pair_review(resolved_out, persistent_rows)
    report_path = resolved_out / "persistent_hard_errors.md"
    write_persistent_report(
        report_path,
        persistent_rows,
        tables,
        min_wrong_count=min_wrong_count,
        pair_report_path=pair_report_path,
    )
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit_dirs": [name for name, _ in tables],
        "min_wrong_count": min_wrong_count,
        "num_persistent_errors": len(persistent_rows),
        "csv_path": project_relative(csv_path),
        "report_path": project_relative(report_path),
        "pair_summary_csv": project_relative(pair_summary_path),
        "pair_report_path": project_relative(pair_report_path),
        "pair_montages": [project_relative(path) for path in pair_montages],
    }
    write_json(summary, resolved_out / "summary.json")
    return summary


def load_filenames(data: np.lib.npyio.NpzFile, count: int) -> list[str]:
    if "train_filenames" in data:
        return [str(item) for item in data["train_filenames"].tolist()]
    return [f"row_{idx:05d}" for idx in range(count)]


def read_manifest_by_filename(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def build_error_rows(
    probs: np.ndarray,
    y_true: np.ndarray,
    filenames: list[str],
    class_names: list[str],
    manifest: dict[str, dict[str, str]],
    hard_classes: set[str],
) -> list[dict[str, Any]]:
    pred_idx = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else sorted_probs[:, -1]
    rows = []
    for idx, filename in enumerate(filenames):
        true_label = class_names[int(y_true[idx])]
        pred_label = class_names[int(pred_idx[idx])]
        source = manifest.get(filename, {})
        rows.append(
            {
                "row_index": idx,
                "filename": filename,
                "rel_path": source.get("rel_path", ""),
                "abs_path": source.get("abs_path", ""),
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": bool(true_label == pred_label),
                "confidence": float(probs[idx, pred_idx[idx]]),
                "margin": float(margins[idx]),
                "true_probability": float(probs[idx, int(y_true[idx])]),
                "hard_related": bool(true_label in hard_classes or pred_label in hard_classes),
            }
        )
    return sorted(rows, key=lambda row: (row["correct"], -row["confidence"], row["margin"]))


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows_csv_custom(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "filename",
            "rel_path",
            "abs_path",
            "true_labels",
            "pred_labels",
            "wrong_count",
            "audit_count",
            "mean_confidence",
            "mean_margin",
            "min_true_probability",
            "audits",
        ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_persistent_report(
    path: Path,
    rows: list[dict[str, Any]],
    tables: list[tuple[str, list[dict[str, str]]]],
    *,
    min_wrong_count: int,
    pair_report_path: Path | None = None,
) -> None:
    lines = [
        "# Persistent Hard-Class Error Audit",
        "",
        f"- Audit inputs: {len(tables)}",
        f"- Minimum wrong count: {min_wrong_count}",
        f"- Persistent errors: {len(rows)}",
        "",
        "## Inputs",
        "",
    ]
    if pair_report_path is not None:
        lines.extend(["## Pair Review", "", f"- Pair-level report: `{project_relative(pair_report_path)}`", ""])
    for audit_name, audit_rows in tables:
        wrong_count = sum(1 for row in audit_rows if str(row["correct"]) == "False")
        lines.append(f"- `{audit_name}`: rows={len(audit_rows)}, wrong={wrong_count}")
    lines.extend(
        [
            "",
            "## Top Persistent Errors",
            "",
            "| filename | true | repeated pred labels | wrong_count | mean_conf | mean_margin | min_true_prob |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows[:80]:
        lines.append(
            f"| `{row['filename']}` | {row['true_labels']} | {row['pred_labels']} | "
            f"{row['wrong_count']} | {row['mean_confidence']:.4f} | "
            f"{row['mean_margin']:.4f} | {row['min_true_probability']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pair_review(out_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path, list[Path]]:
    pair_dir = out_dir / "pair_montages"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = summarize_pairs(rows)
    pair_summary_path = out_dir / "persistent_pair_summary.csv"
    write_rows_csv_custom(pair_summary_path, pair_rows)
    montage_paths = []
    for pair in pair_rows:
        pair_key = str(pair["pair_key"])
        pair_items = [
            row
            for row in rows
            if str(row["true_labels"]) == str(pair["true_label"])
            and str(row["pred_labels"]) == str(pair["pred_labels"])
        ]
        montage_path = write_persistent_pair_montage(pair_dir / f"{pair_key}.png", pair_items[:30], cols=5)
        if montage_path is not None:
            montage_paths.append(montage_path)
            pair["montage_path"] = project_relative(montage_path)
        else:
            pair["montage_path"] = ""
    write_rows_csv_custom(pair_summary_path, pair_rows)
    report_path = out_dir / "persistent_pair_review.md"
    write_pair_report(report_path, pair_rows)
    return pair_summary_path, report_path, montage_paths


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["true_labels"]), str(row["pred_labels"]))
        grouped.setdefault(key, []).append(row)
    pair_rows = []
    for (true_label, pred_labels), items in grouped.items():
        pair_rows.append(
            {
                "pair_key": safe_pair_key(true_label, pred_labels),
                "true_label": true_label,
                "pred_labels": pred_labels,
                "count": len(items),
                "mean_confidence": float(np.mean([float(row["mean_confidence"]) for row in items])),
                "mean_margin": float(np.mean([float(row["mean_margin"]) for row in items])),
                "min_true_probability": float(np.min([float(row["min_true_probability"]) for row in items])),
                "montage_path": "",
            }
        )
    return sorted(pair_rows, key=lambda row: (row["count"], row["mean_confidence"]), reverse=True)


def safe_pair_key(true_label: str, pred_labels: str) -> str:
    raw = f"{true_label}_to_{pred_labels}".replace(";", "_or_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)


def write_persistent_pair_montage(path: Path, rows: list[dict[str, Any]], *, cols: int) -> Path | None:
    image_rows = [row for row in rows if row.get("abs_path")]
    if not image_rows:
        return None
    tiles = []
    for row in image_rows:
        image_path = resolve_project_path(row["abs_path"])
        if image_path is None or not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB").resize((96, 96), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (150, 126), "white")
        tile.paste(image, (27, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 98), str(row["filename"])[:24], fill="black")
        draw.text((4, 112), f"p={float(row['mean_confidence']):.2f} m={float(row['mean_margin']):.2f}", fill="black")
        tiles.append(tile)
    if not tiles:
        return None
    cols = max(int(cols), 1)
    row_count = int(np.ceil(len(tiles) / cols))
    montage = Image.new("RGB", (cols * 150, row_count * 126), "white")
    for idx, tile in enumerate(tiles):
        montage.paste(tile, ((idx % cols) * 150, (idx // cols) * 126))
    montage.save(path)
    return path


def write_pair_report(path: Path, pair_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Persistent Error Pair Review",
        "",
        "| true | repeated pred labels | count | mean_conf | mean_margin | montage |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in pair_rows:
        montage = f"`{row['montage_path']}`" if row.get("montage_path") else ""
        lines.append(
            f"| {row['true_label']} | {row['pred_labels']} | {row['count']} | "
            f"{row['mean_confidence']:.4f} | {row['mean_margin']:.4f} | {montage} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    oof_path: str,
    manifest_path: str | None,
    metrics: dict[str, Any],
    class_names: list[str],
    hard_classes: list[str],
    hard_error_rows: list[dict[str, Any]],
    top_k: int,
) -> None:
    lines = [
        "# Scheme01 Hard-Class OOF Audit",
        "",
        f"- OOF: `{oof_path}`",
        f"- Manifest: `{manifest_path}`" if manifest_path else "- Manifest: not provided",
        f"- Macro-F1: {metrics['macro_f1']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- Hard classes: {', '.join(hard_classes)}",
        f"- Hard-related errors: {len(hard_error_rows)}",
        "",
        "## Confusion Matrix",
        "",
        "| true \\ pred | " + " | ".join(class_names) + " |",
        "|---|" + "|".join(["---"] * len(class_names)) + "|",
    ]
    matrix = metrics["confusion_matrix"]
    for name, row in zip(class_names, matrix):
        lines.append("| " + name + " | " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "## Top Hard-Class Errors",
            "",
            "| filename | true | pred | confidence | margin | true_prob |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in hard_error_rows[:top_k]:
        lines.append(
            f"| `{row['filename']}` | {row['true_label']} | {row['pred_label']} | "
            f"{row['confidence']:.4f} | {row['margin']:.4f} | {row['true_probability']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_montage(path: Path, rows: list[dict[str, Any]], *, cols: int) -> Path | None:
    image_rows = [row for row in rows if row.get("abs_path")]
    if not image_rows:
        return None
    thumbs = []
    for row in image_rows:
        image_path = resolve_project_path(row["abs_path"])
        if image_path is None or not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB").resize((96, 96), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (132, 122), "white")
        tile.paste(image, (18, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 98), f"{row['true_label']}->{row['pred_label']}", fill="black")
        draw.text((4, 110), f"p={row['confidence']:.2f} m={row['margin']:.2f}", fill="black")
        thumbs.append(tile)
    if not thumbs:
        return None
    cols = max(int(cols), 1)
    rows_count = int(np.ceil(len(thumbs) / cols))
    montage = Image.new("RGB", (cols * 132, rows_count * 122), "white")
    for idx, tile in enumerate(thumbs):
        montage.paste(tile, ((idx % cols) * 132, (idx // cols) * 122))
    montage.save(path)
    return path
