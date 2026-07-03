"""Reference-dataset label alignment with pretrained foundation features.

This module is for naming anonymous `Class_X` labels only.  It must not feed
reference labels, reference features, or reference-trained predictions into the
final contest classifier, because the course rules only permit external
pretrained weights and unlabeled test usage for model training.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from ml_final.data.audit import IMAGE_EXTENSIONS
from ml_final.features.extract import build_extractor, iter_tta_views
from ml_final.features.store import l2_normalize
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import ensure_dir, project_relative


def run_reference_alignment(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Align anonymous current classes to a labeled reference dataset."""

    config = load_yaml(config_path)
    run_name = run_name or str(config.get("run_name", "reference_label_alignment"))
    current_manifest = resolve_project_path(config["current_manifest"])
    if current_manifest is None:
        raise ValueError("current_manifest cannot be None")
    current_rows = read_manifest(current_manifest)
    current_classes = sorted({row["label"] for row in current_rows})

    reference_classes = [str(item) for item in config.get("reference_classes", [])]
    if not reference_classes:
        raise ValueError("reference_classes cannot be empty")
    reference_rows = load_reference_rows(config, reference_classes)
    reference_rows = sample_reference_rows(
        reference_rows,
        reference_classes=reference_classes,
        sample_per_class=int(config.get("sample_per_class", 1000)),
        seed=int(config.get("seed", 2026)),
    )

    out_dir = resolve_project_path(config.get("out_dir", f"artifacts/reference_alignment/{run_name}"))
    if out_dir is None:
        raise ValueError("out_dir cannot be None")
    ensure_dir(out_dir)

    backbones = list(config.get("backbones", []))
    if smoke:
        backbones = [{"key": "pixel_stats", "backend": "pixel_stats", "input_size": 32, "pretrained": False}]
    if not backbones:
        raise ValueError("no reference-alignment backbones configured")

    reports = []
    for backbone_cfg in backbones:
        if should_skip_backbone(backbone_cfg, config):
            continue
        report = align_one_backbone(
            backbone_cfg=backbone_cfg,
            current_rows=current_rows,
            reference_rows=reference_rows,
            current_classes=current_classes,
            reference_classes=reference_classes,
            config=config,
            out_dir=out_dir,
        )
        reports.append(report)
    if not reports:
        raise ValueError("all reference-alignment backbones were skipped")

    consensus = build_consensus(
        reports=reports,
        current_classes=current_classes,
        reference_classes=reference_classes,
        prompt_name_map=config.get("prompt_name_map", {}),
    )
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "purpose": "class-name alignment only; not a training source",
        "current_manifest": project_relative(current_manifest),
        "reference_source": describe_reference_source(config),
        "reference_sample_counts": dict(Counter(row["label"] for row in reference_rows)),
        "current_classes": current_classes,
        "reference_classes": reference_classes,
        "backbones": reports,
        "consensus_mapping": consensus,
    }
    summary_path = out_dir / "reference_alignment_summary.json"
    report_path = out_dir / "reference_alignment_report.md"
    mapping_path = resolve_project_path(config.get("mapping_out", "artifacts/selection/reference_class_mapping.yaml"))
    if mapping_path is None:
        raise ValueError("mapping_out cannot be None")
    ensure_dir(mapping_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_alignment_report(summary, report_path)
    mapping_path.write_text(yaml.safe_dump({"mapping": consensus}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "summary_path": project_relative(summary_path),
        "report_path": project_relative(report_path),
        "mapping_path": project_relative(mapping_path),
        "summary": summary,
    }


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reference_rows(config: dict[str, Any], reference_classes: list[str]) -> list[dict[str, str]]:
    """Load reference rows either from a directory or directly from a zip."""

    reference_zip = config.get("reference_zip")
    if reference_zip:
        zip_path = Path(str(reference_zip)).expanduser()
        if not zip_path.exists():
            raise FileNotFoundError(f"reference_zip not found: {zip_path}")
        return scan_reference_zip(zip_path, reference_classes)

    reference_root_raw = config.get("reference_root")
    if not reference_root_raw:
        raise ValueError("set either reference_zip or reference_root")
    reference_root = Path(str(reference_root_raw)).expanduser()
    if not reference_root.exists():
        raise FileNotFoundError(f"reference_root not found: {reference_root}")
    return scan_reference_directory(reference_root, reference_classes)


def scan_reference_zip(zip_path: Path, reference_classes: list[str]) -> list[dict[str, str]]:
    """Scan a labeled reference dataset without extracting it."""

    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith("/") or Path(member).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = infer_reference_label(member, reference_classes)
            if label is None:
                continue
            rows.append(
                {
                    "source": "zip",
                    "zip_path": str(zip_path),
                    "member": member,
                    "filename": Path(member).name,
                    "label": label,
                }
            )
    return rows


def scan_reference_directory(reference_root: Path, reference_classes: list[str]) -> list[dict[str, str]]:
    """Scan a labeled reference dataset directory."""

    rows = []
    for image_path in sorted(reference_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_reference_label(image_path.relative_to(reference_root).as_posix(), reference_classes)
        if label is None:
            continue
        rows.append(
            {
                "source": "file",
                "abs_path": str(image_path),
                "filename": image_path.name,
                "label": label,
            }
        )
    return rows


def infer_reference_label(path_text: str, reference_classes: list[str]) -> str | None:
    """Infer the reference class from any path component."""

    parts = [part for part in Path(path_text).parts if part not in {"", "."}]
    for part in parts:
        if part in reference_classes:
            return part
    return None


def sample_reference_rows(
    rows: list[dict[str, str]],
    *,
    reference_classes: list[str],
    sample_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    """Deterministically cap reference rows per class for fast alignment."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = np.random.default_rng(seed)
    sampled = []
    for label in reference_classes:
        items = grouped.get(label, [])
        if not items:
            raise ValueError(f"reference class has no images: {label}")
        if sample_per_class > 0 and len(items) > sample_per_class:
            indices = sorted(rng.choice(len(items), size=sample_per_class, replace=False).tolist())
            items = [items[idx] for idx in indices]
        sampled.extend(items)
    return sampled


def should_skip_backbone(backbone_cfg: dict[str, Any], config: dict[str, Any]) -> bool:
    """Skip pending gated models unless explicitly requested."""

    if str(backbone_cfg.get("access_status", "")).lower() != "pending":
        return False
    return not bool(config.get("include_pending_backbones", False))


def align_one_backbone(
    *,
    backbone_cfg: dict[str, Any],
    current_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    current_classes: list[str],
    reference_classes: list[str],
    config: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Run prototype, kNN, and optional reference-head alignment for one backbone."""

    key = str(backbone_cfg["key"])
    input_size = int(backbone_cfg.get("input_size", config.get("input_size", 224)))
    batch_size = int(backbone_cfg.get("batch_size", config.get("batch_size", 32)))
    tta = str(config.get("tta", "none"))
    extractor = build_extractor(backbone_cfg)
    current_features = l2_normalize(
        extract_source_rows(current_rows, extractor, input_size=input_size, tta=tta, batch_size=batch_size)
    )
    reference_features = l2_normalize(
        extract_source_rows(reference_rows, extractor, input_size=input_size, tta=tta, batch_size=batch_size)
    )

    current_labels = np.asarray([row["label"] for row in current_rows], dtype=object)
    reference_labels = np.asarray([row["label"] for row in reference_rows], dtype=object)
    current_proto = class_prototypes(current_features, current_labels, current_classes)
    reference_proto = class_prototypes(reference_features, reference_labels, reference_classes)
    cosine = current_proto @ reference_proto.T
    prototype_rows = []
    for idx, class_name in enumerate(current_classes):
        order = np.argsort(-cosine[idx])
        top1 = reference_classes[int(order[0])]
        top2 = reference_classes[int(order[1])] if len(order) > 1 else ""
        prototype_rows.append(
            {
                "class_name": class_name,
                "top1": top1,
                "top2": top2,
                "top1_cosine": float(cosine[idx, order[0]]),
                "top2_cosine": float(cosine[idx, order[1]]) if len(order) > 1 else 0.0,
                "margin": float(cosine[idx, order[0]] - cosine[idx, order[1]]) if len(order) > 1 else 0.0,
                "ranking": [
                    {"label": reference_classes[int(j)], "cosine": float(cosine[idx, j])}
                    for j in order.tolist()
                ],
            }
        )

    knn_rows = build_knn_votes(
        current_features=current_features,
        current_labels=current_labels,
        reference_features=reference_features,
        reference_labels=reference_labels,
        current_classes=current_classes,
        reference_classes=reference_classes,
        k=int(config.get("knn_k", 31)),
    )
    head_rows = (
        build_reference_head_votes(
            current_features=current_features,
            current_labels=current_labels,
            reference_features=reference_features,
            reference_labels=reference_labels,
            current_classes=current_classes,
            reference_classes=reference_classes,
        )
        if bool(config.get("fit_reference_logreg", True))
        else []
    )

    matrix_path = out_dir / f"{key}_prototype_cosine.csv"
    write_matrix_csv(matrix_path, cosine, current_classes, reference_classes)
    return {
        "key": key,
        "backend": str(backbone_cfg.get("backend", "timm")),
        "prototype": prototype_rows,
        "knn": knn_rows,
        "reference_head": head_rows,
        "prototype_matrix_csv": project_relative(matrix_path),
    }


def extract_source_rows(rows: list[dict[str, str]], extractor, *, input_size: int, tta: str, batch_size: int) -> np.ndarray:
    """Extract features for file and zip-backed rows."""

    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    features = []
    with ZipCache() as zip_cache:
        for row in rows:
            image = load_source_image(row, zip_cache)
            views = list(iter_tta_views(image, tta))
            if getattr(extractor, "uses_model_transform", False):
                view_features = extractor.extract_images(views, batch_size=batch_size)
            else:
                resized = [view.resize((input_size, input_size), resample=Image.Resampling.BICUBIC) for view in views]
                view_features = np.stack([extractor(view) for view in resized], axis=0)
            features.append(view_features.mean(axis=0))
    return np.stack(features, axis=0).astype(np.float32)


class ZipCache:
    """Small context manager that keeps zip handles open while scanning rows."""

    def __init__(self) -> None:
        self.handles: dict[str, zipfile.ZipFile] = {}

    def __enter__(self) -> "ZipCache":
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles.values():
            handle.close()

    def get(self, path: str) -> zipfile.ZipFile:
        if path not in self.handles:
            self.handles[path] = zipfile.ZipFile(path)
        return self.handles[path]


def load_source_image(row: dict[str, str], zip_cache: ZipCache) -> Image.Image:
    if row.get("source") == "zip":
        archive = zip_cache.get(row["zip_path"])
        with archive.open(row["member"]) as handle:
            return Image.open(io.BytesIO(handle.read())).convert("RGB")
    image_path = resolve_project_path(row.get("abs_path", ""))
    if image_path is None:
        raise ValueError("file row abs_path cannot be empty")
    return Image.open(image_path).convert("RGB")


def class_prototypes(features: np.ndarray, labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    prototypes = []
    for class_name in class_names:
        mask = labels == class_name
        if not np.any(mask):
            raise ValueError(f"class has no features: {class_name}")
        prototypes.append(features[mask].mean(axis=0))
    return l2_normalize(np.stack(prototypes, axis=0).astype(np.float32))


def build_knn_votes(
    *,
    current_features: np.ndarray,
    current_labels: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    current_classes: list[str],
    reference_classes: list[str],
    k: int,
) -> list[dict[str, Any]]:
    sims = current_features @ reference_features.T
    k = max(1, min(k, reference_features.shape[0]))
    topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = []
    for class_name in current_classes:
        mask = current_labels == class_name
        counts = Counter()
        for row_indices in topk[mask]:
            counts.update(reference_labels[row_indices].tolist())
        total = max(sum(counts.values()), 1)
        distribution = {label: float(counts.get(label, 0) / total) for label in reference_classes}
        top1 = max(reference_classes, key=lambda label: distribution[label])
        rows.append({"class_name": class_name, "top1": top1, "distribution": distribution})
    return rows


def build_reference_head_votes(
    *,
    current_features: np.ndarray,
    current_labels: np.ndarray,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    current_classes: list[str],
    reference_classes: list[str],
) -> list[dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    label_to_idx = {label: idx for idx, label in enumerate(reference_classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_ref = np.asarray([label_to_idx[label] for label in reference_labels], dtype=np.int64)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
    clf.fit(reference_features, y_ref)
    probs = clf.predict_proba(current_features)
    rows = []
    for class_name in current_classes:
        mask = current_labels == class_name
        avg = probs[mask].mean(axis=0)
        order = np.argsort(-avg)
        rows.append(
            {
                "class_name": class_name,
                "top1": idx_to_label[int(order[0])],
                "distribution": {idx_to_label[idx]: float(avg[idx]) for idx in range(len(reference_classes))},
            }
        )
    return rows


def build_consensus(
    *,
    reports: list[dict[str, Any]],
    current_classes: list[str],
    reference_classes: list[str],
    prompt_name_map: dict[str, str],
) -> dict[str, Any]:
    """Vote across prototype, kNN, and reference-head outputs."""

    consensus = {}
    for class_name in current_classes:
        votes = Counter()
        margins = []
        by_backbone = {}
        for report in reports:
            key = report["key"]
            prototype = find_row(report["prototype"], class_name)
            knn = find_row(report["knn"], class_name)
            head = find_row(report["reference_head"], class_name) if report["reference_head"] else None
            by_backbone[key] = {
                "prototype_top1": prototype["top1"],
                "prototype_margin": prototype["margin"],
                "knn_top1": knn["top1"],
                "reference_head_top1": head["top1"] if head else None,
            }
            votes.update([prototype["top1"], knn["top1"]])
            if head:
                votes.update([head["top1"]])
            margins.append(float(prototype["margin"]))
        selected = votes.most_common(1)[0][0]
        consensus[class_name] = {
            "selected_reference_label": selected,
            "prompt_cell_name": str(prompt_name_map.get(selected, selected.lower())),
            "vote_counts": {label: int(votes.get(label, 0)) for label in reference_classes},
            "average_prototype_margin": float(np.mean(margins)) if margins else 0.0,
            "by_backbone": by_backbone,
        }
    return consensus


def find_row(rows: list[dict[str, Any]], class_name: str) -> dict[str, Any]:
    for row in rows:
        if row["class_name"] == class_name:
            return row
    raise KeyError(class_name)


def write_matrix_csv(path: Path, matrix: np.ndarray, row_names: list[str], col_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["current_class", *col_names])
        for idx, row_name in enumerate(row_names):
            writer.writerow([row_name, *[f"{float(value):.8f}" for value in matrix[idx]]])


def describe_reference_source(config: dict[str, Any]) -> dict[str, str]:
    if config.get("reference_zip"):
        return {"type": "zip", "path": str(config["reference_zip"])}
    return {"type": "directory", "path": str(config.get("reference_root", ""))}


def write_alignment_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reference Label Alignment Report",
        "",
        f"Run: `{summary['run_name']}`",
        "",
        "> This report is for naming anonymous classes only. Do not use reference labels or reference-trained heads as contest training data.",
        "",
        "## Consensus Mapping",
        "",
        "| current | selected reference label | prompt name | votes | avg prototype margin |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for class_name, item in summary["consensus_mapping"].items():
        votes = sum(item["vote_counts"].values())
        lines.append(
            f"| `{class_name}` | {item['selected_reference_label']} | {item['prompt_cell_name']} | "
            f"{votes} | {item['average_prototype_margin']:.6f} |"
        )
    lines.extend(["", "## Backbone Details", ""])
    for report in summary["backbones"]:
        lines.append(f"### {report['key']}")
        lines.append("")
        lines.append("| current | prototype top1 | margin | kNN top1 | reference-head top1 |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for row in report["prototype"]:
            class_name = row["class_name"]
            knn = find_row(report["knn"], class_name)
            head = find_row(report["reference_head"], class_name) if report["reference_head"] else None
            lines.append(
                f"| `{class_name}` | {row['top1']} | {row['margin']:.6f} | "
                f"{knn['top1']} | {head['top1'] if head else ''} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
