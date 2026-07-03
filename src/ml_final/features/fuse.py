"""Scheme01 feature-level fusion for frozen encoder bundles."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml_final.features.store import l2_normalize, load_feature_bundle, save_feature_bundle
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths


def run_feature_fusion(config_path: str | Path, *, run_name: str | None = None) -> dict[str, Any]:
    """Fuse aligned Scheme01 feature bundles into one multi-encoder bundle."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme01 feature fusion")
    run_name = run_name or str(config.get("run_name", "s01_feature_fusion"))
    sources = config.get("sources") or []
    if len(sources) < 2:
        raise ValueError("feature fusion requires at least two sources")
    output_root = resolve_project_path(config.get("output_dir", "artifacts/features/scheme_01"))
    if output_root is None:
        raise ValueError("output_dir cannot be None")
    fusion_id = str(config.get("fusion_id", run_name))
    output_dir = output_root / run_name / fusion_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = [load_source_bundle(item) for item in sources]
    validate_aligned_bundles(bundles)
    first = bundles[0]["bundle"]
    block_infos = build_block_infos(sources, bundles)
    train_features = fuse_blocks(
        [np.asarray(item["bundle"]["train_features"], dtype=np.float32) for item in bundles],
        block_infos,
    )
    expanded = all("train_eval_features" in item["bundle"] for item in bundles)
    if any("train_eval_features" in item["bundle"] for item in bundles) and not expanded:
        raise ValueError("all fused sources must either include train_eval_features or omit them")
    train_eval_features = None
    train_origin_filenames = None
    train_origin_indices = None
    if expanded:
        validate_aligned_expanded_bundles(bundles)
        train_eval_features = fuse_blocks(
            [np.asarray(item["bundle"]["train_eval_features"], dtype=np.float32) for item in bundles],
            block_infos,
        )
        train_origin_filenames = np.asarray(first["train_origin_filenames"])
        train_origin_indices = np.asarray(first["train_origin_indices"], dtype=np.int64)
    test_features = None
    if all("test_features" in item["bundle"] for item in bundles):
        test_features = fuse_blocks(
            [np.asarray(item["bundle"]["test_features"], dtype=np.float32) for item in bundles],
            block_infos,
        )

    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fusion_id": fusion_id,
        "fusion_mode": "early_feature_concat",
        "block_normalization": "l2",
        "blocks": block_infos,
        "source_files": [item["path"] for item in bundles],
        "feature_dim": int(train_features.shape[1]),
        "train_count": int(train_features.shape[0]),
        "eval_train_count": int(train_eval_features.shape[0]) if train_eval_features is not None else int(train_features.shape[0]),
        "train_expand_preserved": bool(expanded),
        "test_count": int(test_features.shape[0]) if test_features is not None else 0,
        "teacher_policy": "single fused feature bundle; no prediction ensemble; no TTA required by this fusion step",
    }
    out_path = output_dir / f"{fusion_id}_fused_features.npz"
    save_feature_bundle(
        out_path,
        train_features=train_features,
        train_labels=np.asarray(first["train_labels"], dtype=np.int64),
        train_filenames=np.asarray(first["train_filenames"]),
        class_names=[str(item) for item in first["class_names"].tolist()],
        metadata=metadata,
        test_features=test_features,
        test_filenames=np.asarray(first["test_filenames"]) if test_features is not None else None,
        train_origin_filenames=train_origin_filenames,
        train_origin_indices=train_origin_indices,
        train_eval_features=train_eval_features,
    )
    manifest = {
        "run_name": run_name,
        "fusion_id": fusion_id,
        "feature": {
            "path": project_relative(out_path),
            "feature_dim": int(train_features.shape[1]),
            "train_count": int(train_features.shape[0]),
            "eval_train_count": int(train_eval_features.shape[0]) if train_eval_features is not None else int(train_features.shape[0]),
            "test_count": int(test_features.shape[0]) if test_features is not None else 0,
        },
        "blocks": block_infos,
    }
    manifest_path = output_root / run_name / f"{fusion_id}_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest_path": project_relative(manifest_path), "feature": manifest["feature"], "blocks": block_infos}


def load_source_bundle(source: dict[str, Any]) -> dict[str, Any]:
    path = resolve_project_path(source.get("path"))
    if path is None or not path.exists():
        raise FileNotFoundError(f"feature source not found: {source.get('path')}")
    return {"source": dict(source), "path": project_relative(path), "bundle": load_feature_bundle(path)}


def validate_aligned_bundles(items: list[dict[str, Any]]) -> None:
    ref = items[0]["bundle"]
    ref_labels = np.asarray(ref["train_labels"], dtype=np.int64)
    ref_names = np.asarray(ref["train_filenames"]).astype(str)
    ref_classes = [str(item) for item in ref["class_names"].tolist()]
    for item in items[1:]:
        bundle = item["bundle"]
        if not np.array_equal(np.asarray(bundle["train_labels"], dtype=np.int64), ref_labels):
            raise ValueError(f"train_labels do not align: {item['path']}")
        if not np.array_equal(np.asarray(bundle["train_filenames"]).astype(str), ref_names):
            raise ValueError(f"train_filenames do not align: {item['path']}")
        if [str(value) for value in bundle["class_names"].tolist()] != ref_classes:
            raise ValueError(f"class_names do not align: {item['path']}")
    has_test = ["test_features" in item["bundle"] for item in items]
    if any(has_test) and not all(has_test):
        raise ValueError("all fused sources must either include test_features or omit them")
    if all(has_test):
        ref_test_names = np.asarray(ref["test_filenames"]).astype(str)
        for item in items[1:]:
            current = np.asarray(item["bundle"]["test_filenames"]).astype(str)
            if not np.array_equal(current, ref_test_names):
                raise ValueError(f"test_filenames do not align: {item['path']}")


def validate_aligned_expanded_bundles(items: list[dict[str, Any]]) -> None:
    """Validate D4/train-expanded bundles before preserving eval metadata."""

    ref = items[0]["bundle"]
    ref_origin_names = np.asarray(ref["train_origin_filenames"]).astype(str)
    ref_origin_indices = np.asarray(ref["train_origin_indices"], dtype=np.int64)
    ref_eval_count = int(np.asarray(ref["train_eval_features"]).shape[0])
    for item in items[1:]:
        bundle = item["bundle"]
        origin_names = np.asarray(bundle["train_origin_filenames"]).astype(str)
        origin_indices = np.asarray(bundle["train_origin_indices"], dtype=np.int64)
        eval_count = int(np.asarray(bundle["train_eval_features"]).shape[0])
        if not np.array_equal(origin_names, ref_origin_names):
            raise ValueError(f"train_origin_filenames do not align: {item['path']}")
        if not np.array_equal(origin_indices, ref_origin_indices):
            raise ValueError(f"train_origin_indices do not align: {item['path']}")
        if eval_count != ref_eval_count:
            raise ValueError(f"train_eval_features count does not align: {item['path']}")


def build_block_infos(sources: list[dict[str, Any]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for source, item in zip(sources, items):
        bundle = item["bundle"]
        dim = int(np.asarray(bundle["train_features"]).shape[1])
        infos.append(
            {
                "name": str(source.get("name") or source.get("backbone") or Path(item["path"]).parent.name),
                "path": item["path"],
                "raw_dim": dim,
                "weight": float(source.get("weight", 1.0)),
            }
        )
    return infos


def fuse_blocks(blocks: list[np.ndarray], block_infos: list[dict[str, Any]]) -> np.ndarray:
    fused = []
    for features, info in zip(blocks, block_infos):
        block = l2_normalize(np.asarray(features, dtype=np.float32))
        block *= float(info["weight"])
        fused.append(block)
    return l2_normalize(np.concatenate(fused, axis=1).astype(np.float32))
