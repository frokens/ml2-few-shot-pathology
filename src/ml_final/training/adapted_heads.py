"""Scheme02 adapted-feature classical heads."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from ml_final.backbones.factory import (
    build_classifier,
    freeze_all_except_trainable_adapters_and_head,
    merge_model_overrides,
)
from ml_final.backbones.module_audit import audit_modules
from ml_final.heads.classical_heads import fit_head, predict_proba
from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.training.checkpointing import load_checkpoint
from ml_final.training.data import ManifestImageDataset, build_label_mapping, read_manifest
from ml_final.training.peft_refit import resolve_experiment
from ml_final.training.peft_train import (
    build_model_transform,
    inject_lora,
    merge_nested_dict,
    merge_transform_config,
    optional_positive_int,
    resolve_amp_dtype,
    resolve_device,
    resolve_official_preprocess,
    resolve_pretrained_cfg,
)
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


DEFAULT_HEADS = [
    {"name": "logreg_C1", "family": "logreg", "C": 1.0},
    {
        "name": "logreg_C3_frofa_hard_l0.05x1",
        "family": "logreg",
        "C": 3.0,
        "train_augmentation": {
            "type": "frofa_brightness_c2",
            "target_classes": [2, 3, 4],
            "level": 0.05,
            "copies": 1,
        },
    },
]


def run_adapted_head_cv(
    config_path: str | Path,
    *,
    source_run: str,
    experiment_id: str,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate classical heads on fold-local LoRA-adapted features."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme02 adapted-head CV")
    backbone_cfg, experiment_cfg = resolve_experiment(config, experiment_id)
    run_name = run_name or f"{source_run}__{experiment_id}__adapted_heads"
    seed = int(config.get("seed", 2026))
    set_seed(seed)

    run_dir = resolve_project_path(f"runs/scheme_02/{run_name}")
    source_dir = resolve_project_path(f"runs/scheme_02/{source_run}")
    train_manifest = resolve_project_path(config["train_manifest"])
    if run_dir is None or source_dir is None or train_manifest is None:
        raise ValueError("run_dir, source_dir, and train_manifest are required")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(train_manifest)
    class_names, label_to_idx = build_label_mapping(rows)
    labels = np.asarray([label_to_idx[row["label"]] for row in rows], dtype=np.int64)
    n_splits = int(config.get("n_splits", 5))
    max_folds = config.get("max_folds")
    split_items = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    if max_folds is not None:
        split_items = split_items[: int(max_folds)]
    head_specs = list(config.get("adapted_heads", DEFAULT_HEADS))
    max_eval_batches = optional_positive_int(config.get("max_eval_batches"))
    device = resolve_device(config)
    amp_dtype = resolve_amp_dtype(config)
    batch_size = int(config.get("batch_size", 16))
    loader_workers = int(config.get("num_workers", 0))

    fold_features = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_items):
        checkpoint = source_dir / "checkpoints" / experiment_id / f"fold_{fold_idx}" / "best_macro_f1.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing fold checkpoint for adapted heads: {checkpoint}")
        print(f"[adapted-heads] extracting fold {fold_idx + 1}/{len(split_items)} from {checkpoint}", flush=True)
        model, input_size = build_adapted_model(
            backbone_cfg=backbone_cfg,
            experiment_cfg=experiment_cfg,
            num_classes=len(class_names),
            checkpoint=checkpoint,
            device=device,
        )
        transform_config = merge_transform_config(
            merge_nested_dict(config.get("augmentation", {}), experiment_cfg.get("augmentation", {})),
            backbone_cfg,
            pretrained_cfg=resolve_pretrained_cfg(model),
        )
        transform = build_model_transform(
            transform_config,
            train=False,
            input_size=input_size,
            official_preprocess=resolve_official_preprocess(model),
        )
        dataset = ManifestImageDataset(rows, label_to_idx=label_to_idx, transform=transform)
        train_loader = make_feature_loader(
            dataset,
            train_idx,
            batch_size=batch_size,
            num_workers=loader_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = make_feature_loader(
            dataset,
            val_idx,
            batch_size=batch_size,
            num_workers=loader_workers,
            pin_memory=device.type == "cuda",
        )
        X_train = extract_features(
            model,
            train_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=None,
        )
        X_val = extract_features(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
        fold_features.append(
            {
                "fold_idx": fold_idx,
                "train_idx": train_idx[: len(X_train)],
                "val_idx": val_idx[: len(X_val)],
                "X_train": X_train,
                "X_val": X_val,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = []
    for spec_idx, spec in enumerate(head_specs):
        head_id = f"{experiment_id}__adapted__{spec['name']}"
        print(f"[adapted-heads] fitting head {spec_idx + 1}/{len(head_specs)}: {spec['name']}", flush=True)
        oof_probs = np.zeros((len(rows), len(class_names)), dtype=np.float64)
        oof_counts = np.zeros(len(rows), dtype=np.float64)
        fold_metrics = []
        for fold in fold_features:
            fold_idx = int(fold["fold_idx"])
            used_train_idx = fold["train_idx"]
            used_val_idx = fold["val_idx"]
            head = fit_head(fold["X_train"], labels[used_train_idx], spec, seed=seed + fold_idx)
            probs = predict_proba(head, fold["X_val"])
            oof_probs[used_val_idx] += probs
            oof_counts[used_val_idx] += 1.0
            pred = probs.argmax(axis=1)
            fold_metrics.append(
                {"fold": fold_idx, **compute_classification_metrics(labels[used_val_idx], pred, class_names)}
            )
        oof_probs = oof_probs / np.maximum(oof_counts[:, None], 1.0)
        pred = oof_probs.argmax(axis=1)
        metrics = compute_classification_metrics(labels, pred, class_names)
        oof_path = predictions_dir / f"oof_{head_id}.npz"
        np.savez_compressed(
            oof_path,
            probs=oof_probs,
            y_true=labels,
            y_pred=pred,
            class_names=np.asarray(class_names, dtype=object),
            train_filenames=np.asarray([row["filename"] for row in rows], dtype=object),
            experiment_id=np.asarray(head_id, dtype=object),
            source_experiment_id=np.asarray(experiment_id, dtype=object),
        )
        summary_rows.append(
            {
                "experiment_id": head_id,
                "backbone": str(backbone_cfg["key"]),
                "experiment": str(spec["name"]),
                "mode": "adapted_head",
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "selection_score": metrics["selection_score"],
                "metrics": metrics,
                "folds": fold_metrics,
                "oof_path": project_relative(oof_path),
                "test_path": None,
                "test_prediction_kind": None,
            }
        )
    summary_rows = sorted(summary_rows, key=lambda row: row["selection_score"], reverse=True)
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "source_run": source_run,
        "source_experiment_id": experiment_id,
        "best": compact_row(summary_rows[0]) if summary_rows else None,
        "results": [compact_row(row) for row in summary_rows],
        "resolved_config": project_relative(metrics_dir / "resolved_config.json"),
    }
    (metrics_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_json(summary, metrics_dir / "summary.json")
    return {"run_dir": project_relative(run_dir), "summary": summary}


def make_feature_loader(
    dataset: ManifestImageDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a deterministic feature-extraction loader."""

    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_adapted_model(
    *,
    backbone_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    num_classes: int,
    checkpoint: Path,
    device: torch.device,
):
    """Rebuild one fold model and load its trainable checkpoint."""

    model_cfg = merge_model_overrides(backbone_cfg, experiment_cfg)
    model, input_size = build_classifier(model_cfg, num_classes=num_classes)
    model.to(device)
    if str(experiment_cfg.get("mode", "head_only")) == "lora":
        audit = audit_modules(
            model,
            target_policy=str(experiment_cfg.get("target_policy", "attention_qkv")),
            explicit_targets=experiment_cfg.get("target_modules"),
        )
        model = inject_lora(model, experiment_cfg, audit.selected_lora_targets)
        model.to(device)
        freeze_all_except_trainable_adapters_and_head(model)
    load_checkpoint(checkpoint, model=model, device=device, model_strict=False)
    model.eval()
    return model, input_size


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
) -> np.ndarray:
    """Extract classifier backbone features with adapters active."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "extract_features"):
        raise ValueError("adapted feature extraction requires ImageClassifier.extract_features")
    features = []
    with torch.no_grad():
        for batch_idx, (images, _labels, _filenames) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = base.extract_features(images)
            features.append(out.detach().float().cpu().numpy())
    if not features:
        return np.zeros((0, int(getattr(base, "feature_dim", 0))), dtype=np.float32)
    return np.concatenate(features, axis=0)


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compact an adapted-head result for Scheme02 selection."""

    return {
        "experiment_id": row["experiment_id"],
        "backbone": row["backbone"],
        "experiment": row["experiment"],
        "mode": row["mode"],
        "macro_f1": row["macro_f1"],
        "balanced_accuracy": row["balanced_accuracy"],
        "selection_score": row["selection_score"],
        "oof_path": row["oof_path"],
        "test_path": row.get("test_path"),
        "test_prediction_kind": row.get("test_prediction_kind"),
    }
