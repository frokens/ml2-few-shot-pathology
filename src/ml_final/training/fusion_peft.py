"""Scheme02b multi-encoder representation fusion."""

from __future__ import annotations

import datetime as dt
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from ml_final.backbones.factory import build_classifier, freeze_all_except_trainable_adapters_and_head
from ml_final.backbones.module_audit import audit_modules
from ml_final.features.extract import iter_tta_views
from ml_final.features.store import l2_normalize, load_feature_bundle
from ml_final.heads.classical_heads import fit_head, predict_proba
from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.training.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from ml_final.training.data import (
    MultiTransformManifestDataset,
    MultiTransformUnlabeledDataset,
    build_label_mapping,
    read_manifest,
)
from ml_final.training.losses import cross_entropy_with_optional_smoothing
from ml_final.training.losses import focal_cross_entropy
from ml_final.training.losses import mixed_hard_soft_loss
from ml_final.training.losses import mixed_hard_soft_focal_loss
from ml_final.training.peft_train import (
    build_model_transform,
    build_scheduler,
    inject_lora,
    merge_nested_dict,
    merge_transform_config,
    optional_positive_int,
    resolve_amp_dtype,
    resolve_device,
    resolve_official_preprocess,
    resolve_pretrained_cfg,
)
from ml_final.training.pseudo_dataset import MultiTransformPseudoDataset, load_true_and_pseudo_rows
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


class MultiEncoderFusionClassifier(nn.Module):
    """One model that fuses representations from multiple image encoders."""

    def __init__(
        self,
        branches: dict[str, nn.Module],
        feature_dims: dict[str, int],
        *,
        projection_dim: int,
        num_classes: int,
        composer: str = "concat_linear",
        dropout: float = 0.2,
        classifier_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.branch_keys = list(branches)
        self.branches = nn.ModuleDict(branches)
        self.projections = nn.ModuleDict(
            {key: nn.Linear(int(feature_dims[key]), projection_dim) for key in self.branch_keys}
        )
        self.norms = nn.ModuleDict({key: nn.LayerNorm(projection_dim) for key in self.branch_keys})
        self.composer = composer
        self.dropout = nn.Dropout(dropout)
        if composer == "concat_linear":
            classifier_input_dim = projection_dim * len(self.branch_keys)
            self.fusion_norm = nn.LayerNorm(classifier_input_dim)
            self.classifier = build_fusion_classifier_head(
                classifier_input_dim,
                num_classes,
                hidden_dim=classifier_hidden_dim,
            )
            self.gate = None
        elif composer == "gated_sum":
            classifier_input_dim = projection_dim
            self.fusion_norm = nn.LayerNorm(projection_dim)
            self.classifier = build_fusion_classifier_head(
                classifier_input_dim,
                num_classes,
                hidden_dim=classifier_hidden_dim,
            )
            self.gate = nn.Sequential(
                nn.Linear(projection_dim * len(self.branch_keys), len(self.branch_keys)),
                nn.Softmax(dim=1),
            )
        else:
            raise ValueError(f"unsupported fusion composer: {composer}")

    def forward(self, images_by_branch: dict[str, torch.Tensor]) -> torch.Tensor:
        fused, _aux = self.forward_features(images_by_branch)
        return self.classifier(self.dropout(fused))

    def forward_features(self, images_by_branch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the fused representation plus optional composer diagnostics."""

        raw_features = {
            key: extract_classifier_features(self.branches[key], images_by_branch[key])
            for key in self.branch_keys
        }
        return self.forward_from_branch_features(raw_features)

    def forward_from_branch_features(
        self,
        features_by_branch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse already-extracted branch features before the classifier head."""

        parts = []
        for key in self.branch_keys:
            projected = self.projections[key](features_by_branch[key])
            projected = torch.nn.functional.normalize(self.norms[key](projected), dim=1)
            parts.append(projected)
        concat = torch.cat(parts, dim=1)
        aux: dict[str, torch.Tensor] = {}
        if self.composer == "concat_linear":
            fused = self.fusion_norm(concat)
        else:
            if self.gate is None:
                raise RuntimeError("gated_sum composer was not initialized")
            weights = self.gate(concat)
            stacked = torch.stack(parts, dim=1)
            fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
            fused = self.fusion_norm(fused)
            aux["gate_weights"] = weights
        return fused, aux


def build_fusion_classifier_head(
    input_dim: int,
    num_classes: int,
    *,
    hidden_dim: int = 0,
) -> nn.Module:
    """Build the fusion classifier; hidden_dim=0 preserves the legacy linear head."""

    if hidden_dim <= 0:
        return nn.Linear(input_dim, num_classes)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, num_classes),
    )


def run_fusion_peft_cv(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    smoke: bool = False,
    experiment_names: list[str] | None = None,
    pseudolabels: str | Path | None = None,
) -> dict[str, Any]:
    """Run Scheme02b fusion-head CV."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme02b fusion PEFT")
    run_name = run_name or str(config.get("run_name", "scheme02b_fusion_cv"))
    seed = int(config.get("seed", 2026))
    split_seed = int(config.get("split_seed", seed))
    set_seed(seed)
    run_dir = resolve_project_path(f"runs/scheme_02/{run_name}")
    train_manifest = resolve_project_path(config["train_manifest"])
    if run_dir is None or train_manifest is None:
        raise ValueError("run_dir and train_manifest are required")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    checkpoints_root = run_dir / "checkpoints"
    for directory in (metrics_dir, predictions_dir, checkpoints_root):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(train_manifest)
    class_names, label_to_idx = build_label_mapping(rows)
    labels = np.asarray([label_to_idx[row["label"]] for row in rows], dtype=np.int64)
    lambda_pseudo = float(config.get("lambda_pseudo", 0.0))
    pseudo_payload: dict[str, Any] | None = None
    pseudo_rows: list[dict[str, Any]] | None = None
    pseudo_counts: dict[str, int] | None = None
    validation_only = False
    if pseudolabels is not None:
        test_manifest = resolve_project_path(config.get("test_manifest"))
        if test_manifest is None or not test_manifest.exists():
            raise FileNotFoundError("test_manifest is required when pseudolabels are provided")
        pseudo_payload = load_true_and_pseudo_rows(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            pseudolabels=pseudolabels,
            lambda_pseudo=lambda_pseudo,
            min_pseudo_sample_weight=float(config.get("min_pseudo_sample_weight", 0.0)),
            max_pseudo_per_class=config.get("max_pseudo_per_class"),
            true_soft_labels=config.get("true_soft_labels"),
        )
        payload_train_rows = pseudo_payload["train_rows"]
        if len(payload_train_rows) != len(rows):
            raise ValueError("pseudo training rows do not align with train manifest")
        payload_filenames = np.asarray([row["filename"] for row in payload_train_rows], dtype=object)
        manifest_filenames = np.asarray([row["filename"] for row in rows], dtype=object)
        if not np.array_equal(payload_filenames, manifest_filenames):
            raise ValueError("pseudo training row filenames do not align with train manifest")
        payload_labels = np.asarray([row["label"] for row in payload_train_rows], dtype=object)
        manifest_labels = np.asarray([row["label"] for row in rows], dtype=object)
        if not np.array_equal(payload_labels, manifest_labels):
            raise ValueError("pseudo training row labels do not align with train manifest")
        pseudo_rows = list(pseudo_payload["pseudo_rows"])
        pseudo_counts = dict(pseudo_payload["pseudo_counts"])
        rows = list(payload_train_rows)
        validation_only = True
    n_splits = int(config.get("n_splits", 5 if not smoke else 2))
    max_folds = int(config.get("smoke_max_folds", 1)) if smoke else config.get("max_folds")
    split_items = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed).split(
            np.zeros(len(labels)),
            labels,
        )
    )
    if max_folds is not None:
        split_items = split_items[: int(max_folds)]
    experiments = config.get("experiments") or [{"name": "fusion_concat", "composer": "concat_linear"}]
    experiments = filter_experiments(experiments, experiment_names=experiment_names)
    summary_rows = []
    for experiment_cfg in experiments:
        experiment_name = str(experiment_cfg["name"])
        print(
            f"[fusion] experiment={experiment_name} "
            f"composer={experiment_cfg.get('composer', 'concat_linear')} "
            f"folds={len(split_items)}",
            flush=True,
        )
        oof_probs = np.zeros((len(rows), len(class_names)), dtype=np.float64)
        oof_counts = np.zeros(len(rows), dtype=np.float64)
        fold_summaries = []
        fold_features = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_items):
            print(
                f"[fusion] experiment={experiment_name} fold={fold_idx + 1}/{len(split_items)} "
                f"train={len(train_idx)} val={len(val_idx)}",
                flush=True,
            )
            result = train_fusion_fold(
                rows=rows,
                labels=labels,
                class_names=class_names,
                label_to_idx=label_to_idx,
                train_idx=train_idx,
                val_idx=val_idx,
                fold_idx=fold_idx,
                experiment_cfg=experiment_cfg,
                global_config=config,
                run_dir=run_dir,
                checkpoints_root=checkpoints_root,
                pseudo_rows=pseudo_rows,
            )
            used_val_idx = val_idx[: len(result["val_probs"])]
            oof_probs[used_val_idx] += result["val_probs"]
            oof_counts[used_val_idx] += 1.0
            fold_summaries.append(result["summary"])
            if result.get("features") is not None:
                fold_features.append(result["features"])
            print(
                f"[fusion] experiment={experiment_name} fold={fold_idx + 1}/{len(split_items)} "
                f"selection={result['summary']['val_selection_score']:.6f}",
                flush=True,
            )
        oof_probs = oof_probs / np.maximum(oof_counts[:, None], 1.0)
        y_pred = oof_probs.argmax(axis=1)
        metrics = compute_classification_metrics(labels, y_pred, class_names)
        experiment_id = f"fusion__{experiment_name}"
        oof_path = predictions_dir / f"oof_{experiment_id}.npz"
        np.savez_compressed(
            oof_path,
            probs=oof_probs,
            y_true=labels,
            y_pred=y_pred,
            class_names=np.asarray(class_names, dtype=object),
            train_filenames=np.asarray([row["filename"] for row in rows], dtype=object),
            experiment_id=np.asarray(experiment_id, dtype=object),
        )
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "backbone": "multi_encoder",
                "experiment": experiment_name,
                "mode": "fusion_peft_pseudo_cv" if pseudo_rows is not None else "fusion_peft",
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "selection_score": metrics["selection_score"],
                "folds": fold_summaries,
                "oof_path": project_relative(oof_path),
                "test_path": None,
                "test_prediction_kind": None,
                "num_pseudo": len(pseudo_rows or []),
                "pseudo_counts": pseudo_counts,
                "lambda_pseudo": lambda_pseudo,
                "validation_only": validation_only,
                "pseudolabels": project_relative(resolve_project_path(pseudolabels)) if pseudolabels else None,
                "baseline": read_fusion_source_row(config.get("source_summary"), experiment_id)
                if config.get("source_summary")
                else {},
            }
        )
        summary_rows.extend(
            evaluate_fusion_feature_heads(
                head_specs=list(experiment_cfg.get("fusion_heads", config.get("fusion_heads", []))),
                fold_features=fold_features,
                labels=labels,
                rows=rows,
                class_names=class_names,
                predictions_dir=predictions_dir,
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                seed=seed,
            )
        )
    summary_rows = sorted(summary_rows, key=lambda row: row["selection_score"], reverse=True)
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "seed": seed,
        "split_seed": split_seed,
        "best": compact_row(summary_rows[0]) if summary_rows else None,
        "results": [compact_row(row) for row in summary_rows],
        "resolved_config": project_relative(metrics_dir / "resolved_config.json"),
        "pseudolabels": project_relative(resolve_project_path(pseudolabels)) if pseudolabels else None,
        "validation_only": validation_only,
    }
    (metrics_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_json(summary, metrics_dir / "summary.json")
    return {"run_dir": project_relative(run_dir), "summary": summary}


def run_fusion_single_refit(
    config_path: str | Path,
    *,
    experiment_name: str,
    run_name: str | None = None,
    source_summary: str | Path | None = None,
    oof_path: str | Path | None = None,
    pseudolabels: str | Path | None = None,
) -> dict[str, Any]:
    """Train one Scheme02b fusion candidate on all labels and predict test once."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(
        {"config": config, "pseudolabels": pseudolabels},
        context="Scheme02b fusion single refit",
    )
    run_name = run_name or str(config.get("run_name", f"{experiment_name}__single_refit"))
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    run_dir = resolve_project_path(f"runs/scheme_03/{run_name}")
    train_manifest = resolve_project_path(config["train_manifest"])
    test_manifest = resolve_project_path(config.get("test_manifest"))
    if run_dir is None or train_manifest is None:
        raise ValueError("run_dir and train_manifest are required")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    checkpoints_root = run_dir / "checkpoints"
    for directory in (metrics_dir, predictions_dir, checkpoints_root):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(train_manifest)
    test_rows = read_manifest(test_manifest) if test_manifest is not None and test_manifest.exists() else []
    class_names, label_to_idx = build_label_mapping(rows)
    experiment_cfg = resolve_fusion_experiment(config, experiment_name)
    device = resolve_device(config)
    model, transforms_by_key = build_fusion_model(
        config,
        experiment_cfg,
        num_classes=len(class_names),
        fold_idx=int(experiment_cfg.get("adapter_fold_idx", 0)),
    )
    model.to(device)
    freeze_branch_parameters(
        model,
        train_adapter_branches=bool(experiment_cfg.get("train_adapter_branches", False)),
    )

    lambda_pseudo = float(config.get("lambda_pseudo", 0.0))
    pseudo_counts = {name: 0 for name in class_names}
    if pseudolabels:
        payload = load_true_and_pseudo_rows(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            pseudolabels=pseudolabels,
            lambda_pseudo=lambda_pseudo,
            min_pseudo_sample_weight=float(config.get("min_pseudo_sample_weight", 0.0)),
            max_pseudo_per_class=config.get("max_pseudo_per_class"),
            true_soft_labels=config.get("true_soft_labels"),
        )
        train_dataset = MultiTransformPseudoDataset(
            payload["train_rows"],
            payload["pseudo_rows"],
            class_names=payload["class_names"],
            label_to_idx=payload["label_to_idx"],
            transforms_by_key={key: item["train"] for key, item in transforms_by_key.items()},
            n_classes=len(payload["class_names"]),
        )
        pseudo_counts = payload["pseudo_counts"]
        num_pseudo = len(payload["pseudo_rows"])
        use_pseudo_loss = True
    else:
        train_dataset = MultiTransformManifestDataset(
            rows,
            label_to_idx=label_to_idx,
            transforms_by_key={key: item["train"] for key, item in transforms_by_key.items()},
        )
        num_pseudo = 0
        use_pseudo_loss = False

    batch_size = int(experiment_cfg.get("batch_size", config.get("batch_size", 4)))
    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = build_fusion_optimizer(model, experiment_cfg, config)
    epochs = int(experiment_cfg.get("epochs", config.get("epochs", 20)))
    scheduler = build_scheduler(optimizer, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    criterion = cross_entropy_with_optional_smoothing(
        float(experiment_cfg.get("label_smoothing", config.get("label_smoothing", 0.0)))
    )
    class_weight = resolve_class_weight_tensor(
        experiment_cfg,
        config,
        class_names=class_names,
        labels=np.asarray([label_to_idx[row["label"]] for row in rows], dtype=np.int64),
    )
    focal_loss = resolve_focal_loss(experiment_cfg, config)
    pseudo_loss_scale = float(experiment_cfg.get("pseudo_loss_scale", config.get("pseudo_loss_scale", 1.0)))
    true_soft_loss_scale = float(
        experiment_cfg.get("true_soft_loss_scale", config.get("true_soft_loss_scale", 0.0))
    )
    amp_dtype = resolve_amp_dtype(config)
    max_train_batches = optional_positive_int(
        experiment_cfg.get("max_train_batches", config.get("max_train_batches"))
    )
    history = []
    for epoch in range(epochs):
        if use_pseudo_loss:
            train_metrics = train_fusion_pseudo_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device=device,
                amp_dtype=amp_dtype,
                label_smoothing=float(experiment_cfg.get("label_smoothing", config.get("label_smoothing", 0.0))),
                class_weight=class_weight,
                focal_loss=focal_loss,
                pseudo_loss_scale=pseudo_loss_scale,
                true_soft_loss_scale=true_soft_loss_scale,
                max_batches=max_train_batches,
            )
        else:
            train_metrics = train_fusion_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device=device,
                amp_dtype=amp_dtype,
                class_weight=class_weight,
                focal_loss=focal_loss,
                mixup=experiment_cfg.get("mixup", config.get("mixup", {})),
                max_batches=max_train_batches,
            )
        history.append({"epoch": epoch + 1, "train": train_metrics})
        print(
            f"[fusion-refit] experiment={experiment_name} epoch={epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.6f}",
            flush=True,
        )

    experiment_id = f"fusion__{experiment_name}"
    ckpt_path = checkpoints_root / experiment_id / "single_refit" / "single_refit.pt"
    state = CheckpointState(epoch=epochs, best_macro_f1=-1.0, best_balanced_accuracy=-1.0, history=history)
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config={"experiment": experiment_cfg},
        model_scope="trainable",
        save_optimizer=False,
        save_rng=False,
    )
    test_path = None
    submission_path = None
    if test_rows:
        eval_feature_strategies = resolve_eval_feature_strategies(experiment_cfg, config)
        if eval_feature_strategies:
            test_probs, _test_diagnostics = predict_fusion_rows_feature_tta(
                model,
                test_rows,
                transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
                branch_feature_strategies=eval_feature_strategies,
                device=device,
                amp_dtype=amp_dtype,
                batch_size=batch_size,
                max_rows=None,
            )
        else:
            test_dataset = MultiTransformUnlabeledDataset(
                test_rows,
                transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=(device.type == "cuda"),
            )
            test_probs = predict_fusion_unlabeled(
                model,
                test_loader,
                device=device,
                amp_dtype=amp_dtype,
            )
        test_path = predictions_dir / f"test_{experiment_id}__single_refit.npz"
        test_filenames = np.asarray([row["filename"] for row in test_rows[: len(test_probs)]], dtype=object)
        np.savez_compressed(
            test_path,
            probs=test_probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=test_filenames,
            experiment_id=np.asarray(experiment_id, dtype=object),
            prediction_kind=np.asarray("single_refit", dtype=object),
            model_count=np.asarray(1, dtype=np.int64),
        )
        submission_path = write_fusion_submission(test_probs, test_filenames, class_names, run_name)
    source_row = read_fusion_source_row(source_summary, experiment_id)
    resolved_oof = project_relative(resolve_project_path(oof_path)) if oof_path else source_row.get("oof_path")
    result_row = {
        "experiment_id": experiment_id,
        "backbone": "multi_encoder",
        "experiment": experiment_name,
        "mode": "fusion_single_refit_pseudo" if use_pseudo_loss else "fusion_single_refit",
        "macro_f1": float(source_row.get("macro_f1", 0.0)),
        "balanced_accuracy": float(source_row.get("balanced_accuracy", 0.0)),
        "selection_score": float(source_row.get("selection_score", 0.0)),
        "oof_path": resolved_oof,
        "test_path": project_relative(test_path) if test_path else None,
        "test_prediction_kind": "single_refit" if test_path else None,
        "model_count": 1,
        "checkpoint": project_relative(ckpt_path),
        "submission": project_relative(submission_path) if submission_path else None,
        "num_pseudo": int(num_pseudo),
        "pseudo_counts": pseudo_counts,
        "lambda_pseudo": lambda_pseudo,
        "pseudo_loss_scale": pseudo_loss_scale,
        "true_soft_loss_scale": true_soft_loss_scale,
        "adapter_provenance": getattr(model, "adapter_provenance", {}),
    }
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "train_manifest": project_relative(train_manifest),
        "test_manifest": project_relative(test_manifest) if test_manifest and test_manifest.exists() else None,
        "pseudolabels": project_relative(resolve_project_path(pseudolabels)) if pseudolabels else None,
        "best": result_row,
        "results": [result_row],
        "test_prediction_files": [project_relative(test_path)] if test_path else [],
        "submission": project_relative(submission_path) if submission_path else None,
        "resolved_config": project_relative(metrics_dir / "resolved_config.json"),
    }
    (metrics_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_json(summary, metrics_dir / "summary.json")
    return {"run_dir": project_relative(run_dir), "summary": summary}


def predict_fusion_single_refit(
    config_path: str | Path,
    *,
    experiment_name: str,
    checkpoint: str | Path,
    run_name: str | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Load one trained fusion single-refit checkpoint and predict test once."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme02b fusion single-refit prediction")
    run_name = run_name or str(config.get("run_name", f"{experiment_name}__predict"))
    train_manifest = resolve_project_path(config["train_manifest"])
    test_manifest = resolve_project_path(config.get("test_manifest"))
    checkpoint_path = resolve_project_path(checkpoint)
    if train_manifest is None or test_manifest is None or not test_manifest.exists():
        raise FileNotFoundError("test_manifest is required for fusion single-refit prediction")
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"fusion checkpoint not found: {checkpoint}")
    rows = read_manifest(train_manifest)
    test_rows = read_manifest(test_manifest)
    class_names, _label_to_idx = build_label_mapping(rows)
    experiment_cfg = resolve_fusion_experiment(config, experiment_name)
    device = resolve_device(config)
    model, transforms_by_key = build_fusion_model(
        config,
        experiment_cfg,
        num_classes=len(class_names),
        fold_idx=int(experiment_cfg.get("adapter_fold_idx", 0)),
    )
    model.to(device)
    freeze_branch_parameters(
        model,
        train_adapter_branches=bool(experiment_cfg.get("train_adapter_branches", False)),
    )
    load_checkpoint(checkpoint_path, model=model, device=device, model_strict=False)
    batch_size = int(experiment_cfg.get("batch_size", config.get("batch_size", 4)))
    workers = int(config.get("num_workers", 0))
    amp_dtype = resolve_amp_dtype(config)
    eval_feature_strategies = resolve_eval_feature_strategies(experiment_cfg, config)
    if eval_feature_strategies:
        test_probs, _test_diagnostics = predict_fusion_rows_feature_tta(
            model,
            test_rows,
            transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
            branch_feature_strategies=eval_feature_strategies,
            device=device,
            amp_dtype=amp_dtype,
            batch_size=batch_size,
            max_rows=None,
        )
    else:
        test_dataset = MultiTransformUnlabeledDataset(
            test_rows,
            transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
        test_probs = predict_fusion_unlabeled(
            model,
            test_loader,
            device=device,
            amp_dtype=amp_dtype,
        )
    out_dir = resolve_project_path(out or f"runs/scheme_03/{run_name}/predictions")
    if out_dir is None:
        raise ValueError("out cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"fusion__{experiment_name}"
    test_path = out_dir / f"test_{experiment_id}__single_refit.npz"
    test_filenames = np.asarray([row["filename"] for row in test_rows[: len(test_probs)]], dtype=object)
    np.savez_compressed(
        test_path,
        probs=test_probs,
        class_names=np.asarray(class_names, dtype=object),
        test_filenames=test_filenames,
        experiment_id=np.asarray(experiment_id, dtype=object),
        prediction_kind=np.asarray("single_refit", dtype=object),
        model_count=np.asarray(1, dtype=np.int64),
    )
    submission_path = write_fusion_submission(test_probs, test_filenames, class_names, run_name)
    metadata = {
        "run_name": run_name,
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "checkpoint": project_relative(checkpoint_path),
        "test_manifest": project_relative(test_manifest),
        "test_path": project_relative(test_path),
        "submission": project_relative(submission_path),
        "model_count": 1,
        "prediction_kind": "single_refit",
    }
    metadata_path = out_dir / "prediction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_fusion_classical_diagnostic(
    config_path: str | Path,
    *,
    run_name: str | None = None,
    smoke: bool = False,
    diagnostic_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run raw-branch classical fusion diagnostics for Scheme02b B0."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme02b fusion classical diagnostic")
    run_name = run_name or str(config.get("classical_run_name", "scheme02b_fusion_classical_diagnostic"))
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    run_dir = resolve_project_path(f"runs/scheme_02/{run_name}")
    train_manifest = resolve_project_path(config["train_manifest"])
    if run_dir is None or train_manifest is None:
        raise ValueError("run_dir and train_manifest are required")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    features_dir = run_dir / "features"
    for directory in (metrics_dir, predictions_dir, features_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(train_manifest)
    class_names, label_to_idx = build_label_mapping(rows)
    labels = np.asarray([label_to_idx[row["label"]] for row in rows], dtype=np.int64)
    n_splits = int(config.get("n_splits", 5 if not smoke else 2))
    max_folds = int(config.get("smoke_max_folds", 1)) if smoke else config.get("max_folds")
    split_items = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    if max_folds is not None:
        split_items = split_items[: int(max_folds)]

    diagnostics = config.get("classical_diagnostics") or [
        {
            "name": "b0_raw_classical_all4",
            "adapter_branches": [],
            "branch_keys": [str(item["key"]) for item in config.get("branches", [])],
        }
    ]
    diagnostics = filter_experiments(diagnostics, experiment_names=diagnostic_names)
    summary_rows = []
    for diagnostic_cfg in diagnostics:
        diagnostic_name = str(diagnostic_cfg["name"])
        print(f"[fusion-classical] diagnostic={diagnostic_name} folds={len(split_items)}", flush=True)
        fold_features = []
        fold_summaries = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_items):
            print(
                f"[fusion-classical] diagnostic={diagnostic_name} fold={fold_idx + 1}/{len(split_items)} "
                f"train={len(train_idx)} val={len(val_idx)}",
                flush=True,
            )
            if diagnostic_cfg.get("source_feature_bundle"):
                result = load_source_feature_bundle_fold(
                    rows=rows,
                    labels=labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    fold_idx=fold_idx,
                    diagnostic_cfg=diagnostic_cfg,
                    run_dir=run_dir,
                )
            elif diagnostic_cfg.get("reuse_features_from_run"):
                result = load_cached_fusion_fold(
                    train_idx=train_idx,
                    val_idx=val_idx,
                    fold_idx=fold_idx,
                    diagnostic_cfg=diagnostic_cfg,
                    run_dir=run_dir,
                )
            else:
                result = extract_raw_fusion_fold(
                    rows=rows,
                    labels=labels,
                    class_names=class_names,
                    label_to_idx=label_to_idx,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    fold_idx=fold_idx,
                    diagnostic_cfg=diagnostic_cfg,
                    global_config=config,
                    run_dir=run_dir,
                )
            fold_features.append(result["features"])
            fold_summaries.append(result["summary"])
        experiment_id = f"fusion_classical__{diagnostic_name}"
        head_rows = evaluate_fusion_feature_heads(
            head_specs=list(diagnostic_cfg.get("fusion_heads", config.get("fusion_heads", []))),
            fold_features=fold_features,
            labels=labels,
            rows=rows,
            class_names=class_names,
            predictions_dir=predictions_dir,
            experiment_id=experiment_id,
            experiment_name=diagnostic_name,
            seed=seed,
            mode="fusion_classical_head",
        )
        for row in head_rows:
            row["folds"] = fold_summaries
        summary_rows.extend(head_rows)
    summary_rows = sorted(summary_rows, key=lambda row: row["selection_score"], reverse=True)
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "best": compact_row(summary_rows[0]) if summary_rows else None,
        "results": [compact_row(row) for row in summary_rows],
        "resolved_config": project_relative(metrics_dir / "resolved_config.json"),
        "diagnostic_type": "raw_branch_classical_fusion",
    }
    (metrics_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_json(summary, metrics_dir / "summary.json")
    return {"run_dir": project_relative(run_dir), "summary": summary}


def filter_experiments(
    experiments: list[dict[str, Any]],
    *,
    experiment_names: list[str] | None,
) -> list[dict[str, Any]]:
    """Select a named subset of fusion experiments for staged B0/B1/B2/B3 gates."""

    if not experiment_names:
        return list(experiments)
    wanted = {str(name) for name in experiment_names}
    selected = [experiment for experiment in experiments if str(experiment.get("name")) in wanted]
    missing = sorted(wanted - {str(experiment.get("name")) for experiment in selected})
    if missing:
        raise ValueError(f"unknown Scheme02b fusion experiment_name values: {missing}")
    return selected


def resolve_class_weight_tensor(
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
    *,
    class_names: list[str],
    labels: np.ndarray,
    indices: np.ndarray | None = None,
) -> torch.Tensor | None:
    """Resolve optional per-class loss weights for a new experiment config."""

    raw = experiment_cfg.get("class_weighting", global_config.get("class_weighting", {}))
    if not raw:
        return None
    mode = str(raw.get("mode", "none")).lower()
    if mode in {"none", "off", "false"}:
        return None
    if mode == "balanced":
        selected = labels if indices is None else labels[indices]
        counts = np.bincount(selected.astype(np.int64), minlength=len(class_names)).astype(np.float64)
        weights = np.divide(
            float(len(selected)),
            float(len(class_names)) * counts,
            out=np.zeros_like(counts),
            where=counts > 0,
        )
    elif mode == "manual":
        raw_weights = raw.get("weights")
        if raw_weights is None:
            raise ValueError("class_weighting mode=manual requires weights")
        if isinstance(raw_weights, dict):
            weights = np.asarray([float(raw_weights.get(name, 1.0)) for name in class_names], dtype=np.float64)
        else:
            weights = np.asarray([float(value) for value in raw_weights], dtype=np.float64)
            if len(weights) != len(class_names):
                raise ValueError("manual class weights must match class_names length")
    else:
        raise ValueError(f"unsupported class_weighting mode: {mode}")
    if bool(raw.get("normalize_mean", True)):
        mean = float(np.mean(weights[weights > 0])) if np.any(weights > 0) else 1.0
        weights = weights / max(mean, 1e-12)
    return torch.as_tensor(weights, dtype=torch.float32)


def resolve_checkpoint_metric(experiment_cfg: dict[str, Any], global_config: dict[str, Any]) -> str:
    """Resolve the validation metric used to reload the final fold checkpoint."""

    metric = str(experiment_cfg.get("checkpoint_metric", global_config.get("checkpoint_metric", "macro_f1")))
    allowed = {"macro_f1", "balanced_accuracy", "selection_score"}
    if metric not in allowed:
        raise ValueError(f"checkpoint_metric must be one of {sorted(allowed)}, got {metric!r}")
    return metric


def resolve_eval_feature_strategies(
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve optional feature-level TTA strategies for final evaluation."""

    raw = experiment_cfg.get("eval_feature_strategies", global_config.get("eval_feature_strategies", {}))
    return {str(key): dict(value) for key, value in dict(raw or {}).items()}


def resolve_group_aux_loss(
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any] | None:
    """Resolve optional hard-vs-easy auxiliary loss for the fusion head."""

    raw = dict(experiment_cfg.get("group_aux_loss", global_config.get("group_aux_loss", {})) or {})
    if not raw or not bool(raw.get("enabled", True)):
        return None
    weight = float(raw.get("weight", 0.0))
    if weight <= 0:
        return None
    raw_positive = raw.get("positive_classes", raw.get("hard_classes", [2, 3, 4]))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    positive_indices: list[int] = []
    for item in raw_positive:
        if isinstance(item, str) and item in class_to_idx:
            positive_indices.append(class_to_idx[item])
        else:
            positive_indices.append(int(item))
    positive_indices = sorted(set(positive_indices))
    if not positive_indices or len(positive_indices) >= len(class_names):
        raise ValueError("group_aux_loss positive_classes must be a non-empty strict subset")
    invalid = [idx for idx in positive_indices if idx < 0 or idx >= len(class_names)]
    if invalid:
        raise ValueError(f"group_aux_loss positive_classes out of range: {invalid}")
    return {
        "weight": weight,
        "positive_indices": positive_indices,
        "positive_classes": [class_names[idx] for idx in positive_indices],
    }


def resolve_focal_loss(experiment_cfg: dict[str, Any], global_config: dict[str, Any]) -> dict[str, float] | None:
    """Resolve optional focal loss controls for a new experiment config."""

    raw = dict(experiment_cfg.get("focal_loss", global_config.get("focal_loss", {})) or {})
    if not raw or not bool(raw.get("enabled", True)):
        return None
    gamma = float(raw.get("gamma", 0.0))
    if gamma <= 0:
        return None
    return {"gamma": gamma}


def compute_group_aux_loss(
    logits: torch.Tensor,
    hard_labels: torch.Tensor,
    *,
    positive_indices: list[int],
    soft_labels: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    is_pseudo: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a fold-local hard-vs-easy loss from the same class logits."""

    pos = torch.as_tensor(positive_indices, device=logits.device, dtype=torch.long)
    all_indices = torch.arange(logits.shape[1], device=logits.device)
    neg = all_indices[~torch.isin(all_indices, pos)]
    group_logits = torch.stack(
        [
            torch.logsumexp(logits.index_select(1, neg), dim=1),
            torch.logsumexp(logits.index_select(1, pos), dim=1),
        ],
        dim=1,
    )
    hard_group = torch.isin(hard_labels, pos).to(dtype=torch.long)
    if soft_labels is None or sample_weight is None or is_pseudo is None:
        return torch.nn.functional.cross_entropy(group_logits, hard_group)

    true_mask = ~is_pseudo.bool()
    pseudo_mask = is_pseudo.bool()
    losses = []
    if true_mask.any():
        losses.append(torch.nn.functional.cross_entropy(group_logits[true_mask], hard_group[true_mask]))
    if pseudo_mask.any():
        pos_prob = soft_labels[pseudo_mask].index_select(1, pos).sum(dim=1)
        group_targets = torch.stack([1.0 - pos_prob, pos_prob], dim=1).to(dtype=group_logits.dtype)
        log_probs = torch.nn.functional.log_softmax(group_logits[pseudo_mask], dim=1)
        pseudo_losses = -(group_targets * log_probs).sum(dim=1)
        weights = sample_weight[pseudo_mask].to(dtype=pseudo_losses.dtype)
        losses.append((pseudo_losses * weights).sum() / torch.clamp(weights.sum(), min=1e-12))
    if not losses:
        return logits.sum() * 0.0
    return sum(losses)


def describe_class_weighting(class_weight: torch.Tensor | None, class_names: list[str]) -> dict[str, float] | None:
    """Return JSON-friendly class weights for run metadata."""

    if class_weight is None:
        return None
    values = class_weight.detach().cpu().numpy().astype(float)
    return {name: float(values[idx]) for idx, name in enumerate(class_names)}


def resolve_fusion_experiment(config: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    """Resolve one named Scheme02b fusion experiment."""

    for experiment in config.get("experiments", []):
        if str(experiment.get("name")) == str(experiment_name):
            return dict(experiment)
    raise ValueError(f"unknown Scheme02b fusion experiment_name: {experiment_name}")


def load_cached_fusion_fold(
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    diagnostic_cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Load fold-local fused features saved by a previous classical diagnostic."""

    source_run = str(diagnostic_cfg["reuse_features_from_run"])
    source_diagnostic = str(diagnostic_cfg.get("reuse_diagnostic_name", diagnostic_cfg["name"]))
    source_dir = resolve_project_path(f"runs/scheme_02/{source_run}")
    if source_dir is None:
        raise ValueError("reuse_features_from_run cannot resolve to an empty path")
    source_feature_dir = source_dir / "features" / f"fusion_classical__{source_diagnostic}"
    train_feature_path = source_feature_dir / f"fold_{fold_idx}_train.npz"
    val_feature_path = source_feature_dir / f"fold_{fold_idx}_val.npz"
    if not train_feature_path.exists() or not val_feature_path.exists():
        raise FileNotFoundError(f"missing cached fusion features for fold {fold_idx}: {source_feature_dir}")

    train_bundle = np.load(train_feature_path, allow_pickle=True)
    val_bundle = np.load(val_feature_path, allow_pickle=True)
    cached_train_idx = np.asarray(train_bundle["indices"], dtype=np.int64)
    cached_val_idx = np.asarray(val_bundle["indices"], dtype=np.int64)
    if not np.array_equal(cached_train_idx, train_idx[: len(cached_train_idx)]):
        raise ValueError(f"cached train indices do not match fold {fold_idx}")
    if not np.array_equal(cached_val_idx, val_idx[: len(cached_val_idx)]):
        raise ValueError(f"cached val indices do not match fold {fold_idx}")

    X_train = np.asarray(train_bundle["features"], dtype=np.float32)
    X_val = np.asarray(val_bundle["features"], dtype=np.float32)
    source_summary_path = source_dir / "metrics" / f"fusion_classical__{source_diagnostic}" / f"fold_{fold_idx}.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")).get("summary", {})
    source_branch_keys = [str(key) for key in source_summary.get("branch_keys", [])]
    source_block_dims = {str(key): int(value) for key, value in dict(source_summary.get("block_dims", {})).items()}
    selected_branch_keys = [str(key) for key in diagnostic_cfg.get("reuse_branch_keys", source_branch_keys)]
    branch_weights = {
        str(key): float(value)
        for key, value in dict(diagnostic_cfg.get("branch_weights", {})).items()
    }
    if selected_branch_keys and source_block_dims:
        X_train = slice_cached_fused_features(
            X_train,
            source_branch_keys=source_branch_keys,
            source_block_dims=source_block_dims,
            selected_branch_keys=selected_branch_keys,
            branch_weights=branch_weights,
        )
        X_val = slice_cached_fused_features(
            X_val,
            source_branch_keys=source_branch_keys,
            source_block_dims=source_block_dims,
            selected_branch_keys=selected_branch_keys,
            branch_weights=branch_weights,
        )
    diagnostic_name = str(diagnostic_cfg["name"])
    fold_metrics_dir = run_dir / "metrics" / f"fusion_classical__{diagnostic_name}"
    fold_metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fold": fold_idx,
        "experiment": diagnostic_name,
        "mode": "fusion_classical_cached_head_sweep",
        "source_run": source_run,
        "source_diagnostic": source_diagnostic,
        "source_feature_paths": {
            "train": project_relative(train_feature_path),
            "val": project_relative(val_feature_path),
        },
        "source_branch_keys": source_branch_keys,
        "selected_branch_keys": selected_branch_keys,
        "branch_weights": {key: branch_weights.get(key, 1.0) for key in selected_branch_keys},
        "feature_dim": int(X_train.shape[1]),
        "training_controls": {
            "train_adapter_branches": False,
            "test_prediction_kind": None,
        },
    }
    write_json({"summary": summary}, fold_metrics_dir / f"fold_{fold_idx}.json")
    return {
        "summary": summary,
        "features": {
            "fold_idx": fold_idx,
            "train_idx": cached_train_idx,
            "val_idx": cached_val_idx,
            "X_train": X_train,
            "X_val": X_val,
        },
    }


def load_source_feature_bundle_fold(
    *,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    diagnostic_cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Load fold slices from an existing Scheme01-style fused feature bundle."""

    feature_path = resolve_project_path(diagnostic_cfg["source_feature_bundle"])
    if feature_path is None or not feature_path.exists():
        raise FileNotFoundError(f"source_feature_bundle not found: {diagnostic_cfg['source_feature_bundle']}")
    bundle = load_feature_bundle(feature_path)
    X = np.asarray(bundle["train_features"], dtype=np.float32)
    bundle_labels = np.asarray(bundle["train_labels"], dtype=np.int64)
    bundle_filenames = np.asarray(bundle["train_filenames"]).astype(str)
    manifest_filenames = np.asarray([row["filename"] for row in rows]).astype(str)
    if not np.array_equal(bundle_labels, labels):
        raise ValueError(f"source feature labels do not align: {feature_path}")
    if not np.array_equal(bundle_filenames, manifest_filenames):
        raise ValueError(f"source feature filenames do not align: {feature_path}")

    diagnostic_name = str(diagnostic_cfg["name"])
    fold_metrics_dir = run_dir / "metrics" / f"fusion_classical__{diagnostic_name}"
    fold_metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fold": fold_idx,
        "experiment": diagnostic_name,
        "mode": "fusion_classical_source_feature_bundle",
        "source_feature_bundle": project_relative(feature_path),
        "feature_dim": int(X.shape[1]),
        "training_controls": {
            "train_adapter_branches": False,
            "test_prediction_kind": None,
        },
    }
    write_json({"summary": summary}, fold_metrics_dir / f"fold_{fold_idx}.json")
    return {
        "summary": summary,
        "features": {
            "fold_idx": fold_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "X_train": X[train_idx],
            "X_val": X[val_idx],
        },
    }


def slice_cached_fused_features(
    X: np.ndarray,
    *,
    source_branch_keys: list[str],
    source_block_dims: dict[str, int],
    selected_branch_keys: list[str],
    branch_weights: dict[str, float],
) -> np.ndarray:
    """Select/reweight block ranges from a cached Scheme01-style fused feature matrix."""

    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for key in source_branch_keys:
        dim = int(source_block_dims[key])
        offsets[key] = (start, start + dim)
        start += dim
    if start != int(X.shape[1]):
        raise ValueError(f"cached fused feature dim mismatch: expected {start}, got {X.shape[1]}")
    missing = [key for key in selected_branch_keys if key not in offsets]
    if missing:
        raise ValueError(f"selected cached branch keys are not available: {missing}")
    pieces = [
        np.asarray(X[:, offsets[key][0] : offsets[key][1]], dtype=np.float32) * float(branch_weights.get(key, 1.0))
        for key in selected_branch_keys
    ]
    return l2_normalize(np.concatenate(pieces, axis=1).astype(np.float32))


def extract_raw_fusion_fold(
    *,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    class_names: list[str],
    label_to_idx: dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    diagnostic_cfg: dict[str, Any],
    global_config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Extract fold-local raw branch features and Scheme01-style fused features."""

    device = resolve_device(global_config)
    model, transforms_by_key = build_fusion_model(
        global_config,
        diagnostic_cfg,
        num_classes=len(class_names),
        fold_idx=fold_idx,
    )
    model.to(device)
    freeze_branch_parameters(model, train_adapter_branches=False)
    batch_size = int(diagnostic_cfg.get("batch_size", global_config.get("batch_size", 4)))
    workers = int(global_config.get("num_workers", 0))
    amp_dtype = resolve_amp_dtype(global_config)
    max_eval_batches = optional_positive_int(
        diagnostic_cfg.get("max_eval_batches", global_config.get("max_eval_batches"))
    )
    branch_feature_strategies = {
        str(key): dict(value)
        for key, value in dict(diagnostic_cfg.get("branch_feature_strategies", {})).items()
    }
    if branch_feature_strategies:
        eval_transforms = {key: item["eval"] for key, item in transforms_by_key.items()}
        selected_train_rows = [rows[int(index)] for index in train_idx]
        selected_val_rows = [rows[int(index)] for index in val_idx]
        train_blocks = extract_raw_branch_features_with_tta(
            model,
            selected_train_rows,
            transforms_by_key=eval_transforms,
            branch_feature_strategies=branch_feature_strategies,
            device=device,
            amp_dtype=amp_dtype,
            batch_size=batch_size,
            max_rows=max_eval_batches,
        )
        val_blocks = extract_raw_branch_features_with_tta(
            model,
            selected_val_rows,
            transforms_by_key=eval_transforms,
            branch_feature_strategies=branch_feature_strategies,
            device=device,
            amp_dtype=amp_dtype,
            batch_size=batch_size,
            max_rows=max_eval_batches,
        )
    else:
        eval_dataset = MultiTransformManifestDataset(
            rows,
            label_to_idx=label_to_idx,
            transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
        )
        train_loader = DataLoader(
            Subset(eval_dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            Subset(eval_dataset, val_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
        train_blocks = extract_raw_branch_features(
            model,
            train_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
        val_blocks = extract_raw_branch_features(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
    branch_weights = {
        str(key): float(value)
        for key, value in dict(diagnostic_cfg.get("branch_weights", {})).items()
    }
    X_train = fuse_raw_feature_blocks(train_blocks, model.branch_keys, branch_weights=branch_weights)
    X_val = fuse_raw_feature_blocks(val_blocks, model.branch_keys, branch_weights=branch_weights)
    used_train_idx = train_idx[: len(X_train)]
    used_val_idx = val_idx[: len(X_val)]
    diagnostic_name = str(diagnostic_cfg["name"])
    feature_dir = run_dir / "features" / f"fusion_classical__{diagnostic_name}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    train_feature_path = feature_dir / f"fold_{fold_idx}_train.npz"
    val_feature_path = feature_dir / f"fold_{fold_idx}_val.npz"
    np.savez_compressed(train_feature_path, features=X_train, indices=used_train_idx)
    np.savez_compressed(val_feature_path, features=X_val, indices=used_val_idx)
    block_dims = {key: int(train_blocks[key].shape[1]) for key in model.branch_keys}
    summary = {
        "fold": fold_idx,
        "experiment": diagnostic_name,
        "mode": "fusion_classical_diagnostic",
        "branch_keys": list(model.branch_keys),
        "adapter_branches": list(diagnostic_cfg.get("adapter_branches", [])),
        "adapter_provenance": getattr(model, "adapter_provenance", {}),
        "block_normalization": "l2",
        "fusion_normalization": "l2",
        "block_dims": block_dims,
        "branch_weights": {key: branch_weights.get(key, 1.0) for key in model.branch_keys},
        "branch_feature_strategies": {
            key: branch_feature_strategies.get(key, {"tta": "none"})
            for key in model.branch_keys
        },
        "feature_dim": int(X_train.shape[1]),
        "feature_paths": {
            "train": project_relative(train_feature_path),
            "val": project_relative(val_feature_path),
        },
        "training_controls": {
            "max_eval_batches": max_eval_batches,
            "train_adapter_branches": False,
            "test_prediction_kind": None,
        },
    }
    fold_metrics_dir = run_dir / "metrics" / f"fusion_classical__{diagnostic_name}"
    fold_metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json({"summary": summary}, fold_metrics_dir / f"fold_{fold_idx}.json")
    return {
        "summary": summary,
        "features": {
            "fold_idx": fold_idx,
            "train_idx": used_train_idx,
            "val_idx": used_val_idx,
            "X_train": X_train,
            "X_val": X_val,
        },
    }


def train_fusion_fold(
    *,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    class_names: list[str],
    label_to_idx: dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
    run_dir: Path,
    checkpoints_root: Path,
    pseudo_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train one fusion fold."""

    device = resolve_device(global_config)
    model, transforms_by_key = build_fusion_model(
        global_config,
        experiment_cfg,
        num_classes=len(class_names),
        fold_idx=fold_idx,
    )
    model.to(device)
    freeze_branch_parameters(
        model,
        train_adapter_branches=bool(experiment_cfg.get("train_adapter_branches", False)),
    )
    batch_size = int(experiment_cfg.get("batch_size", global_config.get("batch_size", 4)))
    workers = int(global_config.get("num_workers", 0))
    fold_rows = [rows[int(index)] for index in train_idx.tolist()]
    val_dataset = MultiTransformManifestDataset(
        rows,
        label_to_idx=label_to_idx,
        transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
    )
    if pseudo_rows is not None:
        train_dataset = MultiTransformPseudoDataset(
            fold_rows,
            pseudo_rows,
            class_names=class_names,
            label_to_idx=label_to_idx,
            transforms_by_key={key: item["train"] for key, item in transforms_by_key.items()},
            n_classes=len(class_names),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_dataset = MultiTransformManifestDataset(
            rows,
            label_to_idx=label_to_idx,
            transforms_by_key={key: item["train"] for key, item in transforms_by_key.items()},
        )
        train_loader = DataLoader(
            Subset(train_dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = build_fusion_optimizer(model, experiment_cfg, global_config)
    epochs = int(experiment_cfg.get("epochs", global_config.get("epochs", 20)))
    scheduler = build_scheduler(optimizer, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    criterion = cross_entropy_with_optional_smoothing(
        float(experiment_cfg.get("label_smoothing", global_config.get("label_smoothing", 0.0)))
    )
    class_weight = resolve_class_weight_tensor(
        experiment_cfg,
        global_config,
        class_names=class_names,
        labels=labels,
        indices=train_idx,
    )
    amp_dtype = resolve_amp_dtype(global_config)
    group_aux_loss = resolve_group_aux_loss(experiment_cfg, global_config, class_names)
    focal_loss = resolve_focal_loss(experiment_cfg, global_config)
    pseudo_loss_scale = float(
        experiment_cfg.get("pseudo_loss_scale", global_config.get("pseudo_loss_scale", 1.0))
    )
    true_soft_loss_scale = float(
        experiment_cfg.get("true_soft_loss_scale", global_config.get("true_soft_loss_scale", 0.0))
    )
    max_train_batches = optional_positive_int(
        experiment_cfg.get("max_train_batches", global_config.get("max_train_batches"))
    )
    max_eval_batches = optional_positive_int(
        experiment_cfg.get("max_eval_batches", global_config.get("max_eval_batches"))
    )
    patience = int(experiment_cfg.get("early_stopping_patience", global_config.get("early_stopping_patience", 0)))
    checkpoint_metric = resolve_checkpoint_metric(experiment_cfg, global_config)
    bad_epochs = 0
    history = []
    state = CheckpointState(epoch=0, best_macro_f1=-1.0, best_balanced_accuracy=-1.0, history=history)
    best_selection_score = -1.0
    experiment_name = str(experiment_cfg["name"])
    ckpt_path = checkpoints_root / f"fusion__{experiment_name}" / f"fold_{fold_idx}" / "last.pt"
    best_ckpt_path = checkpoints_root / f"fusion__{experiment_name}" / f"fold_{fold_idx}" / "best_macro_f1.pt"
    best_balanced_path = checkpoints_root / f"fusion__{experiment_name}" / f"fold_{fold_idx}" / "best_balanced_accuracy.pt"
    best_selection_path = checkpoints_root / f"fusion__{experiment_name}" / f"fold_{fold_idx}" / "best_selection_score.pt"
    for epoch in range(epochs):
        if pseudo_rows is not None:
            train_metrics = train_fusion_pseudo_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device=device,
                amp_dtype=amp_dtype,
                label_smoothing=float(experiment_cfg.get("label_smoothing", global_config.get("label_smoothing", 0.0))),
                class_weight=class_weight,
                group_aux_loss=group_aux_loss,
                focal_loss=focal_loss,
                pseudo_loss_scale=pseudo_loss_scale,
                true_soft_loss_scale=true_soft_loss_scale,
                max_batches=max_train_batches,
            )
        else:
            train_metrics = train_fusion_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device=device,
                amp_dtype=amp_dtype,
                class_weight=class_weight,
                group_aux_loss=group_aux_loss,
                focal_loss=focal_loss,
                mixup=experiment_cfg.get("mixup", global_config.get("mixup", {})),
                max_batches=max_train_batches,
            )
        val_metrics, _epoch_val_probs, _epoch_pred, epoch_diagnostics = evaluate_fusion_model(
            model,
            val_loader,
            class_names,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
        state.epoch = epoch + 1
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": {
                    "macro_f1": val_metrics["macro_f1"],
                    "balanced_accuracy": val_metrics["balanced_accuracy"],
                    "selection_score": val_metrics["selection_score"],
                },
                "diagnostics": epoch_diagnostics,
            }
        )
        print(
            f"[fusion] experiment={experiment_cfg['name']} fold={fold_idx} "
            f"epoch={epoch + 1}/{epochs} train_loss={train_metrics['loss']:.6f} "
            f"val_selection={val_metrics['selection_score']:.6f}",
            flush=True,
        )
        improved = False
        if val_metrics["macro_f1"] > state.best_macro_f1:
            state.best_macro_f1 = float(val_metrics["macro_f1"])
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                config={"experiment": experiment_cfg},
                model_scope="trainable",
                save_optimizer=False,
                save_rng=False,
            )
            improved = True
        if val_metrics["balanced_accuracy"] > state.best_balanced_accuracy:
            state.best_balanced_accuracy = float(val_metrics["balanced_accuracy"])
            save_checkpoint(
                best_balanced_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                config={"experiment": experiment_cfg},
                model_scope="trainable",
                save_optimizer=False,
                save_rng=False,
            )
            improved = True
        if val_metrics["selection_score"] > best_selection_score:
            best_selection_score = float(val_metrics["selection_score"])
            save_checkpoint(
                best_selection_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                config={"experiment": experiment_cfg},
                model_scope="trainable",
                save_optimizer=False,
                save_rng=False,
            )
            improved = True
        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config={"experiment": experiment_cfg},
            model_scope="trainable",
        )
        bad_epochs = 0 if improved else bad_epochs + 1
        if patience > 0 and bad_epochs >= patience:
            break
    reload_path = {
        "macro_f1": best_ckpt_path,
        "balanced_accuracy": best_balanced_path,
        "selection_score": best_selection_path,
    }[checkpoint_metric]
    if reload_path.exists():
        load_checkpoint(reload_path, model=model, device=device, model_strict=False)
    eval_feature_strategies = resolve_eval_feature_strategies(experiment_cfg, global_config)
    if eval_feature_strategies:
        selected_val_rows = [rows[int(index)] for index in val_idx.tolist()]
        val_probs, val_diagnostics = predict_fusion_rows_feature_tta(
            model,
            selected_val_rows,
            transforms_by_key={key: item["eval"] for key, item in transforms_by_key.items()},
            branch_feature_strategies=eval_feature_strategies,
            device=device,
            amp_dtype=amp_dtype,
            batch_size=batch_size,
            max_rows=max_eval_batches,
        )
        used_labels = labels[val_idx[: len(val_probs)]]
        val_pred = val_probs.argmax(axis=1)
        val_metrics = compute_classification_metrics(used_labels, val_pred, class_names)
    else:
        val_metrics, val_probs, _pred, val_diagnostics = evaluate_fusion_model(
            model,
            val_loader,
            class_names,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
    train_features = val_features = None
    train_feature_idx = val_feature_idx = None
    feature_paths: dict[str, str] = {}
    if bool(experiment_cfg.get("export_fused_features", global_config.get("export_fused_features", False))) or experiment_cfg.get("fusion_heads", global_config.get("fusion_heads")):
        feature_dir = run_dir / "features" / f"fusion__{experiment_name}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        feature_train_loader = DataLoader(
            Subset(val_dataset, train_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
        train_features = extract_fusion_features(
            model,
            feature_train_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
        val_features = extract_fusion_features(
            model,
            val_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_eval_batches,
        )
        train_feature_idx = train_idx[: len(train_features)]
        val_feature_idx = val_idx[: len(val_features)]
        train_feature_path = feature_dir / f"fold_{fold_idx}_train.npz"
        val_feature_path = feature_dir / f"fold_{fold_idx}_val.npz"
        np.savez_compressed(train_feature_path, features=train_features, indices=train_feature_idx)
        np.savez_compressed(val_feature_path, features=val_features, indices=val_feature_idx)
        feature_paths = {
            "train": project_relative(train_feature_path),
            "val": project_relative(val_feature_path),
        }
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config={"experiment": experiment_cfg},
        model_scope="trainable",
    )
    summary = {
        "fold": fold_idx,
        "experiment": experiment_name,
        "mode": "fusion_peft",
        "branch_keys": list(model.branch_keys),
        "adapter_branches": list(experiment_cfg.get("adapter_branches", [])),
        "adapter_provenance": getattr(model, "adapter_provenance", {}),
        "num_pseudo": len(pseudo_rows or []),
        "pseudo_training_enabled": pseudo_rows is not None,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_selection_score": val_metrics["selection_score"],
        "best_macro_f1": state.best_macro_f1,
        "best_balanced_accuracy": state.best_balanced_accuracy,
        "checkpoints": {
            "last": project_relative(ckpt_path),
            "best_macro_f1": project_relative(best_ckpt_path),
            "best_balanced_accuracy": project_relative(best_balanced_path) if best_balanced_path.exists() else None,
            "best_selection_score": project_relative(best_selection_path) if best_selection_path.exists() else None,
            "loaded": project_relative(reload_path) if reload_path.exists() else None,
        },
        "feature_paths": feature_paths,
        "diagnostics": val_diagnostics,
        "training_controls": {
            "max_train_batches": max_train_batches,
            "max_eval_batches": max_eval_batches,
            "early_stopping_patience": patience,
            "checkpoint_model_scope": "trainable",
            "train_adapter_branches": bool(experiment_cfg.get("train_adapter_branches", False)),
            "pseudo_training_enabled": pseudo_rows is not None,
            "num_pseudo": len(pseudo_rows or []),
            "class_weighting": describe_class_weighting(class_weight, class_names),
            "group_aux_loss": group_aux_loss,
            "focal_loss": focal_loss,
            "pseudo_loss_scale": pseudo_loss_scale,
            "true_soft_loss_scale": true_soft_loss_scale,
            "checkpoint_metric": checkpoint_metric,
            "eval_feature_strategies": eval_feature_strategies,
        },
    }
    fold_metrics_dir = run_dir / "metrics" / f"fusion__{experiment_name}"
    fold_metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json({"summary": summary, "history": history}, fold_metrics_dir / f"fold_{fold_idx}.json")
    features_payload = None
    if train_features is not None and val_features is not None:
        features_payload = {
            "fold_idx": fold_idx,
            "train_idx": train_feature_idx,
            "val_idx": val_feature_idx,
            "X_train": train_features,
            "X_val": val_features,
        }
    return {"summary": summary, "val_probs": val_probs, "features": features_payload}


def build_fusion_model(
    config: dict[str, Any],
    experiment_cfg: dict[str, Any],
    *,
    num_classes: int,
    fold_idx: int,
) -> tuple[MultiEncoderFusionClassifier, dict[str, dict[str, Any]]]:
    """Build branch classifiers and their transforms."""

    branches = {}
    feature_dims = {}
    transforms_by_key = {}
    branch_configs = resolve_fusion_branch_configs(config, experiment_cfg)
    if not branch_configs:
        raise ValueError("Scheme02b fusion config requires branches/backbones")
    adapter_sources = dict(config.get("adapter_sources", {}))
    adapter_branches = {str(key) for key in experiment_cfg.get("adapter_branches", [])}
    adapter_provenance: dict[str, dict[str, Any]] = {}
    for branch_cfg in branch_configs:
        key = str(branch_cfg["key"])
        branch_model, input_size = build_classifier(branch_cfg, num_classes=num_classes)
        if key in adapter_branches:
            adapter_cfg = adapter_sources.get(key)
            if not adapter_cfg:
                raise ValueError(f"experiment requested adapter branch {key!r}, but no adapter_sources entry exists")
            branch_model, provenance = load_branch_adapter(
                branch_model,
                adapter_cfg=adapter_cfg,
                branch_key=key,
                fold_idx=fold_idx,
                device=torch.device("cpu"),
            )
            adapter_provenance[key] = provenance
        transform_config = merge_transform_config(
            merge_nested_dict(config.get("augmentation", {}), experiment_cfg.get("augmentation", {})),
            branch_cfg,
            pretrained_cfg=resolve_pretrained_cfg(branch_model),
        )
        official_preprocess = resolve_official_preprocess(branch_model)
        transforms_by_key[key] = {
            "train": build_model_transform(
                transform_config,
                train=True,
                input_size=input_size,
                official_preprocess=official_preprocess,
            ),
            "eval": build_model_transform(
                transform_config,
                train=False,
                input_size=input_size,
                official_preprocess=official_preprocess,
            ),
        }
        branches[key] = branch_model
        feature_dims[key] = int(getattr(branch_model, "feature_dim", branch_cfg.get("feature_dim", 0)))
    model = MultiEncoderFusionClassifier(
        branches,
        feature_dims,
        projection_dim=int(experiment_cfg.get("projection_dim", config.get("projection_dim", 256))),
        num_classes=num_classes,
        composer=str(experiment_cfg.get("composer", "concat_linear")),
        dropout=float(experiment_cfg.get("dropout", 0.2)),
        classifier_hidden_dim=int(experiment_cfg.get("classifier_hidden_dim", 0)),
    )
    model.adapter_provenance = adapter_provenance
    return model, transforms_by_key


def resolve_fusion_branch_configs(
    config: dict[str, Any],
    experiment_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return branch configs after opt-in experiment-local branch overrides."""

    branch_configs = list(config.get("branches", config.get("backbones", [])))
    branch_keys = experiment_cfg.get("branch_keys")
    if branch_keys is not None:
        wanted = {str(key) for key in branch_keys}
        branch_configs = [branch_cfg for branch_cfg in branch_configs if str(branch_cfg["key"]) in wanted]
    overrides = dict(
        merge_nested_dict(
            config.get("branch_overrides", {}),
            experiment_cfg.get("branch_overrides", {}),
        )
    )
    if not overrides:
        return [dict(branch_cfg) for branch_cfg in branch_configs]
    resolved = []
    for branch_cfg in branch_configs:
        key = str(branch_cfg["key"])
        resolved.append(merge_nested_dict(branch_cfg, overrides.get(key)))
    return resolved


def load_branch_adapter(
    model: nn.Module,
    *,
    adapter_cfg: dict[str, Any],
    branch_key: str,
    fold_idx: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Inject one branch adapter and load its fold-local trainable checkpoint."""

    source_run = str(adapter_cfg["source_run"])
    experiment_id = str(adapter_cfg["experiment_id"])
    checkpoint_name = str(adapter_cfg.get("checkpoint_name", "best_macro_f1.pt"))
    if adapter_cfg.get("checkpoint_path"):
        checkpoint = resolve_project_path(adapter_cfg["checkpoint_path"])
        if checkpoint is None:
            raise ValueError(f"adapter checkpoint_path cannot be empty for {branch_key}")
    else:
        checkpoint = resolve_branch_adapter_checkpoint(
            source_run=source_run,
            experiment_id=experiment_id,
            fold_idx=fold_idx,
            checkpoint_name=checkpoint_name,
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing Scheme02a adapter checkpoint for {branch_key}: {checkpoint}")
    audit = audit_modules(
        model,
        target_policy=str(adapter_cfg.get("target_policy", "attention_qkv")),
        explicit_targets=adapter_cfg.get("target_modules"),
    )
    if not audit.selected_lora_targets:
        raise ValueError(f"adapter branch {branch_key} selected no LoRA target modules")
    model = inject_lora(model, adapter_cfg, audit.selected_lora_targets)
    freeze_all_except_trainable_adapters_and_head(model)
    load_checkpoint(checkpoint, model=model, device=device, model_strict=False)
    return model, {
        "source_run": source_run,
        "experiment_id": experiment_id,
        "fold": None if adapter_cfg.get("checkpoint_path") else fold_idx,
        "checkpoint": project_relative(checkpoint),
        "target_policy": str(adapter_cfg.get("target_policy", "attention_qkv")),
        "selected_lora_targets": audit.selected_lora_targets,
    }


def resolve_branch_adapter_checkpoint(
    *,
    source_run: str,
    experiment_id: str,
    fold_idx: int,
    checkpoint_name: str = "best_macro_f1.pt",
) -> Path:
    """Resolve the fold-local Scheme02a checkpoint used by one fusion branch."""

    source_dir = resolve_project_path(f"runs/scheme_02/{source_run}")
    if source_dir is None:
        raise ValueError("source_run cannot resolve to a project path")
    return source_dir / "checkpoints" / experiment_id / f"fold_{fold_idx}" / checkpoint_name


def freeze_branch_parameters(
    model: MultiEncoderFusionClassifier,
    *,
    train_adapter_branches: bool,
) -> None:
    """Freeze branch parameters unless a B4-style adapter-tuning run asks otherwise."""

    for branch in model.branches.values():
        if train_adapter_branches:
            freeze_all_except_trainable_adapters_and_head(branch)
        else:
            for parameter in branch.parameters():
                parameter.requires_grad = False
    for module in (model.projections, model.norms, model.fusion_norm, model.classifier, model.gate):
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True
    if not train_adapter_branches:
        for branch in model.branches.values():
            branch.eval()


def build_fusion_optimizer(
    model: MultiEncoderFusionClassifier,
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build AdamW, optionally using a smaller opt-in LR for branch adapters."""

    lr = float(experiment_cfg.get("lr", global_config.get("head_lr", 1e-3)))
    weight_decay = float(experiment_cfg.get("weight_decay", global_config.get("weight_decay", 0.01)))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adapter_lr_raw = experiment_cfg.get("adapter_lr", global_config.get("adapter_lr"))
    if adapter_lr_raw is None:
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    branch_param_ids = {
        id(parameter)
        for branch in model.branches.values()
        for parameter in branch.parameters()
        if parameter.requires_grad
    }
    adapter_params = [parameter for parameter in trainable if id(parameter) in branch_param_ids]
    head_params = [parameter for parameter in trainable if id(parameter) not in branch_param_ids]
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": float(adapter_lr_raw)})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_fusion_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
    class_weight: torch.Tensor | None = None,
    group_aux_loss: dict[str, Any] | None = None,
    focal_loss: dict[str, float] | None = None,
    mixup: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Train one fusion epoch."""

    model.train()
    if isinstance(model, MultiEncoderFusionClassifier):
        for branch in model.branches.values():
            branch.eval()
    total_loss = 0.0
    total_count = 0
    for batch_idx, (images_by_branch, labels, _filenames) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images_by_branch = move_images_to_device(images_by_branch, device)
        labels = labels.to(device, non_blocking=True)
        images_by_branch, mixup_target = maybe_mixup_images_by_branch(
            images_by_branch,
            labels,
            alpha=float(dict(mixup or {}).get("alpha", 0.0)),
            probability=float(dict(mixup or {}).get("probability", 0.0)),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images_by_branch)
            if focal_loss is not None:
                loss = focal_cross_entropy(
                    logits,
                    labels,
                    gamma=float(focal_loss["gamma"]),
                    label_smoothing=float(getattr(criterion, "label_smoothing", 0.0)),
                    class_weight=class_weight,
                )
            elif mixup_target is not None:
                y_a, y_b, lam = mixup_target
                kwargs: dict[str, Any] = {
                    "label_smoothing": float(getattr(criterion, "label_smoothing", 0.0)),
                }
                if class_weight is not None:
                    kwargs["weight"] = class_weight.to(device=device, dtype=logits.dtype)
                loss = lam * torch.nn.functional.cross_entropy(logits, y_a, **kwargs) + (
                    1.0 - lam
                ) * torch.nn.functional.cross_entropy(logits, y_b, **kwargs)
            elif class_weight is None:
                loss = criterion(logits, labels)
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    weight=class_weight.to(device=device, dtype=logits.dtype),
                    label_smoothing=float(getattr(criterion, "label_smoothing", 0.0)),
                )
            if group_aux_loss is not None:
                aux = compute_group_aux_loss(
                    logits,
                    labels,
                    positive_indices=list(group_aux_loss["positive_indices"]),
                )
                loss = loss + float(group_aux_loss["weight"]) * aux
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.detach().cpu()) * labels.size(0)
        total_count += labels.size(0)
    return {"loss": total_loss / max(total_count, 1)}


def maybe_mixup_images_by_branch(
    images_by_branch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    alpha: float,
    probability: float,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, float] | None]:
    """Apply one synchronized mixup permutation to every fusion branch."""

    if alpha <= 0 or probability <= 0 or labels.numel() < 2:
        return images_by_branch, None
    if torch.rand((), device=labels.device).item() > probability:
        return images_by_branch, None
    lam = float(torch.distributions.Beta(alpha, alpha).sample().to(labels.device).item())
    permutation = torch.randperm(labels.size(0), device=labels.device)
    mixed = {
        key: lam * value + (1.0 - lam) * value.index_select(0, permutation)
        for key, value in images_by_branch.items()
    }
    return mixed, (labels, labels.index_select(0, permutation), lam)


def train_fusion_pseudo_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    label_smoothing: float,
    max_batches: int | None,
    class_weight: torch.Tensor | None = None,
    group_aux_loss: dict[str, Any] | None = None,
    focal_loss: dict[str, float] | None = None,
    pseudo_loss_scale: float = 1.0,
    true_soft_loss_scale: float = 0.0,
) -> dict[str, float]:
    """Train one fusion epoch on true labels plus weighted soft pseudo labels."""

    model.train()
    if isinstance(model, MultiEncoderFusionClassifier):
        for branch in model.branches.values():
            branch.eval()
    total_loss = 0.0
    total_count = 0
    total_pseudo = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images_by_branch = move_images_to_device(batch["images_by_branch"], device)
        hard_labels = batch["hard_label"].to(device, non_blocking=True)
        soft_labels = batch["soft_label"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        is_pseudo = batch["is_pseudo"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images_by_branch)
            if focal_loss is not None:
                loss = mixed_hard_soft_focal_loss(
                    logits,
                    hard_labels,
                    soft_labels,
                    sample_weight,
                    is_pseudo,
                    gamma=float(focal_loss["gamma"]),
                    label_smoothing=label_smoothing,
                    class_weight=class_weight,
                    pseudo_loss_scale=pseudo_loss_scale,
                    true_soft_loss_scale=true_soft_loss_scale,
                )
            else:
                loss = mixed_hard_soft_loss(
                    logits,
                    hard_labels,
                    soft_labels,
                    sample_weight,
                    is_pseudo,
                    label_smoothing=label_smoothing,
                    class_weight=class_weight,
                    pseudo_loss_scale=pseudo_loss_scale,
                    true_soft_loss_scale=true_soft_loss_scale,
                )
            if group_aux_loss is not None:
                aux = compute_group_aux_loss(
                    logits,
                    hard_labels,
                    positive_indices=list(group_aux_loss["positive_indices"]),
                    soft_labels=soft_labels,
                    sample_weight=sample_weight,
                    is_pseudo=is_pseudo,
                )
                loss = loss + float(group_aux_loss["weight"]) * aux
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.detach().cpu()) * hard_labels.size(0)
        total_count += hard_labels.size(0)
        total_pseudo += int(is_pseudo.detach().cpu().sum())
    return {
        "loss": total_loss / max(total_count, 1),
        "pseudo_fraction": total_pseudo / max(total_count, 1),
    }


def evaluate_fusion_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate a fusion model."""

    model.eval()
    if isinstance(model, MultiEncoderFusionClassifier):
        for branch in model.branches.values():
            branch.eval()
    probs = []
    labels_seen = []
    gate_weights = []
    with torch.no_grad():
        for batch_idx, (images_by_branch, labels, _filenames) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images_by_branch = move_images_to_device(images_by_branch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                if hasattr(model, "forward_features"):
                    fused, aux = model.forward_features(images_by_branch)
                    logits = model.classifier(model.dropout(fused))
                    if "gate_weights" in aux:
                        gate_weights.append(aux["gate_weights"].detach().float().cpu().numpy())
                else:
                    logits = model(images_by_branch)
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
            labels_seen.append(labels.detach().cpu().numpy())
    probs_arr = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(class_names)))
    labels_arr = np.concatenate(labels_seen, axis=0) if labels_seen else np.zeros(0, dtype=np.int64)
    pred = probs_arr.argmax(axis=1)
    diagnostics: dict[str, Any] = {}
    if gate_weights and isinstance(model, MultiEncoderFusionClassifier):
        gate_arr = np.concatenate(gate_weights, axis=0)
        diagnostics["gate_mean"] = {
            key: float(value)
            for key, value in zip(model.branch_keys, gate_arr.mean(axis=0), strict=True)
        }
    return compute_classification_metrics(labels_arr, pred, class_names), probs_arr, pred, diagnostics


def predict_fusion_unlabeled(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    """Predict probabilities for an unlabeled multi-branch manifest."""

    model.eval()
    if isinstance(model, MultiEncoderFusionClassifier):
        for branch in model.branches.values():
            branch.eval()
    probs = []
    with torch.no_grad():
        for images_by_branch, _filenames in loader:
            images_by_branch = move_images_to_device(images_by_branch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images_by_branch)
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float64)


def predict_fusion_rows_feature_tta(
    model: MultiEncoderFusionClassifier,
    rows: list[dict[str, str]],
    *,
    transforms_by_key: dict[str, Any],
    branch_feature_strategies: dict[str, dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    max_rows: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict once from per-branch features averaged across deterministic views."""

    blocks = extract_raw_branch_features_with_tta(
        model,
        rows,
        transforms_by_key=transforms_by_key,
        branch_feature_strategies=branch_feature_strategies,
        device=device,
        amp_dtype=amp_dtype,
        batch_size=batch_size,
        max_rows=max_rows,
    )
    num_rows = min(len(rows), max_rows) if max_rows is not None else len(rows)
    if num_rows <= 0:
        return np.zeros((0, 0), dtype=np.float64), {"eval_feature_strategies": branch_feature_strategies}
    probs = []
    gate_weights = []
    model.eval()
    for branch in model.branches.values():
        branch.eval()
    with torch.no_grad():
        for start in range(0, num_rows, max(1, batch_size)):
            end = min(start + max(1, batch_size), num_rows)
            feature_batch = {
                key: torch.as_tensor(blocks[key][start:end], dtype=torch.float32, device=device)
                for key in model.branch_keys
            }
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                fused, aux = model.forward_from_branch_features(feature_batch)
                logits = model.classifier(model.dropout(fused))
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
            if "gate_weights" in aux:
                gate_weights.append(aux["gate_weights"].detach().float().cpu().numpy())
    diagnostics: dict[str, Any] = {
        "eval_feature_strategies": {
            key: dict(branch_feature_strategies.get(key, {"tta": "none"}))
            for key in model.branch_keys
        },
        "eval_feature_level_tta": True,
    }
    if gate_weights:
        gate_arr = np.concatenate(gate_weights, axis=0)
        diagnostics["gate_mean"] = {
            key: float(value)
            for key, value in zip(model.branch_keys, gate_arr.mean(axis=0), strict=True)
        }
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float64), diagnostics


def extract_fusion_features(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
) -> np.ndarray:
    """Extract deterministic fused representations for OOF head diagnostics."""

    if not hasattr(model, "forward_features"):
        raise ValueError("fusion feature export requires MultiEncoderFusionClassifier.forward_features")
    model.eval()
    if isinstance(model, MultiEncoderFusionClassifier):
        for branch in model.branches.values():
            branch.eval()
    features = []
    with torch.no_grad():
        for batch_idx, (images_by_branch, _labels, _filenames) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images_by_branch = move_images_to_device(images_by_branch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                fused, _aux = model.forward_features(images_by_branch)
            features.append(fused.detach().float().cpu().numpy())
    if not features:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(features, axis=0)


def evaluate_fusion_feature_heads(
    *,
    head_specs: list[dict[str, Any]],
    fold_features: list[dict[str, Any]],
    labels: np.ndarray,
    rows: list[dict[str, str]],
    class_names: list[str],
    predictions_dir: Path,
    experiment_id: str,
    experiment_name: str,
    seed: int,
    mode: str = "fusion_adapted_head",
) -> list[dict[str, Any]]:
    """Fit Scheme01-style heads on fold-local fused features."""

    if not head_specs or not fold_features:
        return []
    summary_rows = []
    head_namespace = "adapted" if mode == "fusion_adapted_head" else "classical"
    for spec in head_specs:
        head_id = f"{experiment_id}__{head_namespace}__{spec['name']}"
        oof_probs = np.zeros((len(rows), len(class_names)), dtype=np.float64)
        oof_counts = np.zeros(len(rows), dtype=np.float64)
        fold_metrics = []
        for fold in fold_features:
            fold_idx = int(fold["fold_idx"])
            train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
            val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
            head = fit_head(fold["X_train"], labels[train_idx], spec, seed=seed + fold_idx)
            probs = predict_proba(head, fold["X_val"])
            oof_probs[val_idx] += probs
            oof_counts[val_idx] += 1.0
            pred = probs.argmax(axis=1)
            fold_metrics.append(
                {"fold": fold_idx, **compute_classification_metrics(labels[val_idx], pred, class_names)}
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
                "backbone": "multi_encoder",
                "experiment": f"{experiment_name}__{head_namespace}__{spec['name']}",
                "mode": mode,
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "selection_score": metrics["selection_score"],
                "folds": fold_metrics,
                "oof_path": project_relative(oof_path),
                "test_path": None,
                "test_prediction_kind": None,
            }
        )
    return summary_rows


def extract_raw_branch_features(
    model: MultiEncoderFusionClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    """Extract raw pre-projection branch features for B0 parity diagnostics."""

    model.eval()
    for branch in model.branches.values():
        branch.eval()
    blocks: dict[str, list[np.ndarray]] = {key: [] for key in model.branch_keys}
    with torch.no_grad():
        for batch_idx, (images_by_branch, _labels, _filenames) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images_by_branch = move_images_to_device(images_by_branch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                for key in model.branch_keys:
                    features = extract_classifier_features(model.branches[key], images_by_branch[key])
                    blocks[key].append(features.detach().float().cpu().numpy())
    return {
        key: np.concatenate(items, axis=0) if items else np.zeros((0, 0), dtype=np.float32)
        for key, items in blocks.items()
    }


def extract_raw_branch_features_with_tta(
    model: MultiEncoderFusionClassifier,
    rows: list[dict[str, str]],
    *,
    transforms_by_key: dict[str, Any],
    branch_feature_strategies: dict[str, dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    batch_size: int,
    max_rows: int | None,
) -> dict[str, np.ndarray]:
    """Extract per-branch features with feature-level view aggregation."""

    model.eval()
    for branch in model.branches.values():
        branch.eval()
    selected_rows = rows[:max_rows] if max_rows is not None else rows
    blocks: dict[str, np.ndarray] = {}
    effective_batch_size = max(1, batch_size)
    with torch.no_grad():
        for key in model.branch_keys:
            branch = model.branches[key]
            transform = transforms_by_key[key]
            strategy = dict(branch_feature_strategies.get(key, {}))
            tta = str(strategy.get("tta", "none"))
            color_tta = strategy.get("color_tta")
            row_sums: list[np.ndarray | None] = [None] * len(selected_rows)
            row_counts = np.zeros(len(selected_rows), dtype=np.int64)
            pending_views: list[torch.Tensor] = []
            pending_rows: list[int] = []

            def flush_pending() -> None:
                if not pending_views:
                    return
                batch = torch.stack(pending_views, dim=0).to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    features = extract_classifier_features(branch, batch)
                feature_array = features.detach().float().cpu().numpy()
                for row_idx, feature in zip(pending_rows, feature_array, strict=True):
                    if row_sums[row_idx] is None:
                        row_sums[row_idx] = feature.astype(np.float32, copy=True)
                    else:
                        row_sums[row_idx] += feature.astype(np.float32, copy=False)
                    row_counts[row_idx] += 1
                pending_views.clear()
                pending_rows.clear()

            for row_idx, row in enumerate(selected_rows):
                image_path = resolve_project_path(row["abs_path"])
                if image_path is None:
                    raise ValueError("manifest row path cannot be empty")
                image = Image.open(image_path).convert("RGB")
                for view in iter_tta_views(image, tta, color_tta=color_tta):
                    tensor = transform(view) if transform is not None else view
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"branch {key} transform must return a torch.Tensor")
                    pending_views.append(tensor)
                    pending_rows.append(row_idx)
                    if len(pending_views) >= effective_batch_size:
                        flush_pending()
            flush_pending()
            missing = [idx for idx, count in enumerate(row_counts.tolist()) if count <= 0]
            if missing:
                raise RuntimeError(f"branch {key} produced no TTA views for rows: {missing[:5]}")
            blocks[key] = np.stack(
                [np.asarray(row_sums[idx], dtype=np.float32) / float(row_counts[idx]) for idx in range(len(selected_rows))],
                axis=0,
            ).astype(np.float32)
            print(
                f"[fusion-classical] extracted branch={key} rows={len(selected_rows)} "
                f"views={int(row_counts.sum())} batch_size={effective_batch_size}",
                flush=True,
            )
    return blocks


def fuse_raw_feature_blocks(
    blocks: dict[str, np.ndarray],
    branch_keys: list[str],
    *,
    branch_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Apply Scheme01-style block L2 normalization, concat, and final L2 normalization."""

    weights = branch_weights or {}
    normalized = [
        l2_normalize(np.asarray(blocks[key], dtype=np.float32)) * float(weights.get(key, 1.0))
        for key in branch_keys
    ]
    if not normalized:
        return np.zeros((0, 0), dtype=np.float32)
    return l2_normalize(np.concatenate(normalized, axis=1).astype(np.float32))


def move_images_to_device(images_by_branch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a collated branch-image dictionary to the training device."""

    return {key: value.to(device, non_blocking=True) for key, value in images_by_branch.items()}


def extract_classifier_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return pre-classifier features from an ImageClassifier-like branch."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "extract_features"):
        raise ValueError("fusion branches must expose ImageClassifier.extract_features")
    if any(parameter.requires_grad for parameter in model.parameters()):
        return base.extract_features(images)
    with torch.no_grad():
        return base.extract_features(images)


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compact a fusion result for Scheme02 selection."""

    row_out = {
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
    for key in ("num_pseudo", "pseudo_counts", "lambda_pseudo", "validation_only", "pseudolabels", "baseline"):
        if key in row:
            row_out[key] = row[key]
    return row_out


def read_fusion_source_row(source_summary: str | Path | None, experiment_id: str) -> dict[str, Any]:
    """Read CV metrics for a fusion refit candidate when available."""

    if source_summary is None:
        return {}
    path = resolve_project_path(source_summary)
    if path is None or not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    for row in summary.get("results", []):
        if row.get("experiment_id") == experiment_id:
            return dict(row)
    if (summary.get("best") or {}).get("experiment_id") == experiment_id:
        return dict(summary["best"])
    return {}


def write_fusion_submission(
    probs: np.ndarray,
    filenames: np.ndarray,
    class_names: list[str],
    run_name: str,
) -> Path:
    """Write a submission CSV for a single fusion refit prediction."""

    out_dir = resolve_project_path("artifacts/submissions")
    if out_dir is None:
        raise ValueError("submission dir cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"submission_{run_name}.csv"
    labels = [class_names[int(idx)] for idx in probs.argmax(axis=1)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        writer.writerows(zip([str(item) for item in filenames.tolist()], labels))
    return path
