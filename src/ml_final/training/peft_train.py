"""Scheme 02 LP-FT and LoRA CV training.

The implementation deliberately separates three concerns:

* module audit decides LoRA target modules from the actual model graph;
* linear probing freezes the backbone and trains only the classifier head;
* LoRA CV uses Hugging Face PEFT for adapter injection and keeps the base
  backbone frozen.
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from ml_final.backbones.factory import (
    build_classifier,
    count_parameters,
    enable_gradient_checkpointing,
    freeze_all_except_trainable_adapters_and_head,
    freeze_backbone,
    freeze_backbone_except_last_blocks,
    merge_model_overrides,
)
from ml_final.backbones.module_audit import (
    assert_lora_targets_valid,
    audit_modules,
    write_module_audit,
)
from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.training.checkpointing import (
    CheckpointState,
    ModelEma,
    load_checkpoint,
    save_checkpoint,
)
from ml_final.training.data import (
    ManifestImageDataset,
    build_label_mapping,
    build_official_preprocess_transform,
    build_transform,
    read_manifest,
)
from ml_final.training.pseudo_dataset import UnlabeledManifestDataset
from ml_final.training.losses import (
    cross_entropy_with_optional_smoothing,
    maybe_mixup,
    mixed_cross_entropy,
)
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


def run_peft_cv(
    config_path: str | Path,
    *,
    selected_backbones: str | Path | None = None,
    run_name: str | None = None,
    resume: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run Scheme 02 cross-validation experiments."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme02 PEFT training")
    run_name = run_name or str(config.get("run_name", "scheme02_peft_cv"))
    seed = int(config.get("seed", 2026))
    set_seed(seed)

    run_dir = resolve_project_path(f"runs/scheme_02/{run_name}")
    if run_dir is None:
        raise ValueError("run_dir cannot be None")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    module_audit_dir = run_dir / "module_audit"
    checkpoints_root = run_dir / "checkpoints"
    for directory in (metrics_dir, predictions_dir, module_audit_dir, checkpoints_root):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "summary.json"
    if bool(config.get("skip_completed", False)) and summary_path.exists() and not resume:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {"run_dir": project_relative(run_dir), "summary": summary, "skipped": True}

    train_manifest = resolve_project_path(config["train_manifest"])
    test_prediction_mode = resolve_test_prediction_mode(config, smoke=smoke)
    test_manifest = resolve_project_path(config.get("test_manifest"))
    if train_manifest is None:
        raise ValueError("train_manifest cannot be None")
    rows = read_manifest(train_manifest)
    test_rows = (
        read_manifest(test_manifest)
        if test_prediction_mode == "cv_mean_debug" and test_manifest and test_manifest.exists()
        else []
    )
    class_names, label_to_idx = build_label_mapping(rows)
    labels = np.asarray([label_to_idx[row["label"]] for row in rows], dtype=np.int64)

    selected_keys = load_selected_backbones(selected_backbones)
    backbone_configs = list(config.get("backbones", []))
    if smoke:
        backbone_configs = [dict(config.get("smoke_backbone", {"key": "tiny_cnn", "backend": "tiny_cnn"}))]
    if selected_keys:
        backbone_configs = [item for item in backbone_configs if str(item["key"]) in selected_keys]
    if not backbone_configs:
        raise ValueError("no Scheme 02 backbones selected")

    experiment_configs = list(config.get("experiments", []))
    if smoke:
        experiment_configs = list(
            config.get(
                "smoke_experiments",
                [
                    {
                        "name": "head_only_smoke",
                        "mode": "head_only",
                        "epochs": 1,
                        "batch_size": 32,
                    }
                ],
            )
        )
    if not experiment_configs:
        raise ValueError("no Scheme 02 experiments configured")

    n_splits = int(config.get("n_splits", 5 if not smoke else 2))
    max_folds = config.get("max_folds")
    if smoke:
        max_folds = int(config.get("smoke_max_folds", 2))

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_items = list(splitter.split(np.zeros(len(labels)), labels))
    if max_folds is not None:
        split_items = split_items[: int(max_folds)]

    summary_rows = []
    oof_prediction_files = []
    resolved_config_path = metrics_dir / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    for backbone_cfg in backbone_configs:
        backbone_key = str(backbone_cfg["key"])
        for experiment_cfg in experiment_configs:
            if not experiment_applies_to_backbone(experiment_cfg, backbone_key):
                continue
            experiment_name = str(experiment_cfg["name"])
            experiment_id = f"{backbone_key}__{experiment_name}"
            oof_probs = np.zeros((len(rows), len(class_names)), dtype=np.float64)
            oof_counts = np.zeros(len(rows), dtype=np.float64)
            test_prob_folds = []
            fold_summaries = []
            for fold_idx, (train_idx, val_idx) in enumerate(split_items):
                fold_result = train_one_fold(
                    rows=rows,
                    test_rows=test_rows,
                    labels=labels,
                    class_names=class_names,
                    label_to_idx=label_to_idx,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    fold_idx=fold_idx,
                    backbone_cfg=backbone_cfg,
                    experiment_cfg=experiment_cfg,
                    global_config=config,
                    run_dir=run_dir,
                    checkpoints_root=checkpoints_root,
                    module_audit_dir=module_audit_dir,
                    resume=resume,
                )
                used_val_idx = val_idx[: len(fold_result["val_probs"])]
                oof_probs[used_val_idx] += fold_result["val_probs"]
                oof_counts[used_val_idx] += 1.0
                if fold_result.get("test_probs") is not None:
                    test_prob_folds.append(fold_result["test_probs"])
                fold_summaries.append(fold_result["summary"])
            oof_probs = oof_probs / np.maximum(oof_counts[:, None], 1.0)
            y_pred = oof_probs.argmax(axis=1)
            metrics = compute_classification_metrics(labels, y_pred, class_names)
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
            oof_prediction_files.append(project_relative(oof_path))
            test_path = None
            if test_prob_folds:
                test_probs = np.mean(test_prob_folds, axis=0)
                test_path = predictions_dir / f"test_{experiment_id}.npz"
                np.savez_compressed(
                    test_path,
                    probs=test_probs,
                    class_names=np.asarray(class_names, dtype=object),
                    test_filenames=np.asarray([row["filename"] for row in test_rows], dtype=object),
                    experiment_id=np.asarray(experiment_id, dtype=object),
                )
            summary_rows.append(
                {
                    "experiment_id": experiment_id,
                    "backbone": backbone_key,
                    "experiment": experiment_name,
                    "mode": experiment_cfg.get("mode", "head_only"),
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "selection_score": metrics["selection_score"],
                    "metrics": metrics,
                    "folds": fold_summaries,
                    "oof_path": project_relative(oof_path),
                    "test_path": project_relative(test_path) if test_path else None,
                    "test_prediction_kind": "cv_mean_debug" if test_path else None,
                }
            )

    summary_rows = sorted(summary_rows, key=lambda row: row["selection_score"], reverse=True)
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "train_manifest": project_relative(train_manifest),
        "n_splits": n_splits,
        "evaluated_folds": len(split_items),
        "class_names": class_names,
        "best": compact_row(summary_rows[0]) if summary_rows else None,
        "results": [compact_row(row) for row in summary_rows],
        "oof_prediction_files": oof_prediction_files,
        "test_prediction_files": [
            row["test_path"] for row in summary_rows if row.get("test_path")
        ],
        "test_prediction_mode": test_prediction_mode,
        "resolved_config": project_relative(resolved_config_path),
    }
    write_json(summary, metrics_dir / "summary.json")
    write_selection_report(summary, metrics_dir / "selection_report.md")
    return {"run_dir": project_relative(run_dir), "summary": summary}


def experiment_applies_to_backbone(experiment_cfg: dict[str, Any], backbone_key: str) -> bool:
    """Return whether an experiment should run for the given backbone."""

    allowed = experiment_cfg.get("backbone_keys")
    if allowed is None:
        allowed = experiment_cfg.get("backbones")
    if allowed is None:
        return True
    if isinstance(allowed, str):
        allowed_keys = {allowed}
    else:
        allowed_keys = {str(item) for item in allowed}
    return backbone_key in allowed_keys


def dataloader_performance_kwargs(config: dict[str, Any], *, num_workers: int, device: torch.device) -> dict[str, Any]:
    """Return optional DataLoader throughput knobs."""

    if num_workers <= 0:
        return {}
    kwargs: dict[str, Any] = {}
    if bool(config.get("persistent_workers", False)):
        kwargs["persistent_workers"] = True
    prefetch_factor = optional_positive_int(config.get("prefetch_factor"))
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if device.type == "cuda" and bool(config.get("pin_memory_device_cuda", False)):
        kwargs["pin_memory_device"] = "cuda"
    return kwargs


def train_one_fold(
    *,
    rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    labels: np.ndarray,
    class_names: list[str],
    label_to_idx: dict[str, int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_idx: int,
    backbone_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
    run_dir: Path,
    checkpoints_root: Path,
    module_audit_dir: Path,
    resume: str | Path | None,
) -> dict[str, Any]:
    """Train one CV fold and return validation probabilities."""

    backbone_key = str(backbone_cfg["key"])
    experiment_name = str(experiment_cfg["name"])
    mode = str(experiment_cfg.get("mode", "head_only"))
    set_seed(int(global_config.get("seed", 2026)) + fold_idx)

    device = resolve_device(global_config)
    model_cfg = merge_model_overrides(backbone_cfg, experiment_cfg)
    model, input_size = build_classifier(model_cfg, num_classes=len(class_names))
    model.to(device)
    if bool(experiment_cfg.get("gradient_checkpointing", global_config.get("gradient_checkpointing", False))):
        enable_gradient_checkpointing(model)

    audit = audit_modules(
        model,
        target_policy=str(experiment_cfg.get("target_policy", "attention_qkv")),
        explicit_targets=experiment_cfg.get("target_modules"),
    )
    audit_paths = write_module_audit(audit, module_audit_dir, model_key=backbone_key)

    if mode == "head_only":
        freeze_backbone(model)
    elif mode in {"last_blocks", "last_block_ft"}:
        selected_blocks = freeze_backbone_except_last_blocks(
            model,
            num_blocks=int(experiment_cfg.get("num_blocks", 1)),
            train_norm=bool(experiment_cfg.get("train_norm", True)),
        )
        experiment_cfg = {
            **experiment_cfg,
            "selected_trainable_blocks": selected_blocks,
        }
    elif mode == "lora":
        model = inject_lora(model, experiment_cfg, audit.selected_lora_targets)
        model.to(device)
        freeze_all_except_trainable_adapters_and_head(model)
    else:
        raise ValueError(f"unsupported Scheme 02 experiment mode: {mode}")

    parameter_counts = count_parameters(model)
    transform_config = merge_transform_config(
        merge_nested_dict(global_config.get("augmentation", {}), experiment_cfg.get("augmentation", {})),
        backbone_cfg,
        pretrained_cfg=resolve_pretrained_cfg(model),
    )
    official_preprocess = resolve_official_preprocess(model)
    train_transform = build_model_transform(
        transform_config,
        train=True,
        input_size=input_size,
        official_preprocess=official_preprocess,
    )
    val_transform = build_model_transform(
        transform_config,
        train=False,
        input_size=input_size,
        official_preprocess=official_preprocess,
    )
    train_dataset = ManifestImageDataset(rows, label_to_idx=label_to_idx, transform=train_transform)
    val_dataset = ManifestImageDataset(rows, label_to_idx=label_to_idx, transform=val_transform)
    test_dataset = UnlabeledManifestDataset(test_rows, transform=val_transform) if test_rows else None
    batch_size = int(experiment_cfg.get("batch_size", global_config.get("batch_size", 16)))
    workers = int(global_config.get("num_workers", 0))
    loader_kwargs = dataloader_performance_kwargs(global_config, num_workers=workers, device=device)
    train_loader = DataLoader(
        Subset(train_dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
            **loader_kwargs,
        )
        if test_dataset is not None
        else None
    )

    optimizer = build_optimizer(model, experiment_cfg, global_config)
    epochs = int(experiment_cfg.get("epochs", global_config.get("epochs", 30)))
    scheduler = build_scheduler(optimizer, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    class_weight = resolve_class_weight_tensor(global_config, experiment_cfg, class_names=class_names)
    criterion = cross_entropy_with_optional_smoothing(
        float(experiment_cfg.get("label_smoothing", global_config.get("label_smoothing", 0.0))),
        weight=class_weight.to(device) if class_weight is not None else None,
    )
    amp_dtype = resolve_amp_dtype(global_config)
    ema = (
        ModelEma(model, decay=float(experiment_cfg.get("ema_decay", global_config.get("ema_decay", 0.0))))
        if float(experiment_cfg.get("ema_decay", global_config.get("ema_decay", 0.0))) > 0
        else None
    )
    max_grad_norm = float(experiment_cfg.get("max_grad_norm", global_config.get("max_grad_norm", 1.0)))
    max_train_batches = optional_positive_int(
        experiment_cfg.get("max_train_batches", global_config.get("max_train_batches"))
    )
    max_eval_batches = optional_positive_int(
        experiment_cfg.get("max_eval_batches", global_config.get("max_eval_batches"))
    )

    fold_ckpt_dir = checkpoints_root / f"{backbone_key}__{experiment_name}" / f"fold_{fold_idx}"
    fold_metrics_dir = run_dir / "metrics" / f"{backbone_key}__{experiment_name}"
    fold_metrics_dir.mkdir(parents=True, exist_ok=True)
    events_path = fold_metrics_dir / f"fold_{fold_idx}_events.jsonl"
    complete_path = fold_ckpt_dir / "COMPLETE"
    last_path = fold_ckpt_dir / "last.pt"
    best_macro_path = fold_ckpt_dir / "best_macro_f1.pt"
    best_balanced_path = fold_ckpt_dir / "best_balanced_accuracy.pt"
    ema_path = fold_ckpt_dir / "ema.pt"
    state = CheckpointState(epoch=0, best_macro_f1=-1.0, best_balanced_accuracy=-1.0, history=[])
    if bool(experiment_cfg.get("skip_completed", global_config.get("skip_completed", False))) and complete_path.exists():
        if best_macro_path.exists():
            load_checkpoint(best_macro_path, model=model, ema=ema, device=device)
        val_metrics, val_probs, val_pred, eval_payload = evaluate_model_for_selection(
            model,
            val_loader,
            labels[val_idx],
            class_names,
            device=device,
            amp_dtype=amp_dtype,
            ema=ema,
            max_batches=max_eval_batches,
        )
        test_probs = (
            predict_unlabeled_model_for_selection(
                model,
                test_loader,
                device=device,
                amp_dtype=amp_dtype,
                ema=ema,
                max_batches=max_eval_batches,
            )
            if test_loader is not None
            else None
        )
        return {
            "summary": {
                "fold": fold_idx,
                "backbone": backbone_key,
                "experiment": experiment_name,
                "mode": mode,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_selection_score": val_metrics["selection_score"],
                "best_macro_f1": val_metrics["macro_f1"],
                "best_balanced_accuracy": val_metrics["balanced_accuracy"],
                "parameter_counts": parameter_counts,
                "module_audit": audit_paths,
                "training_controls": training_controls_summary(
                    ema=ema,
                    max_grad_norm=max_grad_norm,
                    max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches,
                ),
                "eval": eval_payload,
                "checkpoints": {
                    "last": project_relative(last_path),
                    "best_macro_f1": project_relative(best_macro_path),
                    "best_balanced_accuracy": project_relative(best_balanced_path),
                    "ema": project_relative(ema_path) if ema_path.exists() else None,
                },
                "skipped_completed": True,
            },
            "val_probs": val_probs,
            "val_pred": val_pred,
            "test_probs": test_probs,
        }
    resume_path = resolve_resume_path(resume, last_path)
    if resume_path is not None and resume_path.exists():
        state = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
        )

    patience = int(experiment_cfg.get("early_stopping_patience", global_config.get("early_stopping_patience", 0)))
    bad_epochs = 0
    for epoch in range(state.epoch, epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device=device,
            amp_dtype=amp_dtype,
            mixup=experiment_cfg.get("mixup", global_config.get("mixup", {})),
            ema=ema,
            max_grad_norm=max_grad_norm,
            max_batches=max_train_batches,
        )
        val_metrics, val_probs, _, eval_payload = evaluate_model_for_selection(
            model,
            val_loader,
            labels[val_idx],
            class_names,
            device=device,
            amp_dtype=amp_dtype,
            ema=ema,
            max_batches=max_eval_batches,
        )
        epoch_item = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": {
                "macro_f1": val_metrics["macro_f1"],
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "selection_score": val_metrics["selection_score"],
            },
            "eval": eval_payload,
        }
        state.history.append(epoch_item)
        append_event(events_path, epoch_item)
        state.epoch = epoch + 1

        improved = False
        if val_metrics["macro_f1"] > state.best_macro_f1:
            state.best_macro_f1 = float(val_metrics["macro_f1"])
            save_checkpoint(
                best_macro_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                config={"backbone": backbone_cfg, "experiment": experiment_cfg},
                ema=ema,
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
                config={"backbone": backbone_cfg, "experiment": experiment_cfg},
                ema=ema,
                model_scope="trainable",
                save_optimizer=False,
                save_rng=False,
            )
            improved = True
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config={"backbone": backbone_cfg, "experiment": experiment_cfg},
            ema=ema,
            model_scope="trainable",
        )

        bad_epochs = 0 if improved else bad_epochs + 1
        if patience > 0 and bad_epochs >= patience:
            break

    if best_macro_path.exists():
        load_checkpoint(best_macro_path, model=model, ema=ema, device=device)
    final_metrics, val_probs, val_pred, final_eval_payload = evaluate_model_for_selection(
        model,
        val_loader,
        labels[val_idx],
        class_names,
        device=device,
        amp_dtype=amp_dtype,
        ema=ema,
        max_batches=max_eval_batches,
    )
    test_probs = (
        predict_unlabeled_model_for_selection(
            model,
            test_loader,
            device=device,
            amp_dtype=amp_dtype,
            ema=ema,
            max_batches=max_eval_batches,
        )
        if test_loader is not None
        else None
    )
    fold_summary = {
        "fold": fold_idx,
        "backbone": backbone_key,
        "experiment": experiment_name,
        "mode": mode,
        "val_macro_f1": final_metrics["macro_f1"],
        "val_balanced_accuracy": final_metrics["balanced_accuracy"],
        "val_selection_score": final_metrics["selection_score"],
        "best_macro_f1": state.best_macro_f1,
        "best_balanced_accuracy": state.best_balanced_accuracy,
        "parameter_counts": parameter_counts,
        "module_audit": audit_paths,
        "training_controls": training_controls_summary(
            ema=ema,
            max_grad_norm=max_grad_norm,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
        "eval": final_eval_payload,
        "checkpoints": {
            "last": project_relative(last_path),
            "best_macro_f1": project_relative(best_macro_path),
            "best_balanced_accuracy": project_relative(best_balanced_path),
            "ema": project_relative(ema_path) if ema_path.exists() else None,
        },
    }
    write_json(
        {"summary": fold_summary, "history": state.history, "final_metrics": final_metrics},
        fold_metrics_dir / f"fold_{fold_idx}.json",
    )
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path.write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "summary": fold_summary,
        "val_probs": val_probs,
        "val_pred": val_pred,
        "test_probs": test_probs,
    }


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    mixup: dict[str, Any],
    ema: ModelEma | None,
    max_grad_norm: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Train one epoch."""

    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_idx, (images, labels, _filenames) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, mixup_target = maybe_mixup(
            images,
            labels,
            alpha=float(mixup.get("alpha", 0.0)),
            probability=float(mixup.get("probability", 0.0)),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = mixed_cross_entropy(criterion, logits, labels, mixup_target)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_count += images.size(0)
    return {"loss": total_loss / max(total_count, 1)}


def training_controls_summary(
    *,
    ema: ModelEma | None,
    max_grad_norm: float,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    """Return training controls recorded in fold summaries."""

    return {
        "ema_enabled": ema is not None,
        "ema_decay": ema.decay if ema is not None else 0.0,
        "max_grad_norm": max_grad_norm,
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "checkpoint_model_scope": "trainable",
    }


def evaluate_model_for_selection(
    model: torch.nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    class_names: list[str],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    ema: ModelEma | None,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate raw weights and, when enabled, use EMA for selection."""

    raw_metrics, raw_probs, raw_pred = evaluate_model(
        model,
        loader,
        labels,
        class_names,
        device=device,
        amp_dtype=amp_dtype,
        max_batches=max_batches,
    )
    if ema is None:
        return raw_metrics, raw_probs, raw_pred, {"selected": "raw", "raw": raw_metrics}
    with ema.apply_to(model):
        ema_metrics, ema_probs, ema_pred = evaluate_model(
            model,
            loader,
            labels,
            class_names,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=max_batches,
        )
    return ema_metrics, ema_probs, ema_pred, {"selected": "ema", "raw": raw_metrics, "ema": ema_metrics}


def predict_unlabeled_model_for_selection(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    ema: ModelEma | None,
    max_batches: int | None = None,
) -> np.ndarray:
    """Predict with EMA weights when EMA is enabled."""

    if ema is None:
        return predict_unlabeled_model(model, loader, device=device, amp_dtype=amp_dtype, max_batches=max_batches)
    with ema.apply_to(model):
        return predict_unlabeled_model(model, loader, device=device, amp_dtype=amp_dtype, max_batches=max_batches)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    """Append one training event as JSONL for live tailing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    class_names: list[str],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate a model and return metrics/probabilities/predictions."""

    model.eval()
    probs = []
    seen_labels = []
    with torch.no_grad():
        for batch_idx, (images, batch_labels, _filenames) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
            seen_labels.append(batch_labels.detach().cpu().numpy())
    probs_arr = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(class_names)))
    labels_arr = np.concatenate(seen_labels, axis=0) if seen_labels else labels[:0]
    pred = probs_arr.argmax(axis=1)
    return compute_classification_metrics(labels_arr, pred, class_names), probs_arr, pred


def predict_unlabeled_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None = None,
) -> np.ndarray:
    """Predict probabilities for an unlabeled image loader."""

    model.eval()
    probs = []
    with torch.no_grad():
        for batch_idx, (images, _filenames) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float64)


def inject_lora(model: torch.nn.Module, experiment_cfg: dict[str, Any], targets: list[str]) -> torch.nn.Module:
    """Inject LoRA adapters using Hugging Face PEFT."""

    if not targets:
        raise ValueError("LoRA experiment selected no target modules")
    assert_lora_targets_valid(model, targets)
    from peft import LoraConfig, get_peft_model

    lora_cfg = experiment_cfg.get("lora", {})
    kwargs = {
        "r": int(lora_cfg.get("r", 8)),
        "lora_alpha": int(lora_cfg.get("alpha", 16)),
        "target_modules": targets,
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
        "bias": str(lora_cfg.get("bias", "none")),
        "modules_to_save": list(lora_cfg.get("modules_to_save", ["classifier"])),
    }
    signature = inspect.signature(LoraConfig)
    for cfg_key, peft_key in (("use_rslora", "use_rslora"), ("use_dora", "use_dora")):
        if cfg_key not in lora_cfg:
            continue
        if peft_key not in signature.parameters:
            raise ValueError(f"installed PEFT LoraConfig does not support {peft_key}")
        kwargs[peft_key] = bool(lora_cfg[cfg_key])
    config = LoraConfig(**kwargs)
    return get_peft_model(model, config)


def merge_transform_config(
    augmentation: dict[str, Any],
    backbone_cfg: dict[str, Any],
    *,
    pretrained_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge global augmentation with backbone-specific preprocessing.

    Global spatial/color augmentation stays shared across experiments, while
    model-card preprocessing such as mean/std can be overridden per backbone.
    """

    merged = dict(augmentation or {})
    if pretrained_cfg:
        for key in ("mean", "std", "crop_pct", "interpolation"):
            if key in pretrained_cfg and pretrained_cfg[key] is not None:
                merged[key] = pretrained_cfg[key]
    preprocessing = dict(backbone_cfg.get("preprocessing", {}))
    for key in ("mean", "std", "crop_pct", "interpolation", "scale_to", "scale_pad_fill_rgb"):
        if key in preprocessing:
            merged[key] = preprocessing[key]
    for key in ("scale_to", "scale_pad_fill_rgb"):
        if key in backbone_cfg:
            merged[key] = backbone_cfg[key]
    return merged


def merge_nested_dict(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Merge experiment overrides into a shallow augmentation config."""

    merged = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def resolve_class_weight_tensor(
    global_config: dict[str, Any],
    experiment_cfg: dict[str, Any],
    *,
    class_names: list[str],
) -> torch.Tensor | None:
    """Resolve opt-in manual class weights for supervised PEFT training."""

    cfg = merge_nested_dict(global_config.get("class_weighting", {}), experiment_cfg.get("class_weighting", {}))
    if not cfg:
        return None
    mode = str(cfg.get("mode", "manual")).lower()
    if mode in {"none", "off", "false"}:
        return None
    if mode != "manual":
        raise ValueError(f"unsupported class_weighting mode for PEFT training: {mode}")
    raw_weights = cfg.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("class_weighting.weights must be a mapping from class name to weight")
    values = []
    for name in class_names:
        if name not in raw_weights:
            raise ValueError(f"class_weighting.weights missing class: {name}")
        values.append(float(raw_weights[name]))
    weights = torch.tensor(values, dtype=torch.float32)
    if bool(cfg.get("normalize_mean", True)):
        weights = weights / torch.clamp(weights.mean(), min=1e-12)
    return weights


def build_model_transform(
    config: dict[str, Any],
    *,
    train: bool,
    input_size: int,
    official_preprocess,
):
    """Build either project torchvision preprocessing or an official model preprocess."""

    if official_preprocess is not None:
        transform_config = {**dict(config or {}), "input_size": input_size}
        return build_official_preprocess_transform(transform_config, train=train, official_preprocess=official_preprocess)
    return build_transform(config, train=train, input_size=input_size)


def resolve_official_preprocess(model: torch.nn.Module):
    """Find official preprocessing attached to a backbone, including PEFT wrappers."""

    queue: list[Any] = [model]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        preprocess = getattr(current, "official_preprocess", None)
        if preprocess is not None:
            return preprocess
        for attribute in ("backbone", "base_model", "model", "module"):
            child = getattr(current, attribute, None)
            if child is not None:
                queue.append(child)
        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except (AttributeError, TypeError):
                pass
    return None


def optional_positive_int(value: Any) -> int | None:
    """Parse optional positive integer config values."""

    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("batch limit values must be positive when set")
    return parsed


def resolve_test_prediction_mode(config: dict[str, Any], *, smoke: bool) -> str:
    """Validate how Scheme02 CV handles unlabeled test predictions."""

    mode = str(config.get("test_prediction_mode", "none"))
    allowed = {"none", "single_refit", "cv_mean_debug"}
    if mode not in allowed:
        raise ValueError(f"unsupported test_prediction_mode: {mode}")
    if mode == "cv_mean_debug" and not (smoke or bool(config.get("debug", False))):
        raise ValueError("test_prediction_mode=cv_mean_debug is only allowed for smoke/debug runs")
    return mode


def resolve_pretrained_cfg(model: torch.nn.Module) -> dict[str, Any] | None:
    """Find the timm pretrained config through classifier and PEFT wrappers."""

    queue: list[Any] = [model]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        pretrained_cfg = getattr(current, "pretrained_cfg", None)
        if isinstance(pretrained_cfg, dict) and pretrained_cfg:
            return dict(pretrained_cfg)
        for attribute in ("backbone", "base_model", "model", "module"):
            child = getattr(current, attribute, None)
            if child is not None:
                queue.append(child)
        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except (AttributeError, TypeError):
                pass
    return None


def build_optimizer(
    model: torch.nn.Module,
    experiment_cfg: dict[str, Any],
    global_config: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build AdamW with separate head and adapter learning rates."""

    head_lr = float(experiment_cfg.get("head_lr", global_config.get("head_lr", 1e-3)))
    adapter_lr = float(experiment_cfg.get("adapter_lr", global_config.get("adapter_lr", 5e-5)))
    weight_decay = float(experiment_cfg.get("weight_decay", global_config.get("weight_decay", 0.01)))
    head_params = []
    adapter_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "classifier" in name or "modules_to_save" in name:
            head_params.append(parameter)
        elif "lora_" in name:
            adapter_params.append(parameter)
        else:
            other_params.append(parameter)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": adapter_lr, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": adapter_lr, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("no trainable parameters found")
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with 5% warmup."""

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve training device from config."""

    requested = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def resolve_amp_dtype(config: dict[str, Any]) -> torch.dtype | None:
    """Resolve autocast dtype; bf16/fp16 only enabled on CUDA."""

    precision = str(config.get("precision", "fp32")).lower()
    if not torch.cuda.is_available():
        return None
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return None


def resolve_resume_path(resume: str | Path | None, last_path: Path) -> Path | None:
    """Resolve explicit or auto resume path."""

    if resume:
        return resolve_project_path(resume)
    if last_path.exists():
        return last_path
    return None


def load_selected_backbones(path: str | Path | None) -> set[str]:
    """Load selected backbone keys from a text file, if present."""

    if path is None:
        return set()
    resolved = resolve_project_path(path)
    if resolved is None or not resolved.exists():
        return set()
    keys = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        keys.append(value)
    return set(keys)


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compact a result row for summary tables."""

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


def write_selection_report(summary: dict[str, Any], out_path: Path) -> None:
    """Write a markdown report for fast experiment inspection."""

    rows = summary.get("results", [])
    lines = [
        "# Scheme 02 PEFT CV Selection Report",
        "",
        f"- run_name: `{summary.get('run_name')}`",
        f"- evaluated_folds: `{summary.get('evaluated_folds')}`",
        "",
        "| rank | experiment_id | mode | macro_f1 | balanced_acc | selection |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | `{row['experiment_id']}` | `{row['mode']}` | "
            f"{row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | "
            f"{row['selection_score']:.6f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_best_checkpoint(src: Path, dst: Path) -> None:
    """Copy a checkpoint if it exists."""

    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
