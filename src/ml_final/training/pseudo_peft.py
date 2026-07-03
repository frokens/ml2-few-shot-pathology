"""S03-6 pseudo-label LoRA training."""

from __future__ import annotations

import csv
import datetime as dt
import json
import statistics
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
    merge_model_overrides,
)
from ml_final.backbones.module_audit import audit_modules, write_module_audit
from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.training.checkpointing import CheckpointState, ModelEma, load_checkpoint, save_checkpoint
from ml_final.training.data import ManifestImageDataset, build_transform, read_manifest
from ml_final.training.losses import mixed_hard_soft_focal_loss, mixed_hard_soft_loss
from ml_final.training.peft_train import (
    append_event,
    build_optimizer,
    build_scheduler,
    evaluate_model_for_selection,
    inject_lora,
    merge_transform_config,
    optional_positive_int,
    predict_unlabeled_model_for_selection,
    resolve_amp_dtype,
    resolve_class_weight_tensor,
    resolve_device,
    resolve_pretrained_cfg,
    resolve_resume_path,
    training_controls_summary,
)
from ml_final.training.pseudo_dataset import (
    PseudoImageDataset,
    UnlabeledManifestDataset,
    load_true_and_pseudo_rows,
)
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


def run_pseudo_lora(
    config_path: str | Path,
    *,
    selected_peft: str | Path,
    pseudolabels: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    run_name: str | None = None,
    resume: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run pseudo-label LoRA CV and averaged test inference."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(
        {
            "config": config,
            "selected_peft": selected_peft,
            "pseudolabels": pseudolabels,
            "train_manifest": train_manifest,
            "test_manifest": test_manifest,
        },
        context="Scheme03 pseudo LoRA training",
    )
    run_name = run_name or str(config.get("run_name", "scheme03_pseudo_lora"))
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    run_dir = resolve_project_path(f"runs/scheme_03/{run_name}")
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

    lambda_pseudo = float(config.get("lambda_pseudo", 0.05))
    payload = load_true_and_pseudo_rows(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        pseudolabels=pseudolabels,
        lambda_pseudo=lambda_pseudo,
        min_pseudo_sample_weight=float(config.get("min_pseudo_sample_weight", 0.0)),
        max_pseudo_per_class=config.get("max_pseudo_per_class"),
        pseudo_class_weight=config.get("pseudo_class_weight"),
    )
    train_rows = payload["train_rows"]
    labels = np.asarray([payload["label_to_idx"][row["label"]] for row in train_rows], dtype=np.int64)
    class_names = payload["class_names"]
    test_rows = read_manifest(resolve_project_path(test_manifest))

    backbone_cfg, experiment_cfg = resolve_selected_peft_config(config, selected_peft=selected_peft, smoke=smoke)
    n_splits = int(config.get("n_splits", 5 if not smoke else 2))
    max_folds = config.get("max_folds")
    if smoke:
        max_folds = int(config.get("smoke_max_folds", 2))
    split_items = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    if max_folds is not None:
        split_items = split_items[: int(max_folds)]

    oof_probs = np.zeros((len(train_rows), len(class_names)), dtype=np.float64)
    oof_counts = np.zeros(len(train_rows), dtype=np.float64)
    test_probs_accum = []
    fold_summaries = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_items):
        fold = train_pseudo_lora_fold(
            train_rows=train_rows,
            test_rows=test_rows,
            pseudo_rows=payload["pseudo_rows"],
            labels=labels,
            class_names=class_names,
            label_to_idx=payload["label_to_idx"],
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
        oof_probs[val_idx] += fold["val_probs"]
        oof_counts[val_idx] += 1.0
        if fold["test_probs"] is not None:
            test_probs_accum.append(fold["test_probs"])
        fold_summaries.append(fold["summary"])

    oof_probs = oof_probs / np.maximum(oof_counts[:, None], 1.0)
    y_pred = oof_probs.argmax(axis=1)
    metrics = compute_classification_metrics(labels, y_pred, class_names)
    experiment_id = f"{backbone_cfg['key']}__{experiment_cfg['name']}__pseudo_lora"
    oof_path = predictions_dir / f"oof_{experiment_id}.npz"
    np.savez_compressed(
        oof_path,
        probs=oof_probs,
        y_true=labels,
        y_pred=y_pred,
        class_names=np.asarray(class_names, dtype=object),
        train_filenames=np.asarray([row["filename"] for row in train_rows], dtype=object),
        experiment_id=np.asarray(experiment_id, dtype=object),
    )
    test_path = None
    submission_path = None
    if test_probs_accum:
        test_probs = np.mean(test_probs_accum, axis=0)
        test_path = predictions_dir / f"test_{experiment_id}.npz"
        test_filenames = np.asarray([row["filename"] for row in test_rows], dtype=object)
        np.savez_compressed(
            test_path,
            probs=test_probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=test_filenames,
            experiment_id=np.asarray(experiment_id, dtype=object),
        )
        submission_path = write_submission(test_probs, test_filenames, class_names, run_name)

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "selected_peft": project_relative(resolve_project_path(selected_peft) or selected_peft),
        "pseudolabels": project_relative(resolve_project_path(pseudolabels) or pseudolabels),
        "lambda_pseudo": lambda_pseudo,
        "num_true": int(len(train_rows)),
        "num_pseudo": int(len(payload["pseudo_rows"])),
        "pseudo_counts": payload["pseudo_counts"],
        "pseudo_class_weight": payload["pseudo_class_weight"],
        "pseudo_effective_weight_sum_by_class": payload["pseudo_effective_weight_sum_by_class"],
        "pseudo_effective_weight_sum": payload["pseudo_effective_weight_sum"],
        "class_names": class_names,
        "experiment_id": experiment_id,
        "metrics": metrics,
        "folds": fold_summaries,
        "oof_path": project_relative(oof_path),
        "test_path": project_relative(test_path) if test_path else None,
        "submission": submission_path,
    }
    write_json(summary, summary_path)
    write_pseudo_lora_report(metrics_dir / "pseudo_lora_report.md", summary)
    return {"run_dir": project_relative(run_dir), "summary": summary}


def run_pseudo_lora_single_refit(
    config_path: str | Path,
    *,
    selected_peft: str | Path,
    pseudolabels: str | Path,
    train_manifest: str | Path,
    test_manifest: str | Path,
    run_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one pseudo-label LoRA model on all labeled rows and predict test once."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(
        {
            "config": config,
            "selected_peft": selected_peft,
            "pseudolabels": pseudolabels,
            "train_manifest": train_manifest,
            "test_manifest": test_manifest,
        },
        context="Scheme03 pseudo LoRA single refit",
    )
    run_name = run_name or str(config.get("run_name", "scheme03_pseudo_lora_single_refit"))
    seed = int(config.get("seed", 2026))
    set_seed(seed)

    run_dir = resolve_project_path(f"runs/scheme_03/{run_name}")
    if run_dir is None:
        raise ValueError("run_dir cannot be None")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    module_audit_dir = run_dir / "module_audit"
    checkpoints_root = run_dir / "checkpoints"
    for directory in (metrics_dir, predictions_dir, module_audit_dir, checkpoints_root):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "summary.json"
    if bool(config.get("skip_completed", False)) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {"run_dir": project_relative(run_dir), "summary": summary, "skipped": True}

    lambda_pseudo = float(config.get("lambda_pseudo", 0.05))
    payload = load_true_and_pseudo_rows(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        pseudolabels=pseudolabels,
        lambda_pseudo=lambda_pseudo,
        min_pseudo_sample_weight=float(config.get("min_pseudo_sample_weight", 0.0)),
        max_pseudo_per_class=config.get("max_pseudo_per_class"),
        pseudo_class_weight=config.get("pseudo_class_weight"),
    )
    train_rows = payload["train_rows"]
    class_names = payload["class_names"]
    test_rows = read_manifest(resolve_project_path(test_manifest))
    backbone_cfg, experiment_cfg = resolve_selected_peft_config(config, selected_peft=selected_peft, smoke=smoke)
    experiment_id = f"{backbone_cfg['key']}__{experiment_cfg['name']}__pseudo_lora"

    device = resolve_device(config)
    model_cfg = merge_model_overrides(backbone_cfg, experiment_cfg)
    model, input_size = build_classifier(model_cfg, num_classes=len(class_names))
    model.to(device)
    if bool(experiment_cfg.get("gradient_checkpointing", config.get("gradient_checkpointing", False))):
        enable_gradient_checkpointing(model)
    audit = audit_modules(
        model,
        target_policy=str(experiment_cfg.get("target_policy", "attention_qkv")),
        explicit_targets=experiment_cfg.get("target_modules"),
    )
    audit_paths = write_module_audit(audit, module_audit_dir, model_key=str(backbone_cfg["key"]))
    model = inject_lora(model, experiment_cfg, audit.selected_lora_targets)
    model.to(device)
    freeze_all_except_trainable_adapters_and_head(model)
    parameter_counts = count_parameters(model)

    transform_config = merge_transform_config(
        config.get("augmentation", {}),
        backbone_cfg,
        pretrained_cfg=resolve_pretrained_cfg(model),
    )
    train_transform = build_transform(transform_config, train=True, input_size=input_size)
    test_transform = build_transform(transform_config, train=False, input_size=input_size)
    train_dataset = PseudoImageDataset(
        train_rows,
        payload["pseudo_rows"],
        class_names=class_names,
        label_to_idx=payload["label_to_idx"],
        transform=train_transform,
        n_classes=len(class_names),
    )
    test_dataset = UnlabeledManifestDataset(test_rows, transform=test_transform) if test_rows else None
    batch_size = int(experiment_cfg.get("batch_size", config.get("batch_size", 8)))
    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"))
        if test_dataset is not None
        else None
    )

    optimizer = build_optimizer(model, experiment_cfg, config)
    configured_epochs = int(experiment_cfg.get("epochs", config.get("epochs", 50)))
    epochs, epoch_source = resolve_single_refit_epochs(config, configured_epochs=configured_epochs)
    scheduler = build_scheduler(optimizer, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    class_weight = resolve_class_weight_tensor(config, experiment_cfg, class_names=class_names)
    focal_loss = resolve_focal_loss(experiment_cfg, config)
    ema = (
        ModelEma(model, decay=float(experiment_cfg.get("ema_decay", config.get("ema_decay", 0.0))))
        if float(experiment_cfg.get("ema_decay", config.get("ema_decay", 0.0))) > 0
        else None
    )
    max_grad_norm = float(experiment_cfg.get("max_grad_norm", config.get("max_grad_norm", 1.0)))
    amp_dtype = resolve_amp_dtype(config)
    max_eval_batches = optional_positive_int(experiment_cfg.get("max_eval_batches", config.get("max_eval_batches")))
    event_dir = metrics_dir / experiment_id
    events_path = event_dir / "single_refit_events.jsonl"
    history = []
    for epoch in range(epochs):
        item = {
            "epoch": epoch + 1,
            "train": train_pseudo_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device=device,
                amp_dtype=amp_dtype,
                label_smoothing=float(experiment_cfg.get("label_smoothing", config.get("label_smoothing", 0.0))),
                class_weight=class_weight,
                focal_loss=focal_loss,
                ema=ema,
                max_grad_norm=max_grad_norm,
            ),
        }
        history.append(item)
        append_event(events_path, item)

    state = CheckpointState(epoch=epochs, best_macro_f1=-1.0, best_balanced_accuracy=-1.0, history=history)
    checkpoint_dir = checkpoints_root / experiment_id / "single_refit"
    ckpt_path = checkpoint_dir / "single_refit.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config={"backbone": backbone_cfg, "experiment": experiment_cfg},
        ema=ema,
        model_scope="trainable",
    )

    test_path = None
    submission_path = None
    if test_loader is not None:
        test_probs = predict_unlabeled_model_for_selection(
            model,
            test_loader,
            device=device,
            amp_dtype=amp_dtype,
            ema=ema,
            max_batches=max_eval_batches,
        )
        test_filenames = np.asarray([row["filename"] for row in test_rows[: len(test_probs)]], dtype=object)
        test_path = predictions_dir / f"test_{experiment_id}__single_refit.npz"
        np.savez_compressed(
            test_path,
            probs=test_probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=test_filenames,
            experiment_id=np.asarray(experiment_id, dtype=object),
            prediction_kind=np.asarray("single_refit", dtype=object),
            model_count=np.asarray(1, dtype=np.int64),
        )
        submission_path = write_submission(test_probs, test_filenames, class_names, run_name)

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "mode": "pseudo_lora_single_refit",
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "selected_peft": project_relative(resolve_project_path(selected_peft) or selected_peft),
        "pseudolabels": project_relative(resolve_project_path(pseudolabels) or pseudolabels),
        "lambda_pseudo": lambda_pseudo,
        "num_true": int(len(train_rows)),
        "num_pseudo": int(len(payload["pseudo_rows"])),
        "pseudo_counts": payload["pseudo_counts"],
        "pseudo_class_weight": payload["pseudo_class_weight"],
        "pseudo_effective_weight_sum_by_class": payload["pseudo_effective_weight_sum_by_class"],
        "pseudo_effective_weight_sum": payload["pseudo_effective_weight_sum"],
        "class_names": class_names,
        "experiment_id": experiment_id,
        "epochs": epochs,
        "configured_epochs": configured_epochs,
        "epoch_source": epoch_source,
        "history": history,
        "module_audit": audit_paths,
        "parameter_counts": parameter_counts,
        "training_controls": training_controls_summary(
            ema=ema,
            max_grad_norm=max_grad_norm,
            max_eval_batches=max_eval_batches,
        ),
        "checkpoints": {"single_refit": project_relative(ckpt_path)},
        "test_path": project_relative(test_path) if test_path else None,
        "test_prediction_kind": "single_refit" if test_path else None,
        "model_count": 1 if test_path else None,
        "submission": submission_path,
    }
    write_json(summary, summary_path)
    write_pseudo_lora_single_refit_report(metrics_dir / "pseudo_lora_single_refit_report.md", summary)
    return {"run_dir": project_relative(run_dir), "summary": summary}


def resolve_single_refit_epochs(config: dict[str, Any], *, configured_epochs: int) -> tuple[int, dict[str, Any]]:
    """Resolve optional final-refit epoch count from completed CV fold histories."""

    source = config.get("refit_epoch_source")
    if not source:
        return configured_epochs, {"mode": "configured", "configured_epochs": configured_epochs}
    if isinstance(source, dict):
        raw_path = source.get("summary") or source.get("path") or source.get("run_dir")
        metric_name = str(source.get("metric", "macro_f1"))
    else:
        raw_path = source
        metric_name = "macro_f1"
    if not raw_path:
        return configured_epochs, {"mode": "configured", "configured_epochs": configured_epochs, "warning": "empty refit_epoch_source"}
    source_path = resolve_project_path(raw_path) or Path(raw_path)
    event_files = find_refit_epoch_event_files(source_path)
    best_epochs: list[int] = []
    for event_file in event_files:
        best_epoch = best_epoch_from_events(event_file, metric_name=metric_name)
        if best_epoch is not None:
            best_epochs.append(best_epoch)
    if not best_epochs:
        return (
            configured_epochs,
            {
                "mode": "configured",
                "configured_epochs": configured_epochs,
                "warning": f"no usable fold events under {project_relative(source_path)}",
            },
        )
    median_epoch = int(round(float(statistics.median(best_epochs))))
    median_epoch = max(1, median_epoch)
    return (
        median_epoch,
        {
            "mode": "cv_best_epoch_median",
            "configured_epochs": configured_epochs,
            "resolved_epochs": median_epoch,
            "metric": metric_name,
            "source": project_relative(source_path),
            "fold_best_epochs": best_epochs,
        },
    )


def find_refit_epoch_event_files(source_path: Path) -> list[Path]:
    """Find fold event histories for an optional final-refit epoch source."""

    if source_path.is_file() or source_path.name == "summary.json":
        search_root = source_path.parent
    else:
        search_root = source_path / "metrics" if (source_path / "metrics").exists() else source_path
    return sorted(path for path in search_root.glob("**/fold_*_events.jsonl") if path.is_file())


def best_epoch_from_events(event_file: Path, *, metric_name: str) -> int | None:
    """Return the epoch with the best validation metric in one fold event file."""

    best_epoch = None
    best_value = None
    with event_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            value = item.get("val", {}).get(metric_name)
            if value is None:
                continue
            value_f = float(value)
            if best_value is None or value_f > best_value:
                best_value = value_f
                best_epoch = int(item["epoch"])
    return best_epoch


def train_pseudo_lora_fold(
    *,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    pseudo_rows: list[dict[str, Any]],
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
    """Train one pseudo-LoRA fold."""

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
    audit_paths = write_module_audit(audit, module_audit_dir, model_key=str(backbone_cfg["key"]))
    model = inject_lora(model, experiment_cfg, audit.selected_lora_targets)
    model.to(device)
    freeze_all_except_trainable_adapters_and_head(model)
    parameter_counts = count_parameters(model)
    transform_config = merge_transform_config(
        global_config.get("augmentation", {}),
        backbone_cfg,
        pretrained_cfg=resolve_pretrained_cfg(model),
    )
    train_transform = build_transform(transform_config, train=True, input_size=input_size)
    val_transform = build_transform(transform_config, train=False, input_size=input_size)
    fold_true_rows = [train_rows[int(idx)] for idx in train_idx]
    train_dataset = PseudoImageDataset(
        fold_true_rows,
        pseudo_rows,
        class_names=class_names,
        label_to_idx=label_to_idx,
        transform=train_transform,
        n_classes=len(class_names),
    )
    val_dataset = ManifestImageDataset(train_rows, label_to_idx=label_to_idx, transform=val_transform)
    test_dataset = UnlabeledManifestDataset(test_rows, transform=val_transform) if test_rows else None
    batch_size = int(experiment_cfg.get("batch_size", global_config.get("batch_size", 8)))
    workers = int(global_config.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(Subset(val_dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"))
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=(device.type == "cuda"))
        if test_dataset is not None
        else None
    )
    optimizer = build_optimizer(model, experiment_cfg, global_config)
    epochs = int(experiment_cfg.get("epochs", global_config.get("epochs", 50)))
    scheduler = build_scheduler(optimizer, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    class_weight = resolve_class_weight_tensor(global_config, experiment_cfg, class_names=class_names)
    focal_loss = resolve_focal_loss(experiment_cfg, global_config)
    ema = (
        ModelEma(model, decay=float(experiment_cfg.get("ema_decay", global_config.get("ema_decay", 0.0))))
        if float(experiment_cfg.get("ema_decay", global_config.get("ema_decay", 0.0))) > 0
        else None
    )
    max_grad_norm = float(experiment_cfg.get("max_grad_norm", global_config.get("max_grad_norm", 1.0)))
    amp_dtype = resolve_amp_dtype(global_config)
    fold_ckpt_dir = checkpoints_root / f"{backbone_cfg['key']}__{experiment_cfg['name']}" / f"fold_{fold_idx}"
    last_path = fold_ckpt_dir / "last.pt"
    best_macro_path = fold_ckpt_dir / "best_macro_f1.pt"
    complete_path = fold_ckpt_dir / "COMPLETE"
    fold_metrics_dir = run_dir / "metrics" / f"{backbone_cfg['key']}__{experiment_cfg['name']}"
    events_path = fold_metrics_dir / f"fold_{fold_idx}_events.jsonl"
    state = CheckpointState(epoch=0, best_macro_f1=-1.0, best_balanced_accuracy=-1.0, history=[])
    resume_path = resolve_resume_path(resume, last_path)
    if resume_path is not None and resume_path.exists():
        state = load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, ema=ema, device=device)
    patience = int(experiment_cfg.get("early_stopping_patience", global_config.get("early_stopping_patience", 0)))
    bad_epochs = 0
    for epoch in range(state.epoch, epochs):
        train_metrics = train_pseudo_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            amp_dtype=amp_dtype,
            label_smoothing=float(experiment_cfg.get("label_smoothing", global_config.get("label_smoothing", 0.0))),
            class_weight=class_weight,
            focal_loss=focal_loss,
            ema=ema,
            max_grad_norm=max_grad_norm,
        )
        val_metrics, val_probs, _, eval_payload = evaluate_model_for_selection(
            model,
            val_loader,
            labels[val_idx],
            class_names,
            device=device,
            amp_dtype=amp_dtype,
            ema=ema,
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
        improved = val_metrics["macro_f1"] > state.best_macro_f1
        if improved:
            state.best_macro_f1 = float(val_metrics["macro_f1"])
            state.best_balanced_accuracy = max(state.best_balanced_accuracy, float(val_metrics["balanced_accuracy"]))
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
    )
    test_probs = (
        predict_unlabeled_model_for_selection(model, test_loader, device=device, amp_dtype=amp_dtype, ema=ema)
        if test_loader is not None
        else None
    )
    fold_summary = {
        "fold": fold_idx,
        "backbone": backbone_cfg["key"],
        "experiment": experiment_cfg["name"],
        "val_macro_f1": final_metrics["macro_f1"],
        "val_balanced_accuracy": final_metrics["balanced_accuracy"],
        "val_selection_score": final_metrics["selection_score"],
        "best_macro_f1": state.best_macro_f1,
        "parameter_counts": parameter_counts,
        "module_audit": audit_paths,
        "training_controls": training_controls_summary(ema=ema, max_grad_norm=max_grad_norm),
        "eval": final_eval_payload,
        "checkpoints": {"last": project_relative(last_path), "best_macro_f1": project_relative(best_macro_path)},
    }
    write_json({"summary": fold_summary, "history": state.history, "final_metrics": final_metrics}, fold_metrics_dir / f"fold_{fold_idx}.json")
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path.write_text(dt.datetime.now(dt.timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {"summary": fold_summary, "val_probs": val_probs, "val_pred": val_pred, "test_probs": test_probs}


def train_pseudo_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    label_smoothing: float,
    class_weight: torch.Tensor | None,
    focal_loss: dict[str, float] | None,
    ema: ModelEma | None,
    max_grad_norm: float,
) -> dict[str, float]:
    """Train one pseudo-LoRA epoch."""

    model.train()
    total_loss = 0.0
    total_count = 0
    total_pseudo = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        hard_labels = batch["hard_label"].to(device, non_blocking=True)
        soft_labels = batch["soft_label"].to(device, non_blocking=True)
        weights = batch["sample_weight"].to(device, non_blocking=True)
        is_pseudo = batch["is_pseudo"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            if focal_loss is not None:
                loss = mixed_hard_soft_focal_loss(
                    logits,
                    hard_labels,
                    soft_labels,
                    weights,
                    is_pseudo,
                    gamma=float(focal_loss["gamma"]),
                    label_smoothing=label_smoothing,
                    class_weight=class_weight,
                )
            else:
                loss = mixed_hard_soft_loss(
                    logits,
                    hard_labels,
                    soft_labels,
                    weights,
                    is_pseudo,
                    label_smoothing=label_smoothing,
                    class_weight=class_weight,
                )
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
        total_pseudo += int(is_pseudo.sum().detach().cpu())
    return {"loss": total_loss / max(total_count, 1), "pseudo_fraction": total_pseudo / max(total_count, 1)}


def resolve_focal_loss(experiment_cfg: dict[str, Any], global_config: dict[str, Any]) -> dict[str, float] | None:
    """Resolve optional focal loss for pseudo-LoRA training."""

    raw = dict(experiment_cfg.get("focal_loss", global_config.get("focal_loss", {})) or {})
    if not raw or not bool(raw.get("enabled", True)):
        return None
    gamma = float(raw.get("gamma", 0.0))
    if gamma <= 0:
        return None
    return {"gamma": gamma}


def evaluate_manifest_model(
    model: torch.nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    class_names: list[str],
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate a model on ManifestImageDataset batches."""

    model.eval()
    probs = []
    with torch.no_grad():
        for images, _labels, _filenames in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
    probs_arr = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(class_names)))
    pred = probs_arr.argmax(axis=1)
    return compute_classification_metrics(labels, pred, class_names), probs_arr, pred


def predict_unlabeled(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    """Predict probabilities for an unlabeled loader."""

    model.eval()
    probs = []
    with torch.no_grad():
        for images, _filenames in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images)
            probs.append(torch.softmax(logits, dim=1).detach().float().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float64)


def resolve_selected_peft_config(
    config: dict[str, Any],
    *,
    selected_peft: str | Path,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the one backbone/experiment pair S03-6 is allowed to use."""

    if smoke:
        return (
            dict(config.get("smoke_backbone", {"key": "tiny_cnn", "backend": "tiny_cnn"})),
            dict(
                config.get(
                    "smoke_experiment",
                    {
                        "name": "pseudo_lora_smoke",
                        "mode": "lora",
                        "epochs": 1,
                        "batch_size": 16,
                        "target_policy": "all_linear_except_head",
                        "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "modules_to_save": ["classifier"]},
                    },
                )
            ),
        )
    selection_payload = load_scheme02_best_payload(selected_peft)
    selected_id = selection_payload.get("experiment_id", "")
    backbone_key, experiment_name = parse_experiment_id(selected_id)
    source_config = config
    if bool(config.get("inherit_from_scheme02_selection", True)) and selection_payload.get("resolved_config"):
        source_config_path = resolve_project_path(selection_payload["resolved_config"])
        if source_config_path is not None and source_config_path.exists():
            source_config = load_yaml(source_config_path)
    backbones = list(source_config.get("backbones", []))
    experiments = list(source_config.get("experiments", []))
    backbone = next((item for item in backbones if str(item.get("key")) == backbone_key), None) if backbone_key else None
    experiment = next((item for item in experiments if str(item.get("name")) == experiment_name), None) if experiment_name else None
    if backbone is None or experiment is None:
        raise ValueError(
            "could not resolve selected PEFT backbone/experiment. "
            f"selected_id={selected_id!r}, resolved_config={selection_payload.get('resolved_config')!r}"
        )
    experiment = dict(experiment)
    experiment["name"] = f"{experiment['name']}_pseudo"
    return dict(backbone), experiment


def load_scheme02_best_payload(selected_peft: str | Path) -> dict[str, Any]:
    """Load Scheme02 best PEFT selection from JSON or text artifact."""

    selected_path = resolve_project_path(selected_peft)
    if selected_path is None:
        raise ValueError("selected_peft cannot be empty")
    json_path = selected_path if selected_path.suffix == ".json" else selected_path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    if selected_path.exists():
        selected_id = selected_path.read_text(encoding="utf-8").splitlines()[0].strip()
        return {"experiment_id": selected_id}
    raise FileNotFoundError(f"selected PEFT artifact not found: {selected_peft}")


def parse_experiment_id(value: str) -> tuple[str | None, str | None]:
    """Parse '<backbone>__<experiment>' IDs."""

    if "__" not in value:
        return None, None
    parts = value.split("__", 1)
    return parts[0], parts[1]


def write_submission(probs: np.ndarray, filenames: np.ndarray, class_names: list[str], run_name: str) -> str:
    """Write a submission CSV for pseudo-LoRA predictions."""

    out_dir = resolve_project_path("artifacts/submissions")
    if out_dir is None:
        raise ValueError("submission dir cannot be None")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"submission_{run_name}.csv"
    labels = [class_names[idx] for idx in probs.argmax(axis=1)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        writer.writerows(zip([str(item) for item in filenames.tolist()], labels))
    return project_relative(path)


def write_pseudo_lora_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a concise S03-6 report."""

    lines = [
        "# S03-6 Pseudo + LoRA Report",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Lambda pseudo: `{summary['lambda_pseudo']}`",
        f"- True rows: `{summary['num_true']}`",
        f"- Pseudo rows: `{summary['num_pseudo']}`",
        f"- OOF macro-F1: `{summary['metrics']['macro_f1']:.6f}`",
        f"- OOF balanced accuracy: `{summary['metrics']['balanced_accuracy']:.6f}`",
        f"- Submission: `{summary['submission']}`",
        "",
        "## Pseudo Counts",
        "",
    ]
    for label, count in summary["pseudo_counts"].items():
        lines.append(f"- `{label}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pseudo_lora_single_refit_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a concise report for one full-train pseudo-LoRA refit."""

    lines = [
        "# S03-6 Pseudo + LoRA Single-Refit Report",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Lambda pseudo: `{summary['lambda_pseudo']}`",
        f"- Epochs: `{summary['epochs']}`",
        f"- True rows: `{summary['num_true']}`",
        f"- Pseudo rows: `{summary['num_pseudo']}`",
        f"- Prediction kind: `{summary['test_prediction_kind']}`",
        f"- Model count: `{summary['model_count']}`",
        f"- Test prediction: `{summary['test_path']}`",
        f"- Submission: `{summary['submission']}`",
        "",
        "## Pseudo Counts",
        "",
    ]
    for label, count in summary["pseudo_counts"].items():
        lines.append(f"- `{label}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
