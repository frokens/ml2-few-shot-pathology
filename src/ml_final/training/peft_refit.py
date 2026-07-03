"""Single-model refit for Scheme02 candidates."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_final.backbones.factory import (
    build_classifier,
    count_parameters,
    enable_gradient_checkpointing,
    freeze_all_except_trainable_adapters_and_head,
    freeze_backbone,
    freeze_backbone_except_last_blocks,
    merge_model_overrides,
)
from ml_final.backbones.module_audit import audit_modules, write_module_audit
from ml_final.metrics.classification import write_json
from ml_final.training.checkpointing import CheckpointState, ModelEma, save_checkpoint
from ml_final.training.data import ManifestImageDataset, build_label_mapping, read_manifest
from ml_final.training.losses import cross_entropy_with_optional_smoothing
from ml_final.training.peft_train import (
    build_model_transform,
    build_optimizer,
    build_scheduler,
    inject_lora,
    merge_nested_dict,
    merge_transform_config,
    optional_positive_int,
    predict_unlabeled_model_for_selection,
    resolve_amp_dtype,
    resolve_class_weight_tensor,
    resolve_device,
    resolve_official_preprocess,
    resolve_pretrained_cfg,
    train_epoch,
    training_controls_summary,
)
from ml_final.training.pseudo_dataset import UnlabeledManifestDataset
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


def run_peft_single_refit(
    config_path: str | Path,
    *,
    experiment_id: str,
    run_name: str | None = None,
    source_summary: str | Path | None = None,
    oof_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train one Scheme02 model on all labels and predict test once."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme02 single refit")
    backbone_cfg, experiment_cfg = resolve_experiment(config, experiment_id)
    run_name = run_name or f"{experiment_id}__single_refit"
    seed = int(config.get("seed", 2026))
    set_seed(seed)

    run_dir = resolve_project_path(f"runs/scheme_02/{run_name}")
    if run_dir is None:
        raise ValueError("run_dir cannot be None")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    checkpoint_dir = run_dir / "checkpoints" / experiment_id / "single_refit"
    module_audit_dir = run_dir / "module_audit"
    for directory in (metrics_dir, predictions_dir, checkpoint_dir, module_audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_manifest = resolve_project_path(config["train_manifest"])
    test_manifest = resolve_project_path(config.get("test_manifest"))
    if train_manifest is None:
        raise ValueError("train_manifest is required for single_refit")
    rows = read_manifest(train_manifest)
    test_rows = read_manifest(test_manifest) if test_manifest is not None and test_manifest.exists() else []
    class_names, label_to_idx = build_label_mapping(rows)

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
    mode = str(experiment_cfg.get("mode", "head_only"))
    if mode == "head_only":
        freeze_backbone(model)
    elif mode in {"last_blocks", "last_block_ft"}:
        freeze_backbone_except_last_blocks(
            model,
            num_blocks=int(experiment_cfg.get("num_blocks", 1)),
            train_norm=bool(experiment_cfg.get("train_norm", True)),
        )
    elif mode == "lora":
        model = inject_lora(model, experiment_cfg, audit.selected_lora_targets)
        model.to(device)
        freeze_all_except_trainable_adapters_and_head(model)
    else:
        raise ValueError(f"unsupported Scheme02 refit mode: {mode}")

    transform_config = merge_transform_config(
        merge_nested_dict(config.get("augmentation", {}), experiment_cfg.get("augmentation", {})),
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
    test_transform = build_model_transform(
        transform_config,
        train=False,
        input_size=input_size,
        official_preprocess=official_preprocess,
    )
    batch_size = int(experiment_cfg.get("batch_size", config.get("batch_size", 16)))
    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        ManifestImageDataset(rows, label_to_idx=label_to_idx, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = (
        DataLoader(
            UnlabeledManifestDataset(test_rows, transform=test_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=(device.type == "cuda"),
        )
        if test_rows
        else None
    )

    optimizer = build_optimizer(model, experiment_cfg, config)
    epochs = int(experiment_cfg.get("epochs", config.get("epochs", 30)))
    scheduler = build_scheduler(optimizer, epochs=epochs, steps_per_epoch=max(1, len(train_loader)))
    class_weight = resolve_class_weight_tensor(config, experiment_cfg, class_names=class_names)
    criterion = cross_entropy_with_optional_smoothing(
        float(experiment_cfg.get("label_smoothing", config.get("label_smoothing", 0.0))),
        weight=class_weight.to(device) if class_weight is not None else None,
    )
    amp_dtype = resolve_amp_dtype(config)
    ema = (
        ModelEma(model, decay=float(experiment_cfg.get("ema_decay", config.get("ema_decay", 0.0))))
        if float(experiment_cfg.get("ema_decay", config.get("ema_decay", 0.0))) > 0
        else None
    )
    max_grad_norm = float(experiment_cfg.get("max_grad_norm", config.get("max_grad_norm", 1.0)))
    max_train_batches = optional_positive_int(
        experiment_cfg.get("max_train_batches", config.get("max_train_batches"))
    )
    max_eval_batches = optional_positive_int(
        experiment_cfg.get("max_eval_batches", config.get("max_eval_batches"))
    )
    history = []
    for epoch in range(epochs):
        item = {
            "epoch": epoch + 1,
            "train": train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device=device,
                amp_dtype=amp_dtype,
                mixup=experiment_cfg.get("mixup", config.get("mixup", {})),
                ema=ema,
                max_grad_norm=max_grad_norm,
                max_batches=max_train_batches,
            ),
        }
        history.append(item)

    state = CheckpointState(epoch=epochs, best_macro_f1=-1.0, best_balanced_accuracy=-1.0, history=history)
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
    if test_loader is not None:
        test_probs = predict_unlabeled_model_for_selection(
            model,
            test_loader,
            device=device,
            amp_dtype=amp_dtype,
            ema=ema,
            max_batches=max_eval_batches,
        )
        test_path = predictions_dir / f"test_{experiment_id}__single_refit.npz"
        np.savez_compressed(
            test_path,
            probs=test_probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=np.asarray([row["filename"] for row in test_rows[: len(test_probs)]], dtype=object),
            experiment_id=np.asarray(experiment_id, dtype=object),
            prediction_kind=np.asarray("single_refit", dtype=object),
            model_count=np.asarray(1, dtype=np.int64),
        )

    source_row = read_source_row(source_summary, experiment_id)
    resolved_oof = project_relative(resolve_project_path(oof_path)) if oof_path else source_row.get("oof_path")
    result_row = {
        "experiment_id": experiment_id,
        "backbone": str(backbone_cfg["key"]),
        "experiment": str(experiment_cfg["name"]),
        "mode": mode,
        "macro_f1": float(source_row.get("macro_f1", 0.0)),
        "balanced_accuracy": float(source_row.get("balanced_accuracy", 0.0)),
        "selection_score": float(source_row.get("selection_score", 0.0)),
        "oof_path": resolved_oof,
        "test_path": project_relative(test_path) if test_path else None,
        "test_prediction_kind": "single_refit" if test_path else None,
        "module_audit": audit_paths,
        "checkpoints": {"single_refit": project_relative(ckpt_path)},
        "training_controls": training_controls_summary(
            ema=ema,
            max_grad_norm=max_grad_norm,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
        ),
        "parameter_counts": count_parameters(model),
    }
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "train_manifest": project_relative(train_manifest),
        "test_manifest": project_relative(test_manifest) if test_manifest and test_manifest.exists() else None,
        "best": result_row,
        "results": [result_row],
        "test_prediction_files": [project_relative(test_path)] if test_path else [],
        "resolved_config": project_relative(metrics_dir / "resolved_config.json"),
    }
    (metrics_dir / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_json(summary, metrics_dir / "summary.json")
    return {"run_dir": project_relative(run_dir), "summary": summary}


def resolve_experiment(config: dict[str, Any], experiment_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a `backbone__experiment` identifier from a Scheme02 config."""

    if "__" not in experiment_id:
        raise ValueError("experiment_id must be formatted as backbone__experiment")
    backbone_key, experiment_name = experiment_id.split("__", 1)
    backbone = next((item for item in config.get("backbones", []) if str(item.get("key")) == backbone_key), None)
    experiment = next((item for item in config.get("experiments", []) if str(item.get("name")) == experiment_name), None)
    if backbone is None or experiment is None:
        raise ValueError(f"experiment_id not found in config: {experiment_id}")
    return dict(backbone), dict(experiment)


def read_source_row(source_summary: str | Path | None, experiment_id: str) -> dict[str, Any]:
    """Read CV metrics for the refit candidate when available."""

    if source_summary is None:
        return {}
    path = resolve_project_path(source_summary)
    if path is None or not path.exists():
        return {}
    summary = json.loads(path.read_text(encoding="utf-8"))
    for row in summary.get("results", []):
        if row.get("experiment_id") == experiment_id:
            return dict(row)
    return {}
