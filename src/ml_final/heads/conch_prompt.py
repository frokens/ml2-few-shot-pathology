"""CONCH image-text prompt probing for Scheme 01.

This module keeps CONCH prompt use separate from generic frozen-feature heads.
It uses the official CONCH image and text encoders when the optional
`conch` package is available, then writes prediction files compatible with
Scheme 01/03 OOF teacher discovery.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import ensure_dir, project_relative, project_root
from ml_final.weights.registry import load_lock


class ManifestImageOnlyDataset(Dataset):
    """Image-only dataset used by frozen CONCH prompt probing."""

    def __init__(self, rows: list[dict[str, str]], preprocess) -> None:
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = resolve_project_path(row["abs_path"])
        if image_path is None:
            raise ValueError("manifest row path cannot be empty")
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image), row["filename"]


def run_conch_prompt_probe(config_path: str | Path, *, run_name: str | None = None) -> dict[str, Any]:
    """Evaluate configured CONCH prompt families and write prediction files."""

    config = load_yaml(config_path)
    run_name = run_name or str(config.get("run_name", "conch_prompt_probe"))
    train_manifest = resolve_project_path(config["train_manifest"])
    test_manifest = resolve_project_path(config.get("test_manifest"))
    if train_manifest is None:
        raise ValueError("train_manifest cannot be None")
    train_rows = read_manifest(train_manifest)
    test_rows = read_manifest(test_manifest) if test_manifest and test_manifest.exists() else []

    class_names = sorted({row["label"] for row in train_rows})
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_true = np.asarray([label_to_idx[row["label"]] for row in train_rows], dtype=np.int64)
    train_filenames = np.asarray([row["filename"] for row in train_rows], dtype=object)
    test_filenames = np.asarray([row["filename"] for row in test_rows], dtype=object) if test_rows else None

    class_descriptions = normalize_class_descriptions(config.get("class_descriptions", {}), class_names)
    class_descriptions = apply_reference_mapping_override(class_descriptions, config)
    prompt_families = config.get("prompt_families", {})
    if not prompt_families:
        raise ValueError("prompt_families cannot be empty")

    run_dir = resolve_project_path(config.get("run_dir", f"runs/scheme_01/{run_name}"))
    if run_dir is None:
        raise ValueError("run_dir cannot be None")
    predictions_dir = ensure_dir(run_dir / "predictions")
    metrics_dir = ensure_dir(run_dir / "metrics")

    model, preprocess, zero_shot_classifier = load_conch_model(config)
    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model = model.to(device)
    model.eval()

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    image_features_train = extract_conch_image_features(
        model=model,
        preprocess=preprocess,
        rows=train_rows,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    image_features_test = (
        extract_conch_image_features(
            model=model,
            preprocess=preprocess,
            rows=test_rows,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if test_rows
        else None
    )

    logit_scale = float(config.get("logit_scale", resolve_logit_scale(model)))
    summaries = []
    oof_prediction_files = []
    test_prediction_files = []
    for family_name, family_cfg in prompt_families.items():
        prompts_by_class = render_prompt_family(family_cfg, class_names, class_descriptions)
        text_features = encode_prompt_family(
            model=model,
            zero_shot_classifier=zero_shot_classifier,
            prompts_by_class=prompts_by_class,
            device=device,
        )
        train_logits = logit_scale * image_features_train @ text_features.T
        train_probs = softmax_np(train_logits)
        y_pred = train_probs.argmax(axis=1)
        metrics = compute_classification_metrics(y_true, y_pred, class_names)
        selection_score = float(0.5 * metrics["macro_f1"] + 0.5 * metrics["balanced_accuracy"])
        head_id = f"conch_prompt_{family_name}"
        oof_path = predictions_dir / f"oof_{head_id}.npz"
        np.savez_compressed(
            oof_path,
            probs=train_probs.astype(np.float32),
            y_true=y_true,
            y_pred=y_pred,
            class_names=np.asarray(class_names, dtype=object),
            train_filenames=train_filenames,
            head_id=np.asarray(head_id, dtype=object),
            prompt_family=np.asarray(family_name, dtype=object),
            prompts_json=np.asarray(json.dumps(prompts_by_class, ensure_ascii=False), dtype=object),
            prompt_eval_type=np.asarray("fit_free_train_eval", dtype=object),
        )
        oof_prediction_files.append(project_relative(oof_path))

        test_path = None
        if image_features_test is not None and test_filenames is not None:
            test_logits = logit_scale * image_features_test @ text_features.T
            test_probs = softmax_np(test_logits)
            test_path = predictions_dir / f"test_{head_id}.npz"
            np.savez_compressed(
                test_path,
                probs=test_probs.astype(np.float32),
                class_names=np.asarray(class_names, dtype=object),
                test_filenames=test_filenames,
                head_id=np.asarray(head_id, dtype=object),
                prompt_family=np.asarray(family_name, dtype=object),
                prompts_json=np.asarray(json.dumps(prompts_by_class, ensure_ascii=False), dtype=object),
            )
            test_prediction_files.append(project_relative(test_path))

        summaries.append(
            {
                "family": family_name,
                "head_id": head_id,
                "selection_score": selection_score,
                "metrics": metrics,
                "oof_path": project_relative(oof_path),
                "test_path": project_relative(test_path) if test_path else None,
                "prompts_by_class": prompts_by_class,
            }
        )

    summaries.sort(key=lambda item: item["selection_score"], reverse=True)
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "train_manifest": project_relative(train_manifest),
        "test_manifest": project_relative(test_manifest) if test_rows else None,
        "class_names": class_names,
        "class_descriptions": class_descriptions,
        "reference_mapping_warnings": load_reference_mapping_warnings(config, class_names),
        "logit_scale": logit_scale,
        "prompt_families": summaries,
        "best_family": summaries[0]["family"],
        "best_oof_path": summaries[0]["oof_path"],
        "best_test_path": summaries[0]["test_path"],
        "oof_prediction_files": oof_prediction_files,
        "test_prediction_files": test_prediction_files,
        "results": [
            {
                "head_id": item["head_id"],
                "feature_file": "conch_prompt_text_encoder",
                "spec": {"family": "conch_prompt", "prompt_family": item["family"]},
                "macro_f1": item["metrics"]["macro_f1"],
                "balanced_accuracy": item["metrics"]["balanced_accuracy"],
                "selection_score": item["selection_score"],
                "oof_path": item["oof_path"],
                "test_path": item["test_path"],
            }
            for item in summaries
        ],
    }
    write_json(summary, metrics_dir / "conch_prompt_summary.json")
    write_json(summary, metrics_dir / "summary.json")
    write_prompt_report(summary, metrics_dir / "conch_prompt_report.md")
    write_best_prompt_selection(summary, config)
    return {"run_dir": project_relative(run_dir), "summary": summary}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV manifest."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_conch_model(config: dict[str, Any]):
    """Load CONCH through its official package."""

    try:
        from conch.downstream.zeroshot_path import zero_shot_classifier
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError(
            "CONCH prompt probing requires the official CONCH package. "
            "Install it from the pinned MahmoodLab/CONCH repository before running this command."
        ) from exc

    model_name = str(config.get("model_name", "conch_ViT-B-16"))
    checkpoint_path = resolve_conch_checkpoint(config)
    try:
        model, preprocess = create_model_from_pretrained(model_name, checkpoint_path=checkpoint_path)
    except TypeError:
        model, preprocess = create_model_from_pretrained(model_name, checkpoint_path)
    return model, preprocess, zero_shot_classifier


def resolve_conch_checkpoint(config: dict[str, Any]) -> str:
    """Resolve CONCH checkpoint path from config or model lockfile."""

    explicit = config.get("checkpoint_path")
    if explicit:
        path = resolve_project_path(explicit)
        if path is None:
            raise ValueError("checkpoint_path cannot be None")
        return str(path)

    lock_path = resolve_project_path(config.get("lock_path", "artifacts/model_registry/models.lock.yaml"))
    lock = load_lock(lock_path) if lock_path else None
    entry = (lock or {}).get("models", {}).get(str(config.get("lock_key", "conch")))
    if not entry:
        raise FileNotFoundError("CONCH checkpoint_path was not set and conch is absent from models.lock.yaml")
    local_path = resolve_project_path(entry["local_path"])
    if local_path is None:
        raise ValueError("CONCH lock local_path cannot be None")
    candidates = sorted(local_path.glob("pytorch_model*.bin")) + sorted(local_path.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no CONCH checkpoint file found under {local_path}")
    return str(candidates[0])


def extract_conch_image_features(
    *,
    model,
    preprocess,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Encode manifest images with CONCH image encoder."""

    dataset = ManifestImageOnlyDataset(rows, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features = []
    with torch.inference_mode():
        for images, _filenames in loader:
            images = images.to(device, non_blocking=True)
            encoded = model.encode_image(images, proj_contrast=True, normalize=True)
            encoded = F.normalize(encoded.float(), dim=-1)
            features.append(encoded.cpu().numpy())
    if not features:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(features, axis=0).astype(np.float32)


def encode_prompt_family(
    *,
    model,
    zero_shot_classifier,
    prompts_by_class: dict[str, list[str]],
    device: torch.device,
) -> np.ndarray:
    """Build CONCH text classifier with the official prompt-ensemble helper."""

    classifier = zero_shot_classifier(
        model=model,
        classnames=list(prompts_by_class.values()),
        templates=["CLASSNAME"],
        device=device,
    )
    return classifier.detach().cpu().float().numpy().T.astype(np.float32)


def normalize_class_descriptions(
    raw: dict[str, Any], class_names: list[str]
) -> dict[str, dict[str, str]]:
    """Validate and normalize class prompt metadata."""

    out: dict[str, dict[str, str]] = {}
    missing = [class_name for class_name in class_names if class_name not in raw]
    if missing:
        raise ValueError(f"class_descriptions missing entries for: {missing}")
    for class_name in class_names:
        item = raw[class_name]
        if isinstance(item, str):
            out[class_name] = {"cell_name": item, "description": item}
        else:
            cell_name = str(item["cell_name"])
            description = str(item.get("description", cell_name))
            out[class_name] = {"cell_name": cell_name, "description": description}
    return out


def apply_reference_mapping_override(
    class_descriptions: dict[str, dict[str, str]],
    config: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Override prompt names from pretrained reference-alignment output."""

    if not bool(config.get("use_reference_mapping", True)):
        return class_descriptions
    mapping_path = resolve_project_path(config.get("reference_mapping_path", "artifacts/selection/reference_class_mapping.yaml"))
    if mapping_path is None or not mapping_path.exists():
        if bool(config.get("require_reference_mapping", True)):
            raise FileNotFoundError(
                "CONCH prompt probe requires reference_class_mapping.yaml in formal mode. "
                "Run Stage 0b reference alignment first, or set use_reference_mapping=false for debug only."
            )
        return class_descriptions
    payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    mapping = payload.get("mapping", {})
    updated = {key: dict(value) for key, value in class_descriptions.items()}
    for class_name, item in mapping.items():
        if class_name not in updated:
            continue
        prompt_name = str(item.get("prompt_cell_name") or item.get("selected_reference_label") or "").strip()
        if not prompt_name:
            continue
        updated[class_name]["cell_name"] = prompt_name
        if "description" not in updated[class_name] or updated[class_name]["description"] == class_name:
            updated[class_name]["description"] = prompt_name
    return updated


def load_reference_mapping_warnings(config: dict[str, Any], class_names: list[str]) -> list[str]:
    """Return warnings for weak or missing class-name mapping evidence."""

    if not bool(config.get("use_reference_mapping", True)):
        return ["reference mapping disabled; prompts use config class_descriptions"]
    mapping_path = resolve_project_path(config.get("reference_mapping_path", "artifacts/selection/reference_class_mapping.yaml"))
    if mapping_path is None or not mapping_path.exists():
        return ["reference mapping file is missing"]
    payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    mapping = payload.get("mapping", {})
    warnings = []
    min_margin = float(config.get("reference_mapping_min_margin", 0.02))
    min_vote_fraction = float(config.get("reference_mapping_min_vote_fraction", 0.5))
    for class_name in class_names:
        item = mapping.get(class_name)
        if not item:
            warnings.append(f"{class_name}: missing reference mapping")
            continue
        votes = {str(key): int(value) for key, value in (item.get("vote_counts") or {}).items()}
        vote_total = sum(votes.values())
        top_votes = max(votes.values()) if votes else 0
        vote_fraction = top_votes / max(vote_total, 1)
        margin = float(item.get("average_prototype_margin", 0.0))
        if vote_fraction < min_vote_fraction:
            warnings.append(f"{class_name}: dispersed reference votes ({vote_fraction:.3f})")
        if margin < min_margin:
            warnings.append(f"{class_name}: low prototype margin ({margin:.6f})")
    return warnings


def render_prompt_family(
    family_cfg: dict[str, Any],
    class_names: list[str],
    class_descriptions: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """Render all prompts for a prompt family."""

    templates = list(family_cfg.get("templates", []))
    if not templates:
        raise ValueError("prompt family must define at least one template")
    rendered = {}
    for class_name in class_names:
        values = {"label": class_name, **class_descriptions[class_name]}
        rendered[class_name] = [str(template).format(**values) for template in templates]
    return rendered


def resolve_logit_scale(model) -> float:
    """Return CONCH/CLIP logit scale when exposed by the model."""

    scale = getattr(model, "logit_scale", None)
    if scale is None:
        return 100.0
    if hasattr(scale, "exp"):
        return float(scale.exp().detach().cpu().item())
    return float(scale)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def write_prompt_report(summary: dict[str, Any], path: Path) -> None:
    """Write a markdown report for prompt-family selection."""

    lines = [
        "# CONCH Prompt Probe Report",
        "",
        f"Run: `{summary['run_name']}`",
        f"Best family: `{summary['best_family']}`",
        "",
        "## Reference Mapping Warnings",
        "",
    ]
    warnings = summary.get("reference_mapping_warnings") or []
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Prompt Inputs",
            "",
            "| class | inferred cell name | description | family | prompts |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    descriptions = summary["class_descriptions"]
    for item in summary["prompt_families"]:
        for class_name, prompts in item["prompts_by_class"].items():
            desc = descriptions.get(class_name, {})
            lines.append(
                f"| `{class_name}` | {desc.get('cell_name', '')} | {desc.get('description', '')} | "
                f"`{item['family']}` | {'<br>'.join(prompts)} |"
            )
    lines.extend(
        [
            "",
            "## Prompt Family Metrics",
            "",
            "| rank | family | selection_score | macro_f1 | balanced_accuracy |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, item in enumerate(summary["prompt_families"], start=1):
        metrics = item["metrics"]
        lines.append(
            f"| {rank} | `{item['family']}` | {item['selection_score']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['balanced_accuracy']:.6f} |"
        )
    lines.extend(["", "## Prompt Families", ""])
    for item in summary["prompt_families"]:
        lines.append(f"### {item['family']}")
        for class_name, prompts in item["prompts_by_class"].items():
            lines.append(f"- `{class_name}`")
            for prompt in prompts:
                lines.append(f"  - {prompt}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_prompt_selection(summary: dict[str, Any], config: dict[str, Any]) -> None:
    """Write the selected prompt family into artifacts/selection."""

    output_path = resolve_project_path(
        config.get("selection_path", "artifacts/selection/conch_prompt_best.yaml")
    )
    if output_path is None:
        raise ValueError("selection_path cannot be None")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_family": summary["best_family"],
        "best_oof_path": summary["best_oof_path"],
        "best_test_path": summary["best_test_path"],
        "class_descriptions": summary["class_descriptions"],
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
