"""Tests for experiment orchestration utilities."""

from __future__ import annotations

import csv
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml_final.data.reference_align import run_reference_alignment
from ml_final.features.extract import iter_base_tta_views
from ml_final.backbones.factory import freeze_backbone_except_last_blocks
from ml_final.heads.ensemble_search import grid_weight_search
from ml_final.heads.conch_prompt import (
    apply_reference_mapping_override,
    normalize_class_descriptions,
    render_prompt_family,
)
from ml_final.heads.selection import select_scheme01
from ml_final.inference.submission import infer_final
from ml_final.pseudo.select import choose_per_class_policy_from_oof
from ml_final.training.checkpointing import CheckpointState, ModelEma, load_checkpoint, save_checkpoint
from ml_final.training.data import resolve_fill_rgb
from ml_final.training.peft_train import resolve_pretrained_cfg
from ml_final.training.pseudo_dataset import load_true_and_pseudo_rows
from ml_final.training.selection import select_scheme02
from ml_final.utils.config import load_yaml
from ml_final.utils.hf_cache import prepare_locked_hf_model_name


CLASS_NAMES = [f"Class_{idx}" for idx in range(5)]


def test_config_extends_deep_merge(tmp_path: Path):
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        """
a: 1
nested:
  x: 1
  y: 2
items: [old]
""",
        encoding="utf-8",
    )
    child.write_text(
        """
_extends: parent.yaml
nested:
  y: 3
items: [new]
""",
        encoding="utf-8",
    )

    config = load_yaml(child)

    assert config["a"] == 1
    assert config["nested"] == {"x": 1, "y": 3}
    assert config["items"] == ["new"]


def test_locked_hf_model_name_pins_revision_and_links_cache(tmp_path: Path):
    local_model = tmp_path / "model_store" / "repo"
    local_model.mkdir(parents=True)
    (local_model / "config.json").write_text("{}", encoding="utf-8")
    (local_model / "pytorch_model.bin").write_bytes(b"weights")
    cache_dir = tmp_path / "hf_home" / "hub"
    lock_path = tmp_path / "models.lock.yaml"
    lock_path.write_text(
        f"""
models:
  uni2_h:
    repo_id: MahmoodLab/UNI2-h
    revision: abc123
    local_path: {local_model.as_posix()}
    cache_path: {cache_dir.as_posix()}
""",
        encoding="utf-8",
    )

    model_name = prepare_locked_hf_model_name(
        "hf-hub:MahmoodLab/UNI2-h",
        lock_key="uni2_h",
        lock_path=lock_path,
    )

    assert model_name == "hf-hub:MahmoodLab/UNI2-h@abc123"
    assert os.environ["HF_HUB_CACHE"] == cache_dir.as_posix()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    snapshot = cache_dir / "models--MahmoodLab--UNI2-h" / "snapshots" / "abc123"
    assert (snapshot / "config.json").exists()
    assert (snapshot / "pytorch_model.bin").exists()
    assert (cache_dir / "models--MahmoodLab--UNI2-h" / "refs" / "main").read_text(encoding="utf-8") == "abc123\n"


def test_weight_search_prefers_strong_oof_source():
    y_true = np.arange(25) % len(CLASS_NAMES)
    strong = make_probs(y_true, 0.9)
    weak = make_probs((y_true + 1) % len(CLASS_NAMES), 0.9)

    result = grid_weight_search([strong, weak], y_true, CLASS_NAMES, step=0.5)

    assert result["metrics"]["macro_f1"] == 1.0
    assert result["weights"][0] > result["weights"][1]


def test_min_teacher_agreement_grid_is_class_specific():
    rows = []
    for idx, label in enumerate(CLASS_NAMES):
        probs = np.full(len(CLASS_NAMES), 0.025)
        probs[idx] = 0.9
        rows.append(
            {
                "filename": f"train_{idx}.png",
                "pred_label": label,
                "true_label": label,
                "prob_top1": 0.9,
                "prob_top2": 0.025,
                "margin": 0.875,
                "entropy": 0.4,
                "teacher_agreement": 1.0 if idx % 2 == 0 else 0.5,
                "probs": probs,
            }
        )
    config = {
        "precision_target": 0.8,
        "min_selected_per_class": 0,
        "policy_grid": {
            "prob_top1": [0.8],
            "margin": [0.0],
            "entropy_max": [1.0],
            "quota_per_class": [10],
            "min_teacher_agreement": [0.0, 1.0],
        },
    }

    policy = choose_per_class_policy_from_oof(config, rows, CLASS_NAMES)

    assert isinstance(policy["min_teacher_agreement"], dict)
    assert set(policy["min_teacher_agreement"]) == set(CLASS_NAMES)


def test_pseudo_dataset_loader_applies_lambda_and_matches_manifest(tmp_path: Path):
    train_manifest = tmp_path / "train_manifest.csv"
    test_manifest = tmp_path / "test_manifest.csv"
    pseudo_csv = tmp_path / "selected_pseudolabels.csv"
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()
    train_rows = []
    for idx, label in enumerate(CLASS_NAMES):
        path = train_dir / f"train_{idx}.png"
        make_image(path, value=idx * 20)
        train_rows.append({"filename": path.name, "label": label, "abs_path": str(path)})
    test_path = test_dir / "test_0.png"
    make_image(test_path, value=80)
    write_csv(train_manifest, ["filename", "label", "abs_path"], train_rows)
    write_csv(test_manifest, ["filename", "abs_path"], [{"filename": test_path.name, "abs_path": str(test_path)}])
    pseudo_row = {
        "filename": test_path.name,
        "pseudo_label": "Class_0",
        "sample_weight": "0.4",
        **{f"soft_label_{idx}": "0.2" for idx in range(5)},
    }
    write_csv(pseudo_csv, ["filename", "pseudo_label", "sample_weight"] + [f"soft_label_{idx}" for idx in range(5)], [pseudo_row])

    payload = load_true_and_pseudo_rows(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        pseudolabels=pseudo_csv,
        lambda_pseudo=0.1,
    )

    assert len(payload["train_rows"]) == 5
    assert len(payload["pseudo_rows"]) == 1
    assert abs(payload["pseudo_rows"][0]["sample_weight"] - 0.04) < 1e-12


def test_spatial_transform_fill_defaults_to_model_mean_rgb():
    assert resolve_fill_rgb({}, [0.5, 0.25, 1.0]) == (128, 64, 255)
    assert resolve_fill_rgb({"fill_rgb": [240, 235, 245]}, [0.5, 0.25, 1.0]) == (240, 235, 245)


def test_resolve_pretrained_cfg_through_peft_like_wrapper():
    class Backbone:
        pretrained_cfg = {"mean": (0.7, 0.6, 0.5), "std": (0.1, 0.2, 0.3)}

    class Classifier:
        backbone = Backbone()

    class BaseModel:
        model = Classifier()

    class PeftLikeWrapper:
        base_model = BaseModel()

    cfg = resolve_pretrained_cfg(PeftLikeWrapper())

    assert cfg == {"mean": (0.7, 0.6, 0.5), "std": (0.1, 0.2, 0.3)}


def test_freeze_backbone_except_last_blocks_trains_only_tail_norm_and_head():
    class TinyVit(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(3)])
            self.norm = torch.nn.LayerNorm(3)

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = TinyVit()
            self.classifier = torch.nn.Linear(3, 2)

    model = Classifier()

    selected = freeze_backbone_except_last_blocks(model, num_blocks=1, train_norm=True)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert selected == [2]
    assert "backbone.blocks.0.weight" not in trainable
    assert "backbone.blocks.1.weight" not in trainable
    assert "backbone.blocks.2.weight" in trainable
    assert "backbone.norm.weight" in trainable
    assert "classifier.weight" in trainable


def test_trainable_checkpoint_excludes_frozen_parameters(tmp_path: Path):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 2),
    )
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad])
    checkpoint_path = tmp_path / "last.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        state=CheckpointState(epoch=1, best_macro_f1=0.5, best_balanced_accuracy=0.4, history=[]),
        config={"test": True},
        model_scope="trainable",
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["model_scope"] == "trainable"
    assert "0.weight" not in payload["model"]
    assert "1.weight" in payload["model"]

    reloaded = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 2),
    )
    for parameter in reloaded[0].parameters():
        parameter.requires_grad = False
    state = load_checkpoint(checkpoint_path, model=reloaded, device=torch.device("cpu"))
    assert state.epoch == 1
    assert torch.allclose(reloaded[1].weight, model[1].weight)


def test_model_ema_apply_to_temporarily_swaps_and_restores_weights():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema = ModelEma(model, decay=0.9)
    ema.shadow["weight"].fill_(5.0)

    with ema.apply_to(model):
        assert torch.allclose(model.weight, torch.full_like(model.weight, 5.0))

    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))


def test_d4_is_full_eight_views_and_geom6_preserves_old_six_views():
    image = Image.fromarray(np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3))

    assert len(list(iter_base_tta_views(image, "d4"))) == 8
    assert len(list(iter_base_tta_views(image, "geom6"))) == 6


def test_scheme01_selection_writes_peft_eligible_backbones(tmp_path: Path):
    run_dir = tmp_path / "runs" / "scheme_01" / "s01_single"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    summary = {
        "run_name": "s01_single",
        "results": [
            {
                "head_id": "conch_logreg",
                "feature_file": "artifacts/features/scheme_01/s01_single/conch/features.npz",
                "macro_f1": 0.9,
                "balanced_accuracy": 0.9,
                "selection_score": 0.9,
                "spec": {"family": "logreg"},
            },
            {
                "head_id": "uni2_h_logreg",
                "feature_file": "artifacts/features/scheme_01/s01_single/uni2_h/features.npz",
                "macro_f1": 0.82,
                "balanced_accuracy": 0.82,
                "selection_score": 0.82,
                "spec": {"family": "logreg"},
            },
            {
                "head_id": "virchow2_logreg",
                "feature_file": "artifacts/features/scheme_01/s01_single/virchow2/features.npz",
                "macro_f1": 0.80,
                "balanced_accuracy": 0.80,
                "selection_score": 0.80,
                "spec": {"family": "logreg"},
            },
        ],
        "oof_prediction_files": [],
        "test_prediction_files": [],
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out_dir = tmp_path / "artifacts" / "selection"

    report = select_scheme01(runs=tmp_path / "runs" / "scheme_01", out=out_dir)

    assert report["selected_backbones"] == ["conch"]
    assert report["selected_peft_backbones"] == ["uni2_h"]
    assert (out_dir / "scheme01_top1_backbone.txt").read_text(encoding="utf-8") == "conch\n"
    assert (out_dir / "scheme01_top1_peft_backbone.txt").read_text(encoding="utf-8") == "uni2_h\n"


def test_scheme01_selection_writes_single_best_teacher_source(tmp_path: Path):
    run_dir = tmp_path / "runs" / "scheme_01" / "s01_conch_prompt"
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True)
    predictions_dir.mkdir()
    for name in (
        "oof_conch_prompt_he_cell.npz",
        "test_conch_prompt_he_cell.npz",
        "oof_uni2_h_logreg.npz",
        "test_uni2_h_logreg.npz",
    ):
        (predictions_dir / name).write_bytes(b"placeholder")
    summary = {
        "run_name": "s01_conch_prompt",
        "results": [
            {
                "head_id": "conch_prompt_he_cell",
                "feature_file": "conch_prompt_text_encoder",
                "macro_f1": 0.91,
                "balanced_accuracy": 0.91,
                "selection_score": 0.91,
                "spec": {"family": "conch_prompt"},
            },
            {
                "head_id": "uni2_h_logreg",
                "feature_file": "artifacts/features/scheme_01/s01_single/uni2_h/features.npz",
                "macro_f1": 0.92,
                "balanced_accuracy": 0.92,
                "selection_score": 0.92,
                "spec": {"family": "logreg"},
            },
        ],
        "oof_prediction_files": [str(predictions_dir / "oof_conch_prompt_he_cell.npz")],
        "test_prediction_files": [str(predictions_dir / "test_conch_prompt_he_cell.npz")],
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = select_scheme01(runs=tmp_path / "runs" / "scheme_01", out=tmp_path / "selection")

    assert len(report["teacher_predictions"]) == 2
    assert report["teacher_predictions"][0].endswith("predictions/oof_uni2_h_logreg.npz")
    assert report["teacher_predictions"][1].endswith("predictions/test_uni2_h_logreg.npz")
    source_file = tmp_path / "selection" / "scheme01_teacher_predictions.txt"
    assert source_file.read_text(encoding="utf-8").count("\n") == 2


def test_scheme02_selection_requires_single_refit_for_teacher_source(tmp_path: Path):
    run_dir = tmp_path / "runs" / "scheme_02" / "s02"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    summary = {
        "results": [
            {
                "experiment_id": "strong",
                "mode": "head_only",
                "macro_f1": 0.7,
                "balanced_accuracy": 0.7,
                "selection_score": 0.7,
                "oof_path": "runs/scheme_02/s02/predictions/oof_strong.npz",
                "test_path": "runs/scheme_02/s02/predictions/test_strong.npz",
                "test_prediction_kind": "cv_mean_debug",
            },
            {
                "experiment_id": "weak",
                "mode": "lora",
                "macro_f1": 0.6,
                "balanced_accuracy": 0.6,
                "selection_score": 0.6,
                "oof_path": "runs/scheme_02/s02/predictions/oof_weak.npz",
                "test_path": "runs/scheme_02/s02/predictions/test_weak.npz",
            },
        ]
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = select_scheme02(
        runs=tmp_path / "runs" / "scheme_02",
        out=tmp_path / "selection",
        scheme01=None,
    )

    assert report["best"]["experiment_id"] == "strong"
    source_file = tmp_path / "selection" / "scheme02_teacher_predictions.txt"
    assert source_file.read_text(encoding="utf-8").splitlines() == []


def test_scheme02_selection_writes_single_refit_teacher_source(tmp_path: Path):
    run_dir = tmp_path / "runs" / "scheme_02" / "s02_refit"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    summary = {
        "results": [
            {
                "experiment_id": "strong",
                "mode": "lora",
                "macro_f1": 0.7,
                "balanced_accuracy": 0.7,
                "selection_score": 0.7,
                "oof_path": "runs/scheme_02/s02/predictions/oof_strong.npz",
                "test_path": "runs/scheme_02/s02_refit/predictions/test_strong_single_refit.npz",
                "test_prediction_kind": "single_refit",
            }
        ]
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = select_scheme02(
        runs=tmp_path / "runs" / "scheme_02",
        out=tmp_path / "selection",
        scheme01=None,
    )

    assert report["best"]["experiment_id"] == "strong"
    source_file = tmp_path / "selection" / "scheme02_teacher_predictions.txt"
    assert source_file.read_text(encoding="utf-8").splitlines() == [
        "runs/scheme_02/s02/predictions/oof_strong.npz",
        "runs/scheme_02/s02_refit/predictions/test_strong_single_refit.npz",
    ]


def test_scheme02_selection_skips_smoke_debug_runs(tmp_path: Path):
    smoke_dir = tmp_path / "runs" / "scheme_02" / "s02_gpu_smoke"
    smoke_metrics = smoke_dir / "metrics"
    smoke_metrics.mkdir(parents=True)
    smoke_summary = {
        "run_name": "s02_gpu_smoke",
        "results": [
            {
                "experiment_id": "smoke_winner",
                "mode": "lora",
                "macro_f1": 0.99,
                "balanced_accuracy": 0.99,
                "selection_score": 0.99,
                "oof_path": "runs/scheme_02/s02_gpu_smoke/predictions/oof_smoke.npz",
            }
        ],
    }
    (smoke_metrics / "summary.json").write_text(json.dumps(smoke_summary), encoding="utf-8")

    formal_dir = tmp_path / "runs" / "scheme_02" / "s02_formal"
    formal_metrics = formal_dir / "metrics"
    formal_metrics.mkdir(parents=True)
    formal_summary = {
        "run_name": "s02_formal",
        "results": [
            {
                "experiment_id": "formal",
                "mode": "lora",
                "macro_f1": 0.7,
                "balanced_accuracy": 0.7,
                "selection_score": 0.7,
                "oof_path": "runs/scheme_02/s02_formal/predictions/oof_formal.npz",
            }
        ],
    }
    (formal_metrics / "summary.json").write_text(json.dumps(formal_summary), encoding="utf-8")

    report = select_scheme02(
        runs=tmp_path / "runs" / "scheme_02",
        out=tmp_path / "selection",
        scheme01=None,
    )

    assert report["best"]["experiment_id"] == "formal"
    assert [row["experiment_id"] for row in report["results"]] == ["formal"]


def test_conch_prompt_templates_render_actual_cell_names():
    descriptions = normalize_class_descriptions(
        {
            "Class_0": {
                "cell_name": "epithelial cell",
                "description": "epithelial cells in an H&E stained crop",
            },
            "Class_1": {
                "cell_name": "neutrophil",
                "description": "neutrophils with segmented nuclei",
            },
        },
        ["Class_0", "Class_1"],
    )

    prompts = render_prompt_family(
        {
            "templates": [
                "an H&E stained image showing {cell_name}.",
                "a crop containing {description}.",
            ]
        },
        ["Class_0", "Class_1"],
        descriptions,
    )

    assert prompts["Class_0"][0] == "an H&E stained image showing epithelial cell."
    assert prompts["Class_1"][1] == "a crop containing neutrophils with segmented nuclei."


def test_conch_prompt_uses_reference_mapping_when_available(tmp_path: Path):
    mapping_path = tmp_path / "reference_class_mapping.yaml"
    mapping_path.write_text(
        """
mapping:
  Class_0:
    selected_reference_label: Epithelial
    prompt_cell_name: epithelial cell
""",
        encoding="utf-8",
    )
    descriptions = {"Class_0": {"cell_name": "old name", "description": "old description"}}

    updated = apply_reference_mapping_override(
        descriptions,
        {"reference_mapping_path": str(mapping_path), "use_reference_mapping": True},
    )

    assert updated["Class_0"]["cell_name"] == "epithelial cell"
    assert descriptions["Class_0"]["cell_name"] == "old name"


def test_conch_prompt_requires_reference_mapping_in_formal_mode(tmp_path: Path):
    descriptions = {"Class_0": {"cell_name": "old name", "description": "old description"}}

    with pytest.raises(FileNotFoundError, match="requires reference_class_mapping"):
        apply_reference_mapping_override(
            descriptions,
            {"reference_mapping_path": str(tmp_path / "missing.yaml"), "use_reference_mapping": True},
        )


def test_final_inference_rejects_directory_search_without_explicit_opt_in(tmp_path: Path):
    search_dir = tmp_path / "runs"
    search_dir.mkdir()
    config = tmp_path / "infer.json"
    config.write_text(
        json.dumps({"prediction_search_dir": str(search_dir), "submission_dir": str(tmp_path / "out")}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="requires explicit"):
        infer_final(config)


def test_final_inference_reads_explicit_prediction_source_file(tmp_path: Path):
    prediction_path = tmp_path / "test_model.npz"
    np.savez_compressed(
        prediction_path,
        probs=np.eye(len(CLASS_NAMES), dtype=np.float32),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        test_filenames=np.asarray([f"test_{idx}.png" for idx in range(len(CLASS_NAMES))], dtype=object),
    )
    source_file = tmp_path / "teacher_predictions.txt"
    source_file.write_text(f"{prediction_path}\n", encoding="utf-8")
    config = tmp_path / "infer.json"
    config.write_text(
        json.dumps(
            {
                "test_prediction_sources_file": str(source_file),
                "submission_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )

    result = infer_final(config)

    assert result["num_rows"] == len(CLASS_NAMES)
    assert Path(result["submission"]).exists()
    assert Path(result["metadata"]).exists()


def test_reference_alignment_reads_zip_without_extracting(tmp_path: Path):
    train_manifest = tmp_path / "train_manifest.csv"
    train_dir = tmp_path / "current"
    train_dir.mkdir()
    current_rows = []
    for class_idx, label in enumerate(CLASS_NAMES[:2]):
        for item_idx in range(3):
            path = train_dir / label / f"{item_idx}.png"
            path.parent.mkdir(exist_ok=True)
            make_image(path, value=20 + class_idx * 180)
            current_rows.append({"filename": path.name, "label": label, "abs_path": str(path)})
    write_csv(train_manifest, ["filename", "label", "abs_path"], current_rows)

    reference_zip = tmp_path / "reference.zip"
    with zipfile.ZipFile(reference_zip, "w") as archive:
        for label, value in [("Dark", 20), ("Bright", 200)]:
            for item_idx in range(4):
                image_path = tmp_path / f"{label}_{item_idx}.png"
                make_image(image_path, value=value)
                archive.write(image_path, arcname=f"train/{label}/{item_idx}.png")

    config = tmp_path / "alignment.yaml"
    config.write_text(
        f"""
run_name: smoke_reference_alignment
current_manifest: {train_manifest}
reference_zip: {reference_zip}
out_dir: {tmp_path / "out"}
mapping_out: {tmp_path / "mapping.yaml"}
sample_per_class: 4
reference_classes: [Dark, Bright]
prompt_name_map:
  Dark: dark cell
  Bright: bright cell
fit_reference_logreg: false
backbones:
  - key: pixel_stats
    backend: pixel_stats
    input_size: 32
""",
        encoding="utf-8",
    )

    result = run_reference_alignment(config, smoke=True)

    mapping = result["summary"]["consensus_mapping"]
    assert mapping["Class_0"]["selected_reference_label"] == "Dark"
    assert mapping["Class_1"]["selected_reference_label"] == "Bright"
    assert Path(result["summary_path"]).exists()


def make_probs(labels: np.ndarray, confidence: float) -> np.ndarray:
    probs = np.full((len(labels), len(CLASS_NAMES)), (1.0 - confidence) / (len(CLASS_NAMES) - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def make_image(path: Path, *, value: int) -> None:
    arr = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
