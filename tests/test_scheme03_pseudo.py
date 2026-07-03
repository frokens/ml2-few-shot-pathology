"""Tests for Scheme 03 teacher, pseudo-label selection, and retraining."""

from __future__ import annotations

import json
import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ml_final.features.store import save_feature_bundle
from ml_final.inference.submission import infer_final, validate_submission
from ml_final.inference.class_bias import apply_class_bias, calibrate_class_bias
from ml_final.pseudo.calibrate import build_teacher
from ml_final.pseudo.select import select_pseudolabels
from ml_final.pseudo.retrain import train_pseudo_heads
from ml_final.training import fusion_peft as fusion_peft_module
from ml_final.training.losses import mixed_hard_soft_focal_loss, mixed_hard_soft_loss, soft_cross_entropy
from ml_final.training.pseudo_peft import resolve_single_refit_epochs, run_pseudo_lora_single_refit
from ml_final.training.pseudo_dataset import MultiTransformPseudoDataset, load_true_and_pseudo_rows
from ml_final.utils.config import load_yaml
from ml_final.utils.paths import project_root


CLASS_NAMES = [f"Class_{idx}" for idx in range(5)]


def test_teacher_build_temperature_and_csv_outputs(tmp_path: Path):
    paths = make_prediction_files(tmp_path)
    config = {
        "seed": 2026,
        "class_names": CLASS_NAMES,
        "oof_prediction_files": [str(paths["oof"])],
        "test_prediction_files": [str(paths["test"])],
        "temperature_grid": [0.75, 1.0, 1.5],
    }
    config_path = tmp_path / "teacher.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = build_teacher(config_path, out=tmp_path / "teacher")

    assert result["num_oof_rows"] == 25
    assert result["num_test_rows"] == 8
    assert result["temperature"] in {0.75, 1.0, 1.5}
    assert (tmp_path / "teacher" / "teacher_oof_predictions.csv").exists()
    assert (tmp_path / "teacher" / "teacher_test_predictions.csv").exists()
    assert (tmp_path / "teacher" / "calibration_report.md").exists()


def test_teacher_does_not_search_prediction_dirs_without_explicit_opt_in(tmp_path: Path):
    paths = make_prediction_files(tmp_path)
    config = {
        "seed": 2026,
        "class_names": CLASS_NAMES,
        "oof_prediction_search_dir": str(paths["oof"].parent),
        "oof_pattern": "oof_*.npz",
        "test_prediction_search_dir": str(paths["test"].parent),
        "test_pattern": "test_*.npz",
        "temperature_grid": [1.0],
    }
    config_path = tmp_path / "teacher_no_search.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="no OOF prediction files"):
        build_teacher(config_path, out=tmp_path / "teacher")


def test_teacher_rejects_external_reference_path_in_formal_config(tmp_path: Path):
    config = {
        "seed": 2026,
        "class_names": CLASS_NAMES,
        "oof_prediction_files": ["external_reference/train.zip"],
        "temperature_grid": [1.0],
    }
    config_path = tmp_path / "teacher_bad_reference.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="external reference data path"):
        build_teacher(config_path, out=tmp_path / "teacher")


def test_mixed_hard_soft_loss_default_matches_legacy_sum():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=torch.float32)
    hard = torch.tensor([0, 0, 1], dtype=torch.long)
    soft = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=torch.float32)
    sample_weight = torch.tensor([1.0, 0.1, 1.0])
    is_pseudo = torch.tensor([False, True, True])

    actual = mixed_hard_soft_loss(logits, hard, soft, sample_weight, is_pseudo)
    expected = torch.nn.functional.cross_entropy(logits[:1], hard[:1]) + soft_cross_entropy(
        logits[1:],
        soft[1:],
        sample_weight[1:],
    )

    assert torch.allclose(actual, expected)


def test_mixed_hard_soft_loss_can_scale_pseudo_term():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=torch.float32)
    hard = torch.tensor([0, 0, 1], dtype=torch.long)
    soft = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=torch.float32)
    sample_weight = torch.tensor([1.0, 0.1, 1.0])
    is_pseudo = torch.tensor([False, True, True])

    scaled = mixed_hard_soft_loss(
        logits,
        hard,
        soft,
        sample_weight,
        is_pseudo,
        pseudo_loss_scale=0.25,
    )
    expected = torch.nn.functional.cross_entropy(logits[:1], hard[:1]) + 0.25 * soft_cross_entropy(
        logits[1:],
        soft[1:],
        sample_weight[1:],
    )

    assert torch.allclose(scaled, expected)


def test_mixed_hard_soft_loss_can_add_true_soft_distillation():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    hard = torch.tensor([0, 1], dtype=torch.long)
    soft = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
    sample_weight = torch.tensor([1.0, 1.0])
    is_pseudo = torch.tensor([False, False])

    scaled = mixed_hard_soft_loss(
        logits,
        hard,
        soft,
        sample_weight,
        is_pseudo,
        true_soft_loss_scale=0.2,
    )
    expected = torch.nn.functional.cross_entropy(logits, hard) + 0.2 * soft_cross_entropy(
        logits,
        soft,
        sample_weight,
    )

    assert torch.allclose(scaled, expected)


def test_mixed_hard_soft_loss_uses_true_sample_weight():
    logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
    hard = torch.tensor([0, 1], dtype=torch.long)
    soft = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    is_pseudo = torch.tensor([False, False])
    weighted = mixed_hard_soft_loss(
        logits,
        hard,
        soft,
        torch.tensor([0.1, 1.0]),
        is_pseudo,
    )
    unweighted = mixed_hard_soft_loss(
        logits,
        hard,
        soft,
        torch.tensor([1.0, 1.0]),
        is_pseudo,
    )

    assert weighted < unweighted


def test_mixed_hard_soft_focal_gamma_zero_matches_base_loss():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=torch.float32)
    hard = torch.tensor([0, 0, 1], dtype=torch.long)
    soft = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=torch.float32)
    sample_weight = torch.tensor([1.0, 0.1, 1.0])
    is_pseudo = torch.tensor([False, True, True])

    base = mixed_hard_soft_loss(logits, hard, soft, sample_weight, is_pseudo)
    focal = mixed_hard_soft_focal_loss(
        logits,
        hard,
        soft,
        sample_weight,
        is_pseudo,
        gamma=0.0,
    )

    assert torch.allclose(focal, base)

    scaled_base = mixed_hard_soft_loss(logits, hard, soft, sample_weight, is_pseudo, pseudo_loss_scale=0.25)
    scaled_focal = mixed_hard_soft_focal_loss(
        logits,
        hard,
        soft,
        sample_weight,
        is_pseudo,
        gamma=0.0,
        pseudo_loss_scale=0.25,
    )

    assert torch.allclose(scaled_focal, scaled_base)


def test_mixed_hard_soft_loss_applies_class_weight():
    logits = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]], dtype=torch.float32)
    hard = torch.tensor([0, 1, 1], dtype=torch.long)
    soft = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    is_pseudo = torch.tensor([False, False, True])
    sample_weight = torch.tensor([1.0, 1.0, 1.0])

    base = mixed_hard_soft_loss(logits, hard, soft, sample_weight, is_pseudo)
    weighted = mixed_hard_soft_loss(
        logits,
        hard,
        soft,
        sample_weight,
        is_pseudo,
        class_weight=torch.tensor([2.0, 1.0]),
    )

    assert weighted > base


def test_resolve_group_aux_loss_is_opt_in_and_accepts_class_names():
    assert fusion_peft_module.resolve_group_aux_loss({}, {}, CLASS_NAMES) is None

    resolved = fusion_peft_module.resolve_group_aux_loss(
        {
            "group_aux_loss": {
                "weight": 0.2,
                "positive_classes": ["Class_2", "Class_3", "Class_4"],
            }
        },
        {},
        CLASS_NAMES,
    )

    assert resolved == {
        "weight": 0.2,
        "positive_indices": [2, 3, 4],
        "positive_classes": ["Class_2", "Class_3", "Class_4"],
    }


def test_compute_group_aux_loss_handles_true_and_pseudo_rows():
    logits = torch.tensor(
        [
            [3.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    hard = torch.tensor([0, 2, 2], dtype=torch.long)
    soft = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.1, 0.1],
            [0.2, 0.1, 0.7, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sample_weight = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32)
    is_pseudo = torch.tensor([False, False, True])

    loss = fusion_peft_module.compute_group_aux_loss(
        logits,
        hard,
        positive_indices=[2, 3, 4],
        soft_labels=soft,
        sample_weight=sample_weight,
        is_pseudo=is_pseudo,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss > 0


def test_pseudolabel_simulation_and_test_selection_schema(tmp_path: Path):
    paths = make_prediction_files(tmp_path)
    teacher_config = {
        "seed": 2026,
        "class_names": CLASS_NAMES,
        "oof_prediction_files": [str(paths["oof"])],
        "test_prediction_files": [str(paths["test"])],
        "temperature_grid": [1.0],
    }
    teacher_config_path = tmp_path / "teacher.json"
    teacher_config_path.write_text(json.dumps(teacher_config), encoding="utf-8")
    build_teacher(teacher_config_path, out=tmp_path / "teacher")

    select_config = {
        "class_names": CLASS_NAMES,
        "precision_target": 0.8,
        "min_selected": 1,
        "min_selected_per_class": 1,
        "per_class_policy": True,
        "policy_grid": {
            "prob_top1": [0.5, 0.7],
            "margin": [0.0, 0.05],
            "entropy_max": [1.5],
            "quota_per_class": [2, 10],
        },
        "sample_weight": {"min_weight": 0.1, "max_weight": 0.5},
    }
    select_config_path = tmp_path / "select.json"
    select_config_path.write_text(json.dumps(select_config), encoding="utf-8")
    sim = select_pseudolabels(
        select_config_path,
        mode="simulate",
        teacher=tmp_path / "teacher",
        out=tmp_path / "sim",
    )
    assert sim["num_selected"] > 0
    assert (tmp_path / "sim" / "thresholds_selected.json").exists()
    thresholds = json.loads((tmp_path / "sim" / "thresholds_selected.json").read_text(encoding="utf-8"))
    assert thresholds["selected_by"] == "per_class_oof_simulation"
    assert set(thresholds["prob_top1"]) == set(CLASS_NAMES)
    assert set(thresholds["entropy_max"]) == set(CLASS_NAMES)

    select_test_config = dict(select_config)
    select_test_config["thresholds_from"] = str(tmp_path / "sim" / "thresholds_selected.json")
    select_test_config_path = tmp_path / "select_test.json"
    select_test_config_path.write_text(json.dumps(select_test_config), encoding="utf-8")
    test = select_pseudolabels(
        select_test_config_path,
        mode="select-test",
        teacher=tmp_path / "teacher",
        out=tmp_path / "test_select",
    )
    assert test["num_selected"] > 0
    header = (tmp_path / "test_select" / "selected_pseudolabels.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    for field in [
        "filename",
        "pseudo_label",
        "prob_top1",
        "sample_weight",
        "soft_label_0",
        "soft_label_4",
    ]:
        assert field in header


def test_fixed_policy_simulation_uses_configured_policy_and_weight_estimates(tmp_path: Path):
    teacher_dir = make_teacher_csv_for_fixed_policy(tmp_path)
    config = {
        "class_names": ["Class_0", "Class_1"],
        "policy_source": "fixed",
        "effective_weight_lambdas": [0.02, 0.05],
        "policy": {
            "prob_top1": {"Class_0": 0.8, "Class_1": 0.95},
            "margin": {"Class_0": 0.1, "Class_1": 0.1},
            "entropy_max": {"Class_0": 0.5, "Class_1": 0.5},
            "quota": {"Class_0": 10, "Class_1": 0},
            "sample_weight": {"min_weight": 0.1, "max_weight": 0.3},
        },
    }
    config_path = tmp_path / "fixed_policy.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = select_pseudolabels(config_path, mode="simulate", teacher=teacher_dir, out=tmp_path / "fixed")

    policy = result["policy"]
    assert policy["policy_source"] == "fixed"
    assert policy["selected_by"] == "fixed_oof_simulation"
    assert policy["quota"]["Class_1"] == 0
    assert result["class_distribution"]["counts"] == {"Class_0": 2, "Class_1": 0}
    estimates = result["effective_pseudo_weight_estimates"]
    assert estimates["raw_sample_weight_sum"] > 0
    assert "0.02" in estimates["lambda_scaled"]
    assert "0.05" in estimates["lambda_scaled"]


def test_balanced_soft_expansion_selects_ranked_equal_class_pool(tmp_path: Path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    class_names = ["Class_0", "Class_1"]
    rows = []
    for class_idx, class_name in enumerate(class_names):
        for rank, confidence in enumerate([0.96, 0.91, 0.76, 0.61, 0.40], start=1):
            probs = [1.0 - confidence, 1.0 - confidence]
            probs[class_idx] = confidence
            rows.append(teacher_row(f"{class_name}_{rank}.png", class_name, class_name, probs))
    write_teacher_prediction_csv(teacher_dir / "teacher_oof_predictions.csv", rows, class_names, include_truth=True)
    config = {
        "class_names": class_names,
        "policy_source": "fixed",
        "policy": {
            "prob_top1": 0.85,
            "margin": 0.0,
            "entropy_max": 10.0,
            "quota": 2,
            "sample_weight": {"min_weight": 0.1, "max_weight": 0.5},
        },
        "balanced_soft_expansion": {
            "enabled": True,
            "core_quota": 2,
            "expanded_quota": 4,
            "rank_weight_multipliers": [
                {"start_rank": 1, "end_rank": 2, "multiplier": 1.0},
                {"start_rank": 3, "end_rank": 4, "multiplier": 0.25},
            ],
        },
        "effective_weight_lambdas": [0.001],
    }
    config_path = tmp_path / "soft_expand.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = select_pseudolabels(config_path, mode="simulate", teacher=teacher_dir, out=tmp_path / "soft")

    assert result["num_selected"] == 8
    assert result["class_distribution"]["counts"] == {"Class_0": 4, "Class_1": 4}
    assert result["policy"]["balanced_soft_expansion"]["expanded_quota"] == 4
    selected_rows = list(csv.DictReader((tmp_path / "soft" / "selected_pseudolabels.csv").open(encoding="utf-8")))
    low_weight_rows = [row for row in selected_rows if row["filename"].endswith("_3.png") or row["filename"].endswith("_4.png")]
    assert low_weight_rows
    assert all(float(row["sample_weight"]) < 0.1 for row in low_weight_rows)
    estimates = result["effective_pseudo_weight_estimates"]
    assert "0.001" in estimates["lambda_scaled"]


def test_fixed_policy_pruning_sets_failed_class_quota_to_zero(tmp_path: Path):
    teacher_dir = make_teacher_csv_for_fixed_policy(tmp_path)
    config = {
        "class_names": ["Class_0", "Class_1"],
        "policy_source": "fixed",
        "prune_failed_classes": True,
        "pruning": {
            "Class_0": {"min_precision": 0.9, "min_selected": 1},
            "Class_1": {"min_precision": 0.95, "min_selected": 1},
        },
        "policy": {
            "prob_top1": {"Class_0": 0.8, "Class_1": 0.8},
            "margin": {"Class_0": 0.1, "Class_1": 0.1},
            "entropy_max": {"Class_0": 0.5, "Class_1": 0.5},
            "quota": {"Class_0": 10, "Class_1": 10},
            "sample_weight": {"min_weight": 0.1, "max_weight": 0.3},
        },
    }
    config_path = tmp_path / "prune_policy.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = select_pseudolabels(config_path, mode="simulate", teacher=teacher_dir, out=tmp_path / "pruned")

    assert result["policy"]["pruned_classes"] == ["Class_1"]
    assert result["policy"]["quota"]["Class_1"] == 0
    assert result["class_distribution"]["counts"] == {"Class_0": 2, "Class_1": 0}
    assert result["simulation_metrics"]["per_class"]["Class_1"]["selected"] == 0


def test_strategy_a_config_never_selects_class3_or_class4(tmp_path: Path):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    rows = [
        teacher_row("img0.png", "Class_0", "Class_0", [0.9, 0.025, 0.025, 0.025, 0.025]),
        teacher_row("img1.png", "Class_1", "Class_1", [0.025, 0.9, 0.025, 0.025, 0.025]),
        teacher_row("img2.png", "Class_2", "Class_2", [0.025, 0.025, 0.9, 0.025, 0.025]),
        teacher_row("img3.png", "Class_3", "Class_3", [0.025, 0.025, 0.025, 0.9, 0.025]),
        teacher_row("img4.png", "Class_4", "Class_4", [0.025, 0.025, 0.025, 0.025, 0.9]),
    ]
    write_teacher_prediction_csv(teacher_dir / "teacher_oof_predictions.csv", rows, CLASS_NAMES, include_truth=True)

    result = select_pseudolabels(
        "configs/scheme_03/pseudolabel_s02_strategy_a_safe.yaml",
        mode="simulate",
        teacher=teacher_dir,
        out=tmp_path / "strategy_a",
    )

    counts = result["class_distribution"]["counts"]
    assert counts["Class_3"] == 0
    assert counts["Class_4"] == 0
    assert result["policy"]["quota"]["Class_3"] == 0
    assert result["policy"]["quota"]["Class_4"] == 0


def test_pseudo_retrain_weighted_logreg_on_feature_bundle(tmp_path: Path):
    paths = make_prediction_files(tmp_path)
    teacher_config = {
        "seed": 2026,
        "class_names": CLASS_NAMES,
        "oof_prediction_files": [str(paths["oof"])],
        "test_prediction_files": [str(paths["test"])],
        "temperature_grid": [1.0],
    }
    teacher_config_path = tmp_path / "teacher.json"
    teacher_config_path.write_text(json.dumps(teacher_config), encoding="utf-8")
    build_teacher(teacher_config_path, out=tmp_path / "teacher")
    policy = {
        "class_names": CLASS_NAMES,
        "policy": {
            "prob_top1": {name: 0.4 for name in CLASS_NAMES},
            "margin": {name: 0.0 for name in CLASS_NAMES},
            "entropy_max": 2.0,
            "quota": {name: 10 for name in CLASS_NAMES},
            "sample_weight": {"min_weight": 0.1, "max_weight": 0.5},
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    select_pseudolabels(
        policy_path,
        mode="select-test",
        teacher=tmp_path / "teacher",
        out=tmp_path / "pseudo",
    )
    features_root = tmp_path / "features"
    make_feature_bundle(features_root / "synthetic_features.npz")
    config = {
        "class_names": CLASS_NAMES,
        "lambda_pseudo": 0.2,
        "heads": [{"name": "logreg_pseudo_C1", "family": "logreg", "C": 1.0}],
    }
    config_path = tmp_path / "retrain.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = train_pseudo_heads(
        config_path,
        features=features_root,
        pseudolabels=tmp_path / "pseudo" / "selected_pseudolabels.csv",
        run_name="pytest_scheme03_pseudo",
    )

    assert result["summary"]["status"] == "ok"
    assert result["summary"]["results"]


def test_multi_transform_pseudo_dataset_returns_branch_images(tmp_path: Path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    true_rows = [{"filename": "img.png", "label": "Class_0", "abs_path": str(image_path)}]
    pseudo_rows = [
        {
            "filename": "img.png",
            "abs_path": str(image_path),
            "pseudo_label": "Class_1",
            "soft_label": np.asarray([0.2, 0.8], dtype=np.float32),
            "sample_weight": 0.1,
        }
    ]
    dataset = MultiTransformPseudoDataset(
        true_rows,
        pseudo_rows,
        class_names=["Class_0", "Class_1"],
        label_to_idx={"Class_0": 0, "Class_1": 1},
        transforms_by_key={"a": lambda image: torch.ones(3, 8, 8), "b": lambda image: torch.zeros(3, 8, 8)},
        n_classes=2,
    )

    item = dataset[1]

    assert set(item["images_by_branch"]) == {"a", "b"}
    assert item["is_pseudo"].item() is True
    assert item["hard_label"].item() == 1
    assert torch.isclose(item["sample_weight"], torch.tensor(0.1))


def test_multi_transform_pseudo_dataset_keeps_true_soft_labels(tmp_path: Path):
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    true_rows = [
        {
            "filename": "img.png",
            "label": "Class_0",
            "abs_path": str(image_path),
            "soft_label": np.asarray([0.8, 0.2], dtype=np.float32),
        }
    ]
    dataset = MultiTransformPseudoDataset(
        true_rows,
        [],
        class_names=["Class_0", "Class_1"],
        label_to_idx={"Class_0": 0, "Class_1": 1},
        transforms_by_key={"a": lambda image: torch.ones(3, 8, 8)},
        n_classes=2,
    )

    item = dataset[0]

    assert item["is_pseudo"].item() is False
    assert torch.allclose(item["soft_label"], torch.tensor([0.8, 0.2]))


def test_load_true_and_pseudo_rows_attaches_true_soft_labels(tmp_path: Path):
    train_manifest = tmp_path / "train.csv"
    test_manifest = tmp_path / "test.csv"
    pseudo_csv = tmp_path / "pseudo.csv"
    soft_csv = tmp_path / "true_soft.csv"
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    train_manifest.write_text(
        f"filename,label,abs_path\nimg.png,Class_0,{image_path}\nimg_b.png,Class_1,{image_path}\n",
        encoding="utf-8",
    )
    test_manifest.write_text(
        f"filename,abs_path\ntest.png,{image_path}\n",
        encoding="utf-8",
    )
    pseudo_csv.write_text(
        "filename,pseudo_label,sample_weight,soft_label_0,soft_label_1\n"
        "test.png,Class_1,0.5,0.1,0.9\n",
        encoding="utf-8",
    )
    soft_csv.write_text(
        "filename,sample_weight,soft_label_0,soft_label_1\n"
        "img.png,0.4,0.7,0.3\n",
        encoding="utf-8",
    )

    payload = load_true_and_pseudo_rows(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        pseudolabels=pseudo_csv,
        lambda_pseudo=0.1,
        true_soft_labels=soft_csv,
    )

    assert np.allclose(payload["train_rows"][0]["soft_label"], np.asarray([0.7, 0.3], dtype=np.float32))
    assert payload["train_rows"][0]["sample_weight"] == pytest.approx(0.4)
    assert payload["pseudo_rows"][0]["sample_weight"] == pytest.approx(0.05)


def test_load_true_and_pseudo_rows_default_pseudo_class_weight_matches_legacy(tmp_path: Path):
    train_manifest, test_manifest, pseudo_csv = make_tiny_pseudo_manifests(tmp_path)

    payload = load_true_and_pseudo_rows(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        pseudolabels=pseudo_csv,
        lambda_pseudo=0.1,
    )

    weights = {row["filename"]: row["sample_weight"] for row in payload["pseudo_rows"]}
    assert weights["test_a.png"] == pytest.approx(0.08)
    assert weights["test_b.png"] == pytest.approx(0.05)
    assert payload["pseudo_class_weight"] == {"Class_0": 1.0, "Class_1": 1.0}


def test_load_true_and_pseudo_rows_scales_only_pseudo_class_weight(tmp_path: Path):
    train_manifest, test_manifest, pseudo_csv = make_tiny_pseudo_manifests(tmp_path)
    soft_csv = tmp_path / "true_soft.csv"
    soft_csv.write_text(
        "filename,sample_weight,soft_label_0,soft_label_1\n"
        "train_a.png,0.4,0.7,0.3\n",
        encoding="utf-8",
    )

    payload = load_true_and_pseudo_rows(
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        pseudolabels=pseudo_csv,
        lambda_pseudo=0.1,
        true_soft_labels=soft_csv,
        pseudo_class_weight={"Class_1": 0.5},
    )

    true_a = next(row for row in payload["train_rows"] if row["filename"] == "train_a.png")
    pseudo_weights = {row["filename"]: row["sample_weight"] for row in payload["pseudo_rows"]}
    assert true_a["sample_weight"] == pytest.approx(0.4)
    assert pseudo_weights["test_a.png"] == pytest.approx(0.08)
    assert pseudo_weights["test_b.png"] == pytest.approx(0.025)
    assert payload["pseudo_effective_weight_sum_by_class"] == pytest.approx({"Class_0": 0.08, "Class_1": 0.025})


def test_load_true_and_pseudo_rows_rejects_bad_pseudo_class_weight(tmp_path: Path):
    train_manifest, test_manifest, pseudo_csv = make_tiny_pseudo_manifests(tmp_path)

    with pytest.raises(ValueError, match="unknown classes"):
        load_true_and_pseudo_rows(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            pseudolabels=pseudo_csv,
            lambda_pseudo=0.1,
            pseudo_class_weight={"Class_9": 1.0},
        )
    with pytest.raises(ValueError, match="non-negative"):
        load_true_and_pseudo_rows(
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            pseudolabels=pseudo_csv,
            lambda_pseudo=0.1,
            pseudo_class_weight={"Class_1": -0.1},
        )


def test_scheme03_final_submission_config_is_explicit_single_source() -> None:
    config = load_yaml("configs/scheme_03/final_submission_scheme02b_best.yaml")
    uni2 = load_yaml("configs/scheme_03/peft_uni2_single_refit_best.yaml")
    conch = load_yaml("configs/scheme_03/peft_conch_single_refit_best.yaml")

    assert len(config["test_prediction_files"]) == 1
    assert config["allow_prediction_search"] is False
    assert config["test_prediction_sources_file"] is None
    assert uni2["epochs"] == 35
    assert conch["epochs"] == 26


def test_fusion_pseudo_cv_uses_pseudo_rows_without_test_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_manifest = tmp_path / "train_manifest.csv"
    test_manifest = tmp_path / "test_manifest.csv"
    pseudo_csv = tmp_path / "selected_pseudolabels.csv"
    train_manifest.write_text(
        "\n".join(
            [
                "filename,label,abs_path",
                f"train_0.png,Class_0,{tmp_path / 'train_0.png'}",
                f"train_1.png,Class_0,{tmp_path / 'train_1.png'}",
                f"train_2.png,Class_1,{tmp_path / 'train_2.png'}",
                f"train_3.png,Class_1,{tmp_path / 'train_3.png'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test_manifest.write_text(
        "\n".join(
            [
                "filename,abs_path",
                f"test_0.png,{tmp_path / 'test_0.png'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pseudo_csv.write_text(
        "\n".join(
            [
                "filename,pseudo_label,sample_weight,soft_label_0,soft_label_1",
                "test_0.png,Class_1,0.8,0.1,0.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_name = "pytest_fusion_pseudo_cv"
    run_dir = project_root() / "runs" / "scheme_02" / run_name
    shutil.rmtree(run_dir, ignore_errors=True)
    config = {
        "run_name": run_name,
        "seed": 2026,
        "train_manifest": str(train_manifest),
        "test_manifest": str(test_manifest),
        "n_splits": 2,
        "max_folds": 2,
        "lambda_pseudo": 0.05,
        "min_pseudo_sample_weight": 0.01,
        "experiments": [{"name": "b1_uni_conch_adapters_all4_concat_p32_reg", "composer": "concat_linear"}],
        "fusion_heads": [],
    }
    config_path = tmp_path / "fusion_pseudo_cv.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_train_fusion_fold(**kwargs):
        pseudo_rows = kwargs["pseudo_rows"]
        val_idx = np.asarray(kwargs["val_idx"], dtype=np.int64)
        labels = np.asarray(kwargs["labels"], dtype=np.int64)
        class_names = list(kwargs["class_names"])
        calls.append(
            {
                "train_idx": np.asarray(kwargs["train_idx"], dtype=np.int64).tolist(),
                "val_idx": val_idx.tolist(),
                "pseudo_rows": list(pseudo_rows or []),
            }
        )
        probs = np.full((len(val_idx), len(class_names)), 0.05, dtype=np.float32)
        probs[np.arange(len(val_idx)), labels[val_idx]] = 0.95
        return {
            "summary": {
                "fold": int(kwargs["fold_idx"]),
                "experiment": kwargs["experiment_cfg"]["name"],
                "mode": "fusion_peft_pseudo_cv",
                "val_macro_f1": 1.0,
                "val_balanced_accuracy": 1.0,
                "val_selection_score": 1.0,
                "num_pseudo": len(pseudo_rows or []),
                "pseudo_training_enabled": pseudo_rows is not None,
            },
            "val_probs": probs,
            "features": None,
        }

    monkeypatch.setattr(fusion_peft_module, "train_fusion_fold", fake_train_fusion_fold)
    try:
        result = fusion_peft_module.run_fusion_peft_cv(
            config_path,
            pseudolabels=pseudo_csv,
            experiment_names=["b1_uni_conch_adapters_all4_concat_p32_reg"],
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    assert len(calls) == 2
    assert all(len(call["pseudo_rows"]) == 1 for call in calls)
    assert all("test_0.png" == call["pseudo_rows"][0]["filename"] for call in calls)
    assert all(set(call["train_idx"]).isdisjoint(call["val_idx"]) for call in calls)
    best = result["summary"]["best"]
    assert best["mode"] == "fusion_peft_pseudo_cv"
    assert best["validation_only"] is True
    assert best["test_path"] is None
    assert best["test_prediction_kind"] is None
    assert best["num_pseudo"] == 1
    assert best["pseudo_counts"] == {"Class_0": 0, "Class_1": 1}


def test_pseudo_lora_single_refit_writes_one_model_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, color in {
        "train_0.png": (255, 0, 0),
        "train_1.png": (200, 0, 0),
        "train_2.png": (0, 0, 255),
        "train_3.png": (0, 0, 200),
        "test_0.png": (128, 0, 128),
    }.items():
        Image.new("RGB", (16, 16), color=color).save(tmp_path / name)
    train_manifest = tmp_path / "train_manifest.csv"
    test_manifest = tmp_path / "test_manifest.csv"
    pseudo_csv = tmp_path / "selected_pseudolabels.csv"
    train_manifest.write_text(
        "\n".join(
            [
                "filename,label,abs_path",
                f"train_0.png,Class_0,{tmp_path / 'train_0.png'}",
                f"train_1.png,Class_0,{tmp_path / 'train_1.png'}",
                f"train_2.png,Class_1,{tmp_path / 'train_2.png'}",
                f"train_3.png,Class_1,{tmp_path / 'train_3.png'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test_manifest.write_text(
        f"filename,abs_path\ntest_0.png,{tmp_path / 'test_0.png'}\n",
        encoding="utf-8",
    )
    pseudo_csv.write_text(
        "filename,pseudo_label,sample_weight,soft_label_0,soft_label_1\n"
        "test_0.png,Class_1,0.8,0.1,0.9\n",
        encoding="utf-8",
    )
    run_name = "pytest_pseudo_lora_single_refit"
    run_dir = project_root() / "runs" / "scheme_03" / run_name
    shutil.rmtree(run_dir, ignore_errors=True)
    config = {
        "run_name": run_name,
        "seed": 2026,
        "device": "cpu",
        "epochs": 1,
        "lambda_pseudo": 0.001,
        "num_workers": 0,
        "augmentation": {"resize": 16},
        "smoke_backbone": {"key": "tiny_cnn", "backend": "tiny_cnn"},
        "smoke_experiment": {
            "name": "pseudo_lora_smoke",
            "mode": "lora",
            "epochs": 1,
            "batch_size": 2,
            "target_policy": "all_linear_except_head",
            "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "modules_to_save": ["classifier"]},
        },
    }
    config_path = tmp_path / "pseudo_lora_single_refit.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr("ml_final.training.pseudo_peft.inject_lora", lambda model, _cfg, _targets: model)
    try:
        result = run_pseudo_lora_single_refit(
            config_path,
            selected_peft=tmp_path / "unused_selected.json",
            pseudolabels=pseudo_csv,
            train_manifest=train_manifest,
            test_manifest=test_manifest,
            run_name=run_name,
            smoke=True,
        )
        summary = result["summary"]
        assert summary["mode"] == "pseudo_lora_single_refit"
        assert summary["test_prediction_kind"] == "single_refit"
        assert summary["model_count"] == 1
        assert summary["num_true"] == 4
        assert summary["num_pseudo"] == 1
        assert Path(summary["submission"]).exists()
        payload = np.load(project_root() / summary["test_path"], allow_pickle=True)
        try:
            assert str(payload["prediction_kind"]) == "single_refit"
            assert int(payload["model_count"]) == 1
            assert payload["probs"].shape == (1, 2)
        finally:
            payload.close()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_single_refit_epoch_source_uses_cv_best_epoch_median(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "run" / "metrics" / "exp"
    metrics_dir.mkdir(parents=True)
    best_epochs = [3, 5, 4]
    for fold_idx, best_epoch in enumerate(best_epochs):
        events = []
        for epoch in range(1, 7):
            score = 0.9 if epoch == best_epoch else 0.1 + epoch * 0.01
            events.append({"epoch": epoch, "val": {"macro_f1": score, "selection_score": score}})
        (metrics_dir / f"fold_{fold_idx}_events.jsonl").write_text(
            "\n".join(json.dumps(item) for item in events) + "\n",
            encoding="utf-8",
        )

    epochs, source = resolve_single_refit_epochs(
        {"refit_epoch_source": str(tmp_path / "run" / "metrics" / "summary.json")},
        configured_epochs=40,
    )

    assert epochs == 4
    assert source["mode"] == "cv_best_epoch_median"
    assert source["configured_epochs"] == 40
    assert source["fold_best_epochs"] == best_epochs


def test_infer_final_writes_required_submission_schema(tmp_path: Path):
    test_path = tmp_path / "test_single_refit.npz"
    np.savez_compressed(
        test_path,
        probs=np.asarray([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32),
        class_names=np.asarray(["Class_0", "Class_1"], dtype=object),
        test_filenames=np.asarray(["test_00001.png", "test_00002.png"], dtype=object),
    )
    config = {
        "run_name": "pytest_submission",
        "test_prediction_files": [str(test_path)],
        "allow_prediction_search": False,
        "submission_dir": str(tmp_path / "submission"),
    }
    config_path = tmp_path / "submission.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = infer_final(config_path)

    text = Path(result["submission"]).read_text(encoding="utf-8").splitlines()
    assert text == ["filename,label", "test_00001.png,Class_1", "test_00002.png,Class_0"]


def test_validate_submission_reports_prediction_and_backup_delta(tmp_path: Path):
    manifest = tmp_path / "test_manifest.csv"
    manifest.write_text(
        "filename,abs_path\n"
        f"test_00001.png,{tmp_path / 'a.png'}\n"
        f"test_00002.png,{tmp_path / 'b.png'}\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    submission.write_text("filename,label\ntest_00001.png,Class_1\ntest_00002.png,Class_0\n", encoding="utf-8")
    backup = tmp_path / "backup.csv"
    backup.write_text("filename,label\ntest_00001.png,Class_1\ntest_00002.png,Class_1\n", encoding="utf-8")
    prediction = tmp_path / "test_single_refit.npz"
    np.savez_compressed(
        prediction,
        probs=np.asarray([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32),
        class_names=np.asarray(["Class_0", "Class_1"], dtype=object),
        test_filenames=np.asarray(["test_00001.png", "test_00002.png"], dtype=object),
        prediction_kind=np.asarray("single_refit", dtype=object),
        model_count=np.asarray(1, dtype=np.int64),
    )

    result = validate_submission(
        submission,
        test_manifest=manifest,
        test_prediction=prediction,
        compare_submission=backup,
        expected_rows=2,
        class_names=["Class_0", "Class_1"],
    )

    assert result["class_distribution"] == {"Class_0": 1, "Class_1": 1}
    assert result["prediction"]["prediction_kind"] == "single_refit"
    assert result["prediction"]["model_count"] == 1
    assert result["prediction"]["top1_confidence"]["mean"] == pytest.approx(0.8)
    assert result["comparison"]["changed_labels"] == 1


def test_class_bias_calibration_and_apply(tmp_path: Path):
    oof_path = tmp_path / "oof.npz"
    probs = np.asarray(
        [
            [0.60, 0.40],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.40, 0.60],
        ],
        dtype=np.float32,
    )
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    np.savez_compressed(
        oof_path,
        probs=probs,
        y_true=y_true,
        y_pred=probs.argmax(axis=1),
        class_names=np.asarray(["Class_0", "Class_1"], dtype=object),
        train_filenames=np.asarray([f"train_{idx}.png" for idx in range(4)], dtype=object),
    )
    config = {
        "oof_prediction": str(oof_path),
        "out_dir": str(tmp_path / "bias"),
        "n_splits": 2,
        "seed": 2026,
        "bias_grid": {
            "Class_0": [-0.2, 0.0, 0.2],
            "Class_1": [-0.2, 0.0, 0.2],
        },
    }
    config_path = tmp_path / "bias.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = calibrate_class_bias(config_path)

    assert Path(result["artifacts"]["bias"]).exists()
    assert Path(result["artifacts"]["nested_oof"]).exists()
    assert result["temperature_grid"] == [1.0]
    assert result["all_oof_fit"]["temperature"] == 1.0
    out_path = tmp_path / "biased_test.npz"
    applied = apply_class_bias(oof_path, bias=result["artifacts"]["bias"], out=out_path)
    assert Path(applied["out"]).exists()
    loaded = np.load(out_path, allow_pickle=True)
    try:
        assert loaded["probs"].shape == probs.shape
        assert "class_bias" in loaded.files
        assert float(loaded["class_bias_temperature"]) == 1.0
    finally:
        loaded.close()


def test_class_bias_calibration_accepts_temperature_grid(tmp_path: Path):
    oof_path = tmp_path / "oof_temp.npz"
    probs = np.asarray(
        [
            [0.52, 0.48],
            [0.51, 0.49],
            [0.49, 0.51],
            [0.48, 0.52],
        ],
        dtype=np.float32,
    )
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int64)
    np.savez_compressed(
        oof_path,
        probs=probs,
        y_true=y_true,
        y_pred=probs.argmax(axis=1),
        class_names=np.asarray(["Class_0", "Class_1"], dtype=object),
        train_filenames=np.asarray([f"train_{idx}.png" for idx in range(4)], dtype=object),
    )
    config = {
        "oof_prediction": str(oof_path),
        "out_dir": str(tmp_path / "bias_temp"),
        "n_splits": 2,
        "seed": 2026,
        "temperature_grid": [0.7, 1.0, 1.3],
        "bias_grid": {
            "Class_0": [-0.1, 0.0, 0.1],
            "Class_1": [-0.1, 0.0, 0.1],
        },
    }
    config_path = tmp_path / "bias_temp.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = calibrate_class_bias(config_path)

    assert result["temperature_grid"] == [0.7, 1.0, 1.3]
    assert result["all_oof_fit"]["temperature"] in result["temperature_grid"]
    assert len(result["nested"]["fold_temperatures"]) == 2


def make_prediction_files(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(2026)
    y_true = np.arange(25, dtype=np.int64) % len(CLASS_NAMES)
    logits = rng.normal(0, 0.25, size=(25, len(CLASS_NAMES)))
    logits[np.arange(25), y_true] += 2.0
    probs = softmax(logits)
    oof_path = tmp_path / "oof_model.npz"
    np.savez_compressed(
        oof_path,
        probs=probs,
        y_true=y_true,
        y_pred=probs.argmax(axis=1),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        train_filenames=np.asarray([f"train_{idx:05d}.png" for idx in range(25)], dtype=object),
    )
    test_logits = rng.normal(0, 0.25, size=(8, len(CLASS_NAMES)))
    test_logits[np.arange(8), np.arange(8) % len(CLASS_NAMES)] += 2.0
    test_path = tmp_path / "test_model.npz"
    np.savez_compressed(
        test_path,
        probs=softmax(test_logits),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        test_filenames=np.asarray([f"test_smoke_{idx:05d}.png" for idx in range(8)], dtype=object),
    )
    return {"oof": oof_path, "test": test_path}


def make_teacher_csv_for_fixed_policy(tmp_path: Path) -> Path:
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    class_names = ["Class_0", "Class_1"]
    rows = [
        teacher_row("a.png", "Class_0", "Class_0", [0.9, 0.1]),
        teacher_row("b.png", "Class_0", "Class_0", [0.88, 0.12]),
        teacher_row("c.png", "Class_0", "Class_1", [0.1, 0.9]),
        teacher_row("d.png", "Class_1", "Class_1", [0.1, 0.9]),
    ]
    write_teacher_prediction_csv(teacher_dir / "teacher_oof_predictions.csv", rows, class_names, include_truth=True)
    return teacher_dir


def make_tiny_pseudo_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    for name, color in {
        "train_a.png": (255, 0, 0),
        "train_b.png": (0, 0, 255),
        "test_a.png": (200, 0, 0),
        "test_b.png": (0, 0, 200),
    }.items():
        Image.new("RGB", (8, 8), color=color).save(tmp_path / name)
    train_manifest = tmp_path / "train.csv"
    test_manifest = tmp_path / "test.csv"
    pseudo_csv = tmp_path / "pseudo.csv"
    train_manifest.write_text(
        "filename,label,abs_path\n"
        f"train_a.png,Class_0,{tmp_path / 'train_a.png'}\n"
        f"train_b.png,Class_1,{tmp_path / 'train_b.png'}\n",
        encoding="utf-8",
    )
    test_manifest.write_text(
        "filename,abs_path\n"
        f"test_a.png,{tmp_path / 'test_a.png'}\n"
        f"test_b.png,{tmp_path / 'test_b.png'}\n",
        encoding="utf-8",
    )
    pseudo_csv.write_text(
        "filename,pseudo_label,sample_weight,soft_label_0,soft_label_1\n"
        "test_a.png,Class_0,0.8,0.9,0.1\n"
        "test_b.png,Class_1,0.5,0.2,0.8\n",
        encoding="utf-8",
    )
    return train_manifest, test_manifest, pseudo_csv


def teacher_row(filename: str, true_label: str, pred_label: str, probs: list[float]) -> dict[str, object]:
    probs_array = np.asarray(probs, dtype=np.float64)
    probs_array = probs_array / probs_array.sum()
    order = np.argsort(-probs_array)
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    return {
        "filename": filename,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob_top1": float(probs_array[top1]),
        "prob_top2": float(probs_array[top2]),
        "margin": float(probs_array[top1] - probs_array[top2]),
        "entropy": float(-(probs_array * np.log(np.clip(probs_array, 1e-12, 1.0))).sum()),
        "teacher_agreement": 1.0,
        "correct": int(true_label == pred_label),
        "probs": probs_array,
    }


def write_teacher_prediction_csv(
    path: Path,
    rows: list[dict[str, object]],
    class_names: list[str],
    *,
    include_truth: bool,
) -> None:
    fields = ["filename", "pred_label", "prob_top1", "prob_top2", "margin", "entropy", "teacher_agreement"]
    if include_truth:
        fields += ["true_label", "correct"]
    fields += [f"prob_{idx}" for idx in range(len(class_names))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row[field] for field in fields if not (field.startswith("prob_") and field[5:].isdigit())}
            probs = np.asarray(row["probs"], dtype=np.float64)
            for idx in range(len(class_names)):
                out[f"prob_{idx}"] = float(probs[idx])
            writer.writerow(out)


def make_feature_bundle(path: Path) -> None:
    rng = np.random.default_rng(7)
    train_labels = np.arange(25, dtype=np.int64) % len(CLASS_NAMES)
    train_features = rng.normal(size=(25, 6)).astype(np.float32)
    for idx, label in enumerate(train_labels):
        train_features[idx, label % 5] += 2.0
    test_features = rng.normal(size=(8, 6)).astype(np.float32)
    save_feature_bundle(
        path,
        train_features=train_features,
        train_labels=train_labels,
        train_filenames=np.asarray([f"train_{idx:05d}.png" for idx in range(25)], dtype=object),
        test_features=test_features,
        test_filenames=np.asarray([f"test_smoke_{idx:05d}.png" for idx in range(8)], dtype=object),
        class_names=CLASS_NAMES,
        metadata={"test": "scheme03"},
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
