"""Tests for Scheme 01 frozen-head ensemble behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml_final.backbones.factory import process_timm_output
from ml_final.features.extract import build_extractor, extract_rows, extract_train_rows, preprocess_feature_view
from ml_final.features.fuse import run_feature_fusion
from ml_final.features.store import load_feature_bundle, save_feature_bundle
from ml_final.heads.classical_heads import augment_training_features, build_oof_ensemble, fit_head, run_frozen_cv


CLASS_NAMES = [f"Class_{idx}" for idx in range(5)]


def test_oof_ensemble_uses_top_k_heads(tmp_path):
    """The ensemble must not average every weak head by default."""

    y_true = np.arange(10, dtype=np.int64) % len(CLASS_NAMES)
    predictions_dir = tmp_path / "predictions"
    metrics_dir = tmp_path / "metrics"
    predictions_dir.mkdir()
    metrics_dir.mkdir()

    strong_probs = make_probs(y_true, confidence=0.92)
    weak_probs = make_probs((y_true + 1) % len(CLASS_NAMES), confidence=0.92)
    np.savez_compressed(
        predictions_dir / "oof_strong_head.npz",
        probs=strong_probs,
        y_true=y_true,
        y_pred=strong_probs.argmax(axis=1),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
    )
    np.savez_compressed(
        predictions_dir / "oof_weak_head.npz",
        probs=weak_probs,
        y_true=y_true,
        y_pred=weak_probs.argmax(axis=1),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
    )
    results = [
        {
            "head_id": "strong_head",
            "metrics": {"selection_score": 1.0},
        },
        {
            "head_id": "weak_head",
            "metrics": {"selection_score": 0.0},
        },
    ]

    ensemble = build_oof_ensemble(
        results,
        metrics_dir,
        predictions_dir,
        config={"ensemble": {"top_k": 1}},
    )

    assert ensemble["method"] == "top_k_simple_average"
    assert ensemble["source_head_ids"] == ["strong_head"]
    assert ensemble["metrics"]["macro_f1"] == 1.0


def test_virchow2_feature_policy_uses_cls_and_mean_patch_tokens():
    tokens = torch.arange(2 * 261 * 1280, dtype=torch.float32).reshape(2, 261, 1280)

    features = process_timm_output(tokens, "virchow2_official")

    expected = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=-1)
    assert features.shape == (2, 2560)
    assert torch.equal(features, expected)


def test_extract_rows_batches_model_images_across_rows(tmp_path: Path):
    rows = make_image_rows(tmp_path, count=3)
    extractor = RecordingExtractor()

    features = extract_rows(rows, extractor, input_size=32, scale_to=None, tta="none", color_tta=None, batch_size=2)

    assert features.shape == (3, 2)
    assert extractor.calls == [(3, 2)]


def test_preprocess_feature_view_can_center_pad_scale_view() -> None:
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))

    padded = preprocess_feature_view(image, input_size=32, scale_to=16)

    assert padded.size == (32, 32)
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert padded.getpixel((16, 16)) == (10, 20, 30)


def test_preprocess_feature_view_preserves_default_resize() -> None:
    image = Image.new("RGB", (16, 16), color=(10, 20, 30))

    resized = preprocess_feature_view(image, input_size=32, scale_to=None)

    assert resized.size == (32, 32)
    assert resized.getpixel((16, 16)) == (10, 20, 30)


def test_extract_train_rows_batches_expanded_views_across_rows(tmp_path: Path):
    rows = make_image_rows(tmp_path, count=2)
    extractor = RecordingExtractor()

    features, origins, origin_indices = extract_train_rows(
        rows,
        extractor,
        input_size=32,
        scale_to=None,
        tta="none",
        train_expand="geom6",
        color_tta=None,
        batch_size=4,
    )

    assert features.shape == (12, 2)
    assert extractor.calls == [(12, 4)]
    assert origins.tolist() == ["img_0.png"] * 6 + ["img_1.png"] * 6
    assert origin_indices.tolist() == [0] * 6 + [1] * 6


def test_cell_stats_extractor_is_deterministic(tmp_path: Path):
    rows = make_image_rows(tmp_path, count=2)
    extractor = build_extractor({"backend": "cell_stats"})

    first = extract_rows(rows, extractor, input_size=32, scale_to=None, tta="none", color_tta=None, batch_size=2)
    second = extract_rows(rows, extractor, input_size=32, scale_to=None, tta="none", color_tta=None, batch_size=2)

    assert first.shape == (2, 62)
    assert np.allclose(first, second)


def test_feature_fusion_balances_blocks_and_preserves_alignment(tmp_path: Path):
    source_a = tmp_path / "a.npz"
    source_b = tmp_path / "b.npz"
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    filenames = np.asarray(["a.png", "b.png", "c.png"], dtype=object)
    save_feature_bundle(
        source_a,
        train_features=np.asarray([[10.0, 0.0], [0.0, 5.0], [3.0, 4.0]], dtype=np.float32),
        train_labels=labels,
        train_filenames=filenames,
        class_names=CLASS_NAMES,
        metadata={"source": "a"},
    )
    save_feature_bundle(
        source_b,
        train_features=np.asarray([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 8.0]], dtype=np.float32),
        train_labels=labels,
        train_filenames=filenames,
        class_names=CLASS_NAMES,
        metadata={"source": "b"},
    )
    config = tmp_path / "fuse.yaml"
    config.write_text(
        f"""
run_name: fused
fusion_id: ab
output_dir: {tmp_path.as_posix()}
sources:
  - name: a
    path: {source_a.as_posix()}
    weight: 1.0
  - name: b
    path: {source_b.as_posix()}
    weight: 1.0
""",
        encoding="utf-8",
    )

    result = run_feature_fusion(config)

    bundle = load_feature_bundle(tmp_path / "fused" / "ab" / "ab_fused_features.npz")
    features = bundle["train_features"]
    assert result["feature"]["feature_dim"] == 5
    assert features.shape == (3, 5)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert bundle["metadata"]["blocks"][0]["raw_dim"] == 2
    assert bundle["metadata"]["blocks"][1]["raw_dim"] == 3


def test_feature_fusion_preserves_expanded_eval_metadata(tmp_path: Path):
    source_a = tmp_path / "a_expanded.npz"
    source_b = tmp_path / "b_expanded.npz"
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    filenames = np.asarray(["a.png__aug00", "a.png__aug01", "b.png__aug00", "b.png__aug01"], dtype=object)
    origin_filenames = np.asarray(["a.png", "a.png", "b.png", "b.png"], dtype=object)
    origin_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    save_feature_bundle(
        source_a,
        train_features=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        train_labels=labels,
        train_filenames=filenames,
        class_names=CLASS_NAMES,
        metadata={"source": "a"},
        train_origin_filenames=origin_filenames,
        train_origin_indices=origin_indices,
        train_eval_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    save_feature_bundle(
        source_b,
        train_features=np.asarray([[2.0, 0.0, 0.0], [1.8, 0.2, 0.0], [0.0, 0.0, 2.0], [0.0, 0.2, 1.8]], dtype=np.float32),
        train_labels=labels,
        train_filenames=filenames,
        class_names=CLASS_NAMES,
        metadata={"source": "b"},
        train_origin_filenames=origin_filenames,
        train_origin_indices=origin_indices,
        train_eval_features=np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32),
    )
    config = tmp_path / "fuse_expanded.yaml"
    config.write_text(
        f"""
run_name: fused
fusion_id: ab_expanded
output_dir: {tmp_path.as_posix()}
sources:
  - name: a
    path: {source_a.as_posix()}
    weight: 1.0
  - name: b
    path: {source_b.as_posix()}
    weight: 1.0
""",
        encoding="utf-8",
    )

    result = run_feature_fusion(config)

    bundle = load_feature_bundle(tmp_path / "fused" / "ab_expanded" / "ab_expanded_fused_features.npz")
    assert result["feature"]["train_count"] == 4
    assert result["feature"]["eval_train_count"] == 2
    assert bundle["train_features"].shape == (4, 5)
    assert bundle["train_eval_features"].shape == (2, 5)
    assert bundle["train_origin_filenames"].tolist() == origin_filenames.tolist()
    assert bundle["train_origin_indices"].tolist() == origin_indices.tolist()
    assert bundle["metadata"]["train_expand_preserved"] is True


def test_run_frozen_cv_can_disable_prediction_ensemble(tmp_path: Path):
    feature_dir = tmp_path / "features" / "fused"
    feature_path = feature_dir / "fused_features.npz"
    y = np.arange(25, dtype=np.int64) % len(CLASS_NAMES)
    save_feature_bundle(
        feature_path,
        train_features=np.eye(25, dtype=np.float32),
        train_labels=y,
        train_filenames=np.asarray([f"img_{idx}.png" for idx in range(25)], dtype=object),
        class_names=CLASS_NAMES,
        metadata={"test": True},
    )
    config = tmp_path / "cv.yaml"
    config.write_text(
        f"""
run_name: fused_cv
features_dir: {feature_dir.as_posix()}
run_dir: {tmp_path / "runs"}
seed: 2026
n_splits: 5
skip_completed: false
ensemble:
  enabled: false
heads:
  - name: prototype_center_l2
    family: prototype
    center: true
    normalize: true
""",
        encoding="utf-8",
    )

    result = run_frozen_cv(config)

    assert result["summary"]["ensemble"] == {}
    predictions_dir = tmp_path / "runs" / "predictions"
    assert not (predictions_dir / "oof_simple_average_ensemble.npz").exists()


def test_frofa_feature_augmentation_targets_configured_classes():
    X = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    y = np.asarray([0, 1, 2, 3, 4, 2], dtype=np.int64)
    spec = {
        "train_augmentation": {
            "type": "frofa_brightness_c2",
            "target_classes": [2, 3, 4],
            "copies": 2,
            "level": 0.1,
        }
    }

    X_aug, y_aug = augment_training_features(X, y, spec, seed=123)

    assert X_aug.shape == (14, 4)
    assert y_aug.tolist() == y.tolist() + [2, 3, 4, 2] + [2, 3, 4, 2]
    assert np.array_equal(X_aug[: len(X)], X)
    assert not np.array_equal(X_aug[len(X) : len(X) + 4], X[np.isin(y, [2, 3, 4])])


def test_structured_heads_return_valid_probabilities():
    rng = np.random.default_rng(2026)
    y = np.arange(50, dtype=np.int64) % len(CLASS_NAMES)
    X = rng.normal(size=(50, 12)).astype(np.float32)
    X += y[:, None].astype(np.float32) * 0.2

    pca_model = fit_head(X, y, {"family": "pca_logreg", "n_components": 4, "C": 1.0}, seed=7)
    pca_probs = pca_model.predict_proba(X[:6])
    assert pca_probs.shape == (6, len(CLASS_NAMES))
    assert np.allclose(pca_probs.sum(axis=1), 1.0)

    hard_model = fit_head(
        X,
        y,
        {"family": "hierarchical_hard", "hard_classes": [2, 3, 4], "base_C": 1.0, "gate_C": 1.0, "hard_C": 1.0},
        seed=7,
    )
    hard_probs = hard_model.predict_proba(X[:6])
    assert hard_probs.shape == (6, len(CLASS_NAMES))
    assert np.allclose(hard_probs.sum(axis=1), 1.0)

    bias_model = fit_head(
        X,
        y,
        {"family": "bias_tuned_logreg", "target_classes": [2, 3, 4], "bias_values": [-0.2, 0.0, 0.2], "C": 1.0},
        seed=7,
    )
    bias_probs = bias_model.predict_proba(X[:6])
    assert bias_probs.shape == (6, len(CLASS_NAMES))
    assert np.allclose(bias_probs.sum(axis=1), 1.0)
    assert bias_model.bias_.shape == (len(CLASS_NAMES),)


def make_probs(labels: np.ndarray, *, confidence: float) -> np.ndarray:
    probs = np.full((len(labels), len(CLASS_NAMES)), (1.0 - confidence) / (len(CLASS_NAMES) - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def make_image_rows(tmp_path: Path, *, count: int) -> list[dict[str, str]]:
    rows = []
    for idx in range(count):
        path = tmp_path / f"img_{idx}.png"
        Image.new("RGB", (8, 8), color=(idx, idx + 1, idx + 2)).save(path)
        rows.append({"filename": path.name, "abs_path": str(path), "label": f"Class_{idx % 5}"})
    return rows


class RecordingExtractor:
    uses_model_transform = True

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def extract_images(self, images: list[Image.Image], *, batch_size: int) -> np.ndarray:
        self.calls.append((len(images), batch_size))
        return np.arange(len(images) * 2, dtype=np.float32).reshape(len(images), 2)
