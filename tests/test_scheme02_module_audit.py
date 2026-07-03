from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from ml_final.backbones.factory import (
    ConchImageBackbone,
    build_classifier,
    freeze_all_except_trainable_adapters_and_head,
    merge_model_overrides,
)
from ml_final.backbones.module_audit import audit_modules, select_lora_targets
from ml_final.training import adapted_heads as adapted_heads_module
from ml_final.training import fusion_peft as fusion_peft_module
from ml_final.training.data import CenterScalePad, build_pil_augmentation, build_transform
from ml_final.training.fusion_peft import (
    MultiEncoderFusionClassifier,
    build_fusion_classifier_head,
    build_fusion_optimizer,
    build_fusion_model,
    evaluate_fusion_feature_heads,
    evaluate_fusion_model,
    filter_experiments,
    fuse_raw_feature_blocks,
    maybe_mixup_images_by_branch,
    predict_fusion_rows_feature_tta,
    resolve_checkpoint_metric,
    resolve_class_weight_tensor,
    resolve_eval_feature_strategies,
    resolve_focal_loss,
    resolve_fusion_branch_configs,
    resolve_fusion_experiment,
    resolve_branch_adapter_checkpoint,
    slice_cached_fused_features,
    train_fusion_epoch,
    train_fusion_pseudo_epoch,
)
from ml_final.training.peft_train import (
    dataloader_performance_kwargs,
    experiment_applies_to_backbone,
    inject_lora,
    merge_transform_config,
    resolve_class_weight_tensor as resolve_peft_class_weight_tensor,
    resolve_test_prediction_mode,
    train_epoch,
)
from ml_final.utils.config import load_yaml


class TinyAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)
        self.attn.proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(8, 16)
        self.mlp.fc2 = nn.Linear(16, 8)
        self.classifier = nn.Linear(8, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def test_select_lora_targets_attention_qkv() -> None:
    names = ["blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.0.mlp.fc1", "classifier"]
    assert select_lora_targets(names, policy="attention_qkv") == ["blocks.0.attn.qkv"]


def test_resolve_fusion_branch_configs_applies_opt_in_branch_overrides() -> None:
    config = {
        "branch_overrides": {"conch": {"input_size": 224}},
        "branches": [
            {"key": "uni2_h", "input_size": 224, "model_kwargs": {"dynamic_img_size": True}},
            {"key": "conch", "input_size": 224},
        ]
    }
    experiment = {
        "branch_keys": ["uni2_h", "conch"],
        "branch_overrides": {
            "uni2_h": {
                "input_size": 336,
                "scale_to": 112,
                "model_kwargs": {"img_size": 224},
            }
        },
    }

    resolved = resolve_fusion_branch_configs(config, experiment)

    assert resolved[0]["input_size"] == 336
    assert resolved[0]["scale_to"] == 112
    assert resolved[0]["model_kwargs"] == {"dynamic_img_size": True, "img_size": 224}
    assert resolved[1] == {"key": "conch", "input_size": 224}
    assert config["branches"][0]["input_size"] == 224


def test_build_fusion_optimizer_can_use_adapter_lr_group() -> None:
    branches = {"a": TinyAttentionBlock(), "b": TinyAttentionBlock()}
    model = MultiEncoderFusionClassifier(
        branches,
        {"a": 5, "b": 5},
        projection_dim=4,
        num_classes=5,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.branches["a"].attn.qkv.weight.requires_grad = True
    model.projections["a"].weight.requires_grad = True

    optimizer = build_fusion_optimizer(
        model,
        {"lr": 1e-3, "adapter_lr": 2e-5, "weight_decay": 0.01},
        {},
    )

    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 2e-5]
    assert id(model.projections["a"].weight) in {id(param) for param in optimizer.param_groups[0]["params"]}
    assert id(model.branches["a"].attn.qkv.weight) in {id(param) for param in optimizer.param_groups[1]["params"]}


def test_build_fusion_classifier_head_preserves_linear_default() -> None:
    linear = build_fusion_classifier_head(16, 5, hidden_dim=0)
    hidden = build_fusion_classifier_head(16, 5, hidden_dim=8)

    assert isinstance(linear, nn.Linear)
    assert isinstance(hidden, nn.Sequential)
    assert isinstance(hidden[0], nn.Linear)
    assert hidden[0].out_features == 8


def test_single_classifier_head_preserves_linear_default_and_supports_mlp() -> None:
    base_cfg = {"key": "tiny", "backend": "tiny_cnn", "feature_dim": 16, "input_size": 32}
    linear_model, _ = build_classifier(base_cfg, num_classes=5)
    mlp_model, _ = build_classifier(
        {
            **base_cfg,
            "classifier_head": {"type": "mlp", "hidden_dim": 8, "dropout": 0.1, "layer_norm": True},
        },
        num_classes=5,
    )

    assert isinstance(linear_model.classifier, nn.Linear)
    assert isinstance(mlp_model.classifier, nn.Sequential)
    assert isinstance(mlp_model.classifier[0], nn.LayerNorm)
    assert isinstance(mlp_model.classifier[1], nn.Linear)
    assert mlp_model.classifier[1].out_features == 8


def test_image_adapter_is_opt_in_and_trainable_after_lora_freeze() -> None:
    model, _ = build_classifier(
        {
            "key": "tiny",
            "backend": "tiny_cnn",
            "feature_dim": 16,
            "input_size": 32,
            "image_adapter": {"type": "residual_conv", "hidden_channels": 4, "residual_scale": 0.1},
        },
        num_classes=5,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name

    freeze_all_except_trainable_adapters_and_head(model)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert "image_adapter.net.0.weight" in trainable
    assert "classifier.weight" in trainable
    assert "backbone.net.0.weight" not in trainable


def test_experiment_model_overrides_are_opt_in() -> None:
    backbone_cfg = {"key": "tiny", "backend": "tiny_cnn", "feature_dim": 16}
    experiment_cfg = {
        "classifier_head": {"type": "mlp", "hidden_dim": 8},
        "image_adapter": {"type": "residual_conv", "hidden_channels": 4},
    }

    merged = merge_model_overrides(backbone_cfg, experiment_cfg)

    assert "classifier_head" not in backbone_cfg
    assert merged["classifier_head"]["hidden_dim"] == 8
    assert merged["image_adapter"]["hidden_channels"] == 4


def test_maybe_mixup_images_by_branch_is_synchronized() -> None:
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    images = {
        "a": torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3),
        "b": torch.arange(100, 112, dtype=torch.float32).reshape(4, 3),
    }

    unchanged, target = maybe_mixup_images_by_branch(images, labels, alpha=0.0, probability=1.0)
    assert unchanged is images
    assert target is None

    mixed, target = maybe_mixup_images_by_branch(images, labels, alpha=0.4, probability=1.0)

    assert target is not None
    y_a, y_b, lam = target
    assert torch.equal(y_a, labels)
    assert set(y_b.tolist()) == set(labels.tolist())
    assert 0.0 <= lam <= 1.0
    assert mixed["a"].shape == images["a"].shape
    assert mixed["b"].shape == images["b"].shape


def test_random_resized_crop_is_train_only_for_project_transform() -> None:
    config = {"random_resized_crop": {"scale": [0.7, 1.0], "ratio": [0.9, 1.1]}}

    train_transform = build_transform(config, train=True, input_size=224)
    eval_transform = build_transform(config, train=False, input_size=224)

    assert any(isinstance(op, transforms.RandomResizedCrop) for op in train_transform.transforms)
    assert not any(isinstance(op, transforms.RandomResizedCrop) for op in eval_transform.transforms)
    assert isinstance(eval_transform.transforms[0], transforms.Resize)


def test_merge_transform_config_passes_opt_in_scale_pad_from_branch() -> None:
    merged = merge_transform_config(
        {"mean": [0.1, 0.2, 0.3]},
        {"scale_to": 80, "scale_pad_fill_rgb": [255, 255, 255]},
        pretrained_cfg={"crop_pct": 1.0},
    )

    assert merged["mean"] == [0.1, 0.2, 0.3]
    assert merged["crop_pct"] == 1.0
    assert merged["scale_to"] == 80
    assert merged["scale_pad_fill_rgb"] == [255, 255, 255]


def test_resolve_peft_class_weight_tensor_is_opt_in_and_normalized() -> None:
    class_names = ["Class_0", "Class_1", "Class_2"]

    assert resolve_peft_class_weight_tensor({}, {}, class_names=class_names) is None

    weights = resolve_peft_class_weight_tensor(
        {
            "class_weighting": {
                "mode": "manual",
                "weights": {"Class_0": 1.0, "Class_1": 1.0, "Class_2": 2.0},
            }
        },
        {},
        class_names=class_names,
    )

    assert weights is not None
    expected = torch.tensor([0.75, 0.75, 1.5])
    assert torch.allclose(weights, expected)


def test_scale_to_center_pad_is_opt_in_for_project_transform() -> None:
    transform = build_transform(
        {"scale_to": 16, "scale_pad_fill_rgb": [255, 255, 255], "mean": [0, 0, 0], "std": [1, 1, 1]},
        train=False,
        input_size=32,
    )

    assert isinstance(transform.transforms[0], CenterScalePad)
    padded = transform.transforms[0](Image.new("RGB", (4, 4), color=(10, 20, 30)))

    assert padded.size == (32, 32)
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert padded.getpixel((16, 16)) == (10, 20, 30)


def test_scale_to_center_pad_can_precede_official_preprocess_on_eval() -> None:
    pil_aug = build_pil_augmentation(
        {"scale_to": 16, "input_size": 32, "scale_pad_fill_rgb": [255, 255, 255]},
        train=False,
    )

    assert pil_aug is not None
    assert isinstance(pil_aug.transforms[0], CenterScalePad)


def test_random_resized_crop_can_precede_official_preprocess() -> None:
    config = {"random_resized_crop": {"size": 224, "scale": [0.7, 1.0], "ratio": [0.9, 1.1]}}

    pil_aug = build_pil_augmentation(config, train=True)

    assert pil_aug is not None
    assert any(isinstance(op, transforms.RandomResizedCrop) for op in pil_aug.transforms)


def test_select_lora_targets_attention_qkv_proj() -> None:
    names = ["blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.0.mlp.fc1", "classifier"]
    assert select_lora_targets(names, policy="attention_qkv_proj") == [
        "blocks.0.attn.proj",
        "blocks.0.attn.qkv",
    ]


def test_audit_modules_reports_linear_targets() -> None:
    audit = audit_modules(TinyAttentionBlock(), target_policy="attention_qkv_proj")
    assert "attn.qkv" in audit.selected_lora_targets
    assert "attn.proj" in audit.selected_lora_targets
    assert all("classifier" not in name for name in audit.selected_lora_targets)
    assert audit.parameter_counts["total"] > 0


def test_inject_lora_passes_supported_peft_options(monkeypatch) -> None:
    calls = {}

    class FakeLoraConfig:
        def __init__(
            self,
            *,
            r,
            lora_alpha,
            target_modules,
            lora_dropout,
            bias,
            modules_to_save,
            use_rslora=False,
            use_dora=False,
        ) -> None:
            calls["config"] = {
                "r": r,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
                "lora_dropout": lora_dropout,
                "bias": bias,
                "modules_to_save": modules_to_save,
                "use_rslora": use_rslora,
                "use_dora": use_dora,
            }

    def fake_get_peft_model(model, config):
        calls["model"] = model
        return model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(LoraConfig=FakeLoraConfig, get_peft_model=fake_get_peft_model),
    )
    model = TinyAttentionBlock()

    out = inject_lora(
        model,
        {
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "modules_to_save": ["classifier"],
                "use_rslora": True,
                "use_dora": False,
            }
        },
        ["attn.qkv"],
    )

    assert out is model
    assert calls["config"]["target_modules"] == ["attn.qkv"]
    assert calls["config"]["use_rslora"] is True
    assert calls["config"]["use_dora"] is False


class FakeConchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.calls = []

    def encode_image(self, x, *, proj_contrast: bool, normalize: bool):
        self.calls.append({"proj_contrast": proj_contrast, "normalize": normalize})
        return self.proj(x.mean(dim=(2, 3)))


def test_conch_image_backbone_uses_image_only_official_flags() -> None:
    fake = FakeConchModel()
    backbone = ConchImageBackbone(fake, feature_dim=4, official_preprocess=object())

    out = backbone(torch.ones(2, 3, 8, 8))

    assert out.shape == (2, 4)
    assert fake.calls == [{"proj_contrast": False, "normalize": False}]


def test_cv_mean_debug_test_predictions_are_smoke_only() -> None:
    assert resolve_test_prediction_mode({"test_prediction_mode": "cv_mean_debug"}, smoke=True) == "cv_mean_debug"
    try:
        resolve_test_prediction_mode({"test_prediction_mode": "cv_mean_debug"}, smoke=False)
    except ValueError as exc:
        assert "smoke/debug" in str(exc)
    else:
        raise AssertionError("cv_mean_debug should be rejected outside smoke/debug")


def test_train_epoch_max_batches_limits_loader() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    loader = [
        (torch.ones(3, 2), torch.zeros(3, dtype=torch.long), ["a"] * 3),
        (torch.ones(3, 2), torch.zeros(3, dtype=torch.long), ["b"] * 3),
    ]

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        None,
        criterion,
        device=torch.device("cpu"),
        amp_dtype=None,
        mixup={},
        ema=None,
        max_grad_norm=0.0,
        max_batches=1,
    )

    assert metrics["loss"] > 0


def test_experiment_backbone_filtering() -> None:
    assert experiment_applies_to_backbone({"name": "all"}, "uni2_h")
    assert experiment_applies_to_backbone({"name": "one", "backbone_keys": ["uni2_h"]}, "uni2_h")
    assert not experiment_applies_to_backbone({"name": "one", "backbone_keys": ["uni2_h"]}, "virchow2")
    assert experiment_applies_to_backbone({"name": "single", "backbone_keys": "conch"}, "conch")


def test_dataloader_performance_kwargs_are_explicit() -> None:
    assert dataloader_performance_kwargs({"persistent_workers": True}, num_workers=0, device=torch.device("cpu")) == {}
    kwargs = dataloader_performance_kwargs(
        {"persistent_workers": True, "prefetch_factor": 3},
        num_workers=2,
        device=torch.device("cpu"),
    )
    assert kwargs == {"persistent_workers": True, "prefetch_factor": 3}


def test_fuse_raw_feature_blocks_matches_scheme01_l2_policy() -> None:
    blocks = {
        "a": np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        "b": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    }

    fused = fuse_raw_feature_blocks(blocks, ["a", "b"])

    assert fused.shape == (2, 5)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), np.ones(2), atol=1e-6)

    weighted = fuse_raw_feature_blocks(blocks, ["a", "b"], branch_weights={"b": 0.5})
    assert weighted.shape == (2, 5)
    np.testing.assert_allclose(np.linalg.norm(weighted, axis=1), np.ones(2), atol=1e-6)
    assert not np.allclose(fused, weighted)


def test_slice_cached_fused_features_can_rebuild_branch_subset() -> None:
    blocks = {
        "a": np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
        "b": np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    }
    cached = fuse_raw_feature_blocks(blocks, ["a", "b"])

    only_a = slice_cached_fused_features(
        cached,
        source_branch_keys=["a", "b"],
        source_block_dims={"a": 2, "b": 3},
        selected_branch_keys=["a"],
        branch_weights={},
    )
    expected_a = fuse_raw_feature_blocks(blocks, ["a"])

    np.testing.assert_allclose(only_a, expected_a, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(only_a, axis=1), np.ones(2), atol=1e-6)


def test_fusion_feature_heads_can_write_classical_diagnostic_rows(tmp_path: Path) -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    rows = [{"filename": f"img_{idx}.png"} for idx in range(4)]
    fold_features = [
        {
            "fold_idx": 0,
            "train_idx": np.asarray([0, 2], dtype=np.int64),
            "val_idx": np.asarray([1, 3], dtype=np.int64),
            "X_train": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            "X_val": np.asarray([[0.0, 0.1], [1.0, 0.9]], dtype=np.float32),
        },
        {
            "fold_idx": 1,
            "train_idx": np.asarray([1, 3], dtype=np.int64),
            "val_idx": np.asarray([0, 2], dtype=np.int64),
            "X_train": np.asarray([[0.0, 0.1], [1.0, 0.9]], dtype=np.float32),
            "X_val": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        },
    ]

    rows_out = evaluate_fusion_feature_heads(
        head_specs=[{"name": "logreg_C1", "family": "logreg", "C": 1.0}],
        fold_features=fold_features,
        labels=labels,
        rows=rows,
        class_names=["Class_0", "Class_1"],
        predictions_dir=tmp_path,
        experiment_id="fusion_classical__toy",
        experiment_name="toy",
        seed=7,
        mode="fusion_classical_head",
    )

    assert rows_out[0]["mode"] == "fusion_classical_head"
    assert "__classical__" in rows_out[0]["experiment_id"]
    assert (tmp_path / "oof_fusion_classical__toy__classical__logreg_C1.npz").exists()


def test_adapted_head_cv_fits_only_fold_train_features(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "train_manifest.csv"
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "label", "abs_path"])
        writer.writeheader()
        for idx, label in enumerate(labels):
            writer.writerow(
                {
                    "filename": f"img_{idx}.png",
                    "label": f"Class_{label}",
                    "abs_path": str(tmp_path / f"img_{idx}.png"),
                }
            )
    config_path = tmp_path / "adapted.yaml"
    config_path.write_text(
        f"""
seed: 11
train_manifest: {manifest.as_posix()}
device: cpu
precision: fp32
n_splits: 2
batch_size: 4
num_workers: 0
backbones:
  - key: tiny
    backend: timm
    model_name: tiny
experiments:
  - name: lora_x
    mode: lora
adapted_heads:
  - name: fake_head
    family: logreg
    C: 1.0
""",
        encoding="utf-8",
    )
    source = tmp_path / "runs" / "scheme_02" / "source" / "checkpoints" / "tiny__lora_x"
    for fold_idx in range(2):
        checkpoint = source / f"fold_{fold_idx}" / "best_macro_f1.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint placeholder")

    def fake_resolve(path):
        path = Path(path)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(adapted_heads_module, "resolve_project_path", fake_resolve)
    monkeypatch.setattr(adapted_heads_module, "build_adapted_model", lambda **_kwargs: (object(), 16))
    monkeypatch.setattr(adapted_heads_module, "resolve_pretrained_cfg", lambda _model: {})
    monkeypatch.setattr(adapted_heads_module, "resolve_official_preprocess", lambda _model: None)
    monkeypatch.setattr(adapted_heads_module, "build_model_transform", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapted_heads_module, "make_feature_loader", lambda _dataset, indices, **_kwargs: np.asarray(indices))
    monkeypatch.setattr(
        adapted_heads_module,
        "extract_features",
        lambda _model, loader, **_kwargs: np.asarray(loader, dtype=np.float32).reshape(-1, 1),
    )

    fit_indices = []

    def fake_fit_head(X, y, _spec, *, seed):
        seen = set(np.asarray(X[:, 0], dtype=np.int64).tolist())
        assert np.array_equal(y, labels[list(seen)])
        fit_indices.append(seen)
        return {"n_classes": 2}

    def fake_predict_proba(_head, X):
        probs = np.full((len(X), 2), 0.5, dtype=np.float64)
        return probs

    monkeypatch.setattr(adapted_heads_module, "fit_head", fake_fit_head)
    monkeypatch.setattr(adapted_heads_module, "predict_proba", fake_predict_proba)

    adapted_heads_module.run_adapted_head_cv(
        config_path,
        source_run="source",
        experiment_id="tiny__lora_x",
        run_name="adapted_no_leakage_test",
    )

    expected_splits = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=11).split(np.zeros(len(labels)), labels))
    assert fit_indices == [set(train_idx.tolist()) for train_idx, _val_idx in expected_splits]
    for seen, (_train_idx, val_idx) in zip(fit_indices, expected_splits):
        assert seen.isdisjoint(set(val_idx.tolist()))


class TinyFeatureClassifier(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.feature_dim = dim
        self.linear = nn.Linear(3, dim)
        self.seen_training_modes = []

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_training_modes.append(self.training)
        return self.linear(x.mean(dim=(2, 3)))


def test_multi_encoder_fusion_forward_shapes() -> None:
    model = MultiEncoderFusionClassifier(
        {
            "a": TinyFeatureClassifier(4),
            "b": TinyFeatureClassifier(5),
        },
        {"a": 4, "b": 5},
        projection_dim=3,
        num_classes=5,
        composer="gated_sum",
        dropout=0.0,
    )

    logits = model({"a": torch.ones(2, 3, 8, 8), "b": torch.ones(2, 3, 8, 8)})
    _fused, aux = model.forward_features({"a": torch.ones(2, 3, 8, 8), "b": torch.ones(2, 3, 8, 8)})

    assert logits.shape == (2, 5)
    assert aux["gate_weights"].shape == (2, 2)
    assert torch.allclose(aux["gate_weights"].sum(dim=1), torch.ones(2))


def test_fusion_eval_reports_gate_means() -> None:
    model = MultiEncoderFusionClassifier(
        {
            "a": TinyFeatureClassifier(4),
            "b": TinyFeatureClassifier(5),
        },
        {"a": 4, "b": 5},
        projection_dim=3,
        num_classes=2,
        composer="gated_sum",
        dropout=0.0,
    )
    loader = [
        (
            {"a": torch.ones(2, 3, 8, 8), "b": torch.ones(2, 3, 8, 8)},
            torch.zeros(2, dtype=torch.long),
            ["x", "y"],
        )
    ]

    _metrics, probs, _pred, diagnostics = evaluate_fusion_model(
        model,
        loader,
        ["Class_0", "Class_1"],
        device=torch.device("cpu"),
        amp_dtype=None,
        max_batches=None,
    )

    assert probs.shape == (2, 2)
    assert set(diagnostics["gate_mean"]) == {"a", "b"}
    assert abs(sum(diagnostics["gate_mean"].values()) - 1.0) < 1e-6


def test_resolve_eval_feature_strategies_defaults_to_off() -> None:
    assert resolve_eval_feature_strategies({}, {}) == {}
    strategies = resolve_eval_feature_strategies(
        {"eval_feature_strategies": {"a": {"tta": "d4"}}},
        {},
    )
    assert strategies == {"a": {"tta": "d4"}}


def test_resolve_focal_loss_is_opt_in() -> None:
    assert resolve_focal_loss({}, {}) is None
    assert resolve_focal_loss({"focal_loss": {"gamma": 0.0}}, {}) is None
    assert resolve_focal_loss({"focal_loss": {"gamma": 1.0}}, {}) == {"gamma": 1.0}
    assert resolve_focal_loss({"focal_loss": {"enabled": False, "gamma": 1.0}}, {}) is None


def test_predict_fusion_rows_feature_tta_aggregates_before_head(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(128, 32, 64)).save(image_path)
    rows = [{"filename": "img.png", "abs_path": str(image_path)}]
    model = MultiEncoderFusionClassifier(
        {"a": TinyFeatureClassifier(4)},
        {"a": 4},
        projection_dim=3,
        num_classes=2,
        composer="concat_linear",
        dropout=0.0,
    )

    probs, diagnostics = predict_fusion_rows_feature_tta(
        model,
        rows,
        transforms_by_key={"a": lambda _image: torch.ones(3, 8, 8)},
        branch_feature_strategies={"a": {"tta": "d4"}},
        device=torch.device("cpu"),
        amp_dtype=None,
        batch_size=2,
        max_rows=None,
    )

    assert probs.shape == (1, 2)
    assert np.allclose(probs.sum(axis=1), np.ones(1), atol=1e-6)
    assert diagnostics["eval_feature_level_tta"] is True
    assert diagnostics["eval_feature_strategies"]["a"]["tta"] == "d4"


def test_fusion_training_keeps_frozen_branches_in_eval_mode() -> None:
    branch = TinyFeatureClassifier(4)
    model = MultiEncoderFusionClassifier(
        {"a": branch},
        {"a": 4},
        projection_dim=3,
        num_classes=2,
        composer="concat_linear",
        dropout=0.0,
    )
    for parameter in branch.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loader = [
        (
            {"a": torch.ones(2, 3, 8, 8)},
            torch.zeros(2, dtype=torch.long),
            ["x", "y"],
        )
    ]

    train_fusion_epoch(
        model,
        loader,
        optimizer,
        None,
        nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        amp_dtype=None,
        max_batches=None,
    )

    assert branch.seen_training_modes == [False]


def test_fusion_adapter_checkpoint_path_is_fold_local(tmp_path: Path, monkeypatch) -> None:
    def fake_resolve(path):
        path = Path(path)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(fusion_peft_module, "resolve_project_path", fake_resolve)

    checkpoint = resolve_branch_adapter_checkpoint(
        source_run="source_run",
        experiment_id="a__lora",
        fold_idx=3,
    )

    assert checkpoint == tmp_path / "runs/scheme_02/source_run/checkpoints/a__lora/fold_3/best_macro_f1.pt"


def test_resolve_fusion_experiment_finds_named_candidate() -> None:
    config = load_yaml("configs/scheme_03/fusion_single_refit_best.yaml")

    experiment = resolve_fusion_experiment(config, "b1_uni_conch_adapters_all4_concat_p32_reg")

    assert experiment["composer"] == "concat_linear"
    assert experiment["projection_dim"] == 32
    assert experiment["adapter_branches"] == ["uni2_h", "conch"]


def test_fusion_single_refit_config_uses_exact_adapter_checkpoints() -> None:
    config = load_yaml("configs/scheme_03/fusion_single_refit_best.yaml")

    assert config["epochs"] == 14
    assert config["adapter_sources"]["uni2_h"]["checkpoint_path"].endswith("single_refit.pt")
    assert config["adapter_sources"]["conch"]["checkpoint_path"].endswith("single_refit.pt")
    assert config["test_manifest"] == "artifacts/dataset_audit/test_manifest.csv"


def test_build_fusion_model_loads_adapter_branch(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs/scheme_02/source/checkpoints/a__lora/fold_1/best_macro_f1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    loaded = []

    def fake_resolve(path):
        path = Path(path)
        return path if path.is_absolute() else tmp_path / path

    def fake_build_classifier(branch_cfg, *, num_classes):
        return TinyFeatureClassifier(int(branch_cfg["feature_dim"])), 8

    def fake_inject_lora(model, _cfg, targets):
        assert targets == ["linear"]
        return model

    def fake_load_checkpoint(path, **_kwargs):
        loaded.append(path)

    monkeypatch.setattr(fusion_peft_module, "resolve_project_path", fake_resolve)
    monkeypatch.setattr(fusion_peft_module, "build_classifier", fake_build_classifier)
    monkeypatch.setattr(fusion_peft_module, "inject_lora", fake_inject_lora)
    monkeypatch.setattr(fusion_peft_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(fusion_peft_module, "resolve_pretrained_cfg", lambda _model: {})
    monkeypatch.setattr(fusion_peft_module, "resolve_official_preprocess", lambda _model: None)
    monkeypatch.setattr(fusion_peft_module, "build_model_transform", lambda *_args, **_kwargs: object())

    model, transforms = build_fusion_model(
        {
            "branches": [{"key": "a", "feature_dim": 4}],
            "adapter_sources": {
                "a": {
                    "source_run": "source",
                    "experiment_id": "a__lora",
                    "target_policy": "all_linear_except_head",
                    "lora": {"r": 4, "alpha": 8},
                }
            },
        },
        {"name": "b1", "adapter_branches": ["a"], "composer": "concat_linear"},
        num_classes=2,
        fold_idx=1,
    )

    assert loaded == [checkpoint]
    assert model.branch_keys == ["a"]
    assert set(transforms) == {"a"}
    assert model.adapter_provenance["a"]["source_run"] == "source"


def test_build_fusion_model_loads_exact_adapter_checkpoint(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "exact" / "single_refit.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    loaded = []

    def fake_resolve(path):
        path = Path(path)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr(fusion_peft_module, "resolve_project_path", fake_resolve)
    monkeypatch.setattr(
        fusion_peft_module,
        "build_classifier",
        lambda branch_cfg, *, num_classes: (TinyFeatureClassifier(int(branch_cfg["feature_dim"])), 8),
    )
    monkeypatch.setattr(fusion_peft_module, "inject_lora", lambda model, _cfg, _targets: model)
    monkeypatch.setattr(fusion_peft_module, "load_checkpoint", lambda path, **_kwargs: loaded.append(path))
    monkeypatch.setattr(fusion_peft_module, "resolve_pretrained_cfg", lambda _model: {})
    monkeypatch.setattr(fusion_peft_module, "resolve_official_preprocess", lambda _model: None)
    monkeypatch.setattr(fusion_peft_module, "build_model_transform", lambda *_args, **_kwargs: object())

    model, _transforms = build_fusion_model(
        {
            "branches": [{"key": "a", "feature_dim": 4}],
            "adapter_sources": {
                "a": {
                    "source_run": "source",
                    "experiment_id": "a__lora",
                    "checkpoint_path": "exact/single_refit.pt",
                    "target_policy": "all_linear_except_head",
                    "lora": {"r": 4, "alpha": 8},
                }
            },
        },
        {"name": "b1", "adapter_branches": ["a"], "composer": "concat_linear"},
        num_classes=2,
        fold_idx=4,
    )

    assert loaded == [checkpoint]
    assert model.adapter_provenance["a"]["fold"] is None


def test_fusion_pseudo_epoch_scales_soft_pseudo_loss() -> None:
    model = MultiEncoderFusionClassifier(
        {"a": TinyFeatureClassifier(4)},
        {"a": 4},
        projection_dim=3,
        num_classes=2,
        composer="concat_linear",
        dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = [
        {
            "images_by_branch": {"a": torch.ones(2, 3, 8, 8)},
            "hard_label": torch.tensor([0, 1]),
            "soft_label": torch.tensor([[1.0, 0.0], [0.2, 0.8]]),
            "sample_weight": torch.tensor([1.0, 0.1]),
            "is_pseudo": torch.tensor([False, True]),
        }
    ]

    metrics = train_fusion_pseudo_epoch(
        model,
        loader,
        optimizer,
        None,
        device=torch.device("cpu"),
        amp_dtype=None,
        label_smoothing=0.0,
        max_batches=None,
    )

    assert metrics["loss"] > 0
    assert metrics["pseudo_fraction"] == 0.5


def test_resolve_class_weight_tensor_is_opt_in_and_normalized() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    class_names = ["Class_0", "Class_1", "Class_2"]

    assert resolve_class_weight_tensor({}, {}, class_names=class_names, labels=labels) is None

    weights = resolve_class_weight_tensor(
        {
            "class_weighting": {
                "mode": "manual",
                "weights": {"Class_0": 1.0, "Class_1": 1.0, "Class_2": 2.0},
            }
        },
        {},
        class_names=class_names,
        labels=labels,
    )

    assert weights is not None
    expected = torch.tensor([0.75, 0.75, 1.5], dtype=torch.float32)
    assert torch.allclose(weights, expected)


def test_resolve_checkpoint_metric_defaults_to_macro_and_validates() -> None:
    assert resolve_checkpoint_metric({}, {}) == "macro_f1"
    assert resolve_checkpoint_metric({"checkpoint_metric": "selection_score"}, {}) == "selection_score"
    try:
        resolve_checkpoint_metric({"checkpoint_metric": "loss"}, {})
    except ValueError as exc:
        assert "checkpoint_metric" in str(exc)
    else:
        raise AssertionError("invalid checkpoint_metric should fail")


def test_fusion_feature_heads_fit_only_fold_train(tmp_path: Path, monkeypatch) -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    fold_features = [
        {
            "fold_idx": 0,
            "train_idx": np.asarray([0, 2], dtype=np.int64),
            "val_idx": np.asarray([1, 3], dtype=np.int64),
            "X_train": np.asarray([[0.0], [2.0]], dtype=np.float32),
            "X_val": np.asarray([[1.0], [3.0]], dtype=np.float32),
        }
    ]
    seen_train = []

    def fake_fit_head(X, y, _spec, *, seed):
        seen_train.append((X[:, 0].astype(int).tolist(), y.tolist()))
        return object()

    def fake_predict_proba(_head, X):
        return np.full((len(X), 2), 0.5, dtype=np.float64)

    monkeypatch.setattr(fusion_peft_module, "fit_head", fake_fit_head)
    monkeypatch.setattr(fusion_peft_module, "predict_proba", fake_predict_proba)

    rows = [{"filename": f"img_{idx}.png"} for idx in range(4)]
    results = evaluate_fusion_feature_heads(
        head_specs=[{"name": "fake", "family": "logreg", "C": 1.0}],
        fold_features=fold_features,
        labels=labels,
        rows=rows,
        class_names=["Class_0", "Class_1"],
        predictions_dir=tmp_path,
        experiment_id="fusion__x",
        experiment_name="x",
        seed=2026,
    )

    assert seen_train == [([0, 2], [0, 1])]
    assert results[0]["mode"] == "fusion_adapted_head"
    assert (tmp_path / "oof_fusion__x__adapted__fake.npz").exists()


def test_fusion_formal_config_is_single_model_no_tta() -> None:
    config = load_yaml("configs/scheme_02/fusion_peft_cv.yaml")

    assert config["test_prediction_mode"] == "none"
    assert "tta" not in config
    assert {item["composer"] for item in config["experiments"]} == {"concat_linear", "gated_sum"}
    assert all("test_prediction_files" not in item for item in config["experiments"])


def test_fusion_experiment_filter_enforces_staged_gates() -> None:
    experiments = [
        {"name": "b0_all4_concat_frozen"},
        {"name": "b1_uni_conch_adapters_all4_concat"},
    ]

    selected = filter_experiments(experiments, experiment_names=["b0_all4_concat_frozen"])

    assert selected == [{"name": "b0_all4_concat_frozen"}]
    try:
        filter_experiments(experiments, experiment_names=["missing"])
    except ValueError as exc:
        assert "unknown Scheme02b" in str(exc)
    else:
        raise AssertionError("unknown experiment names should fail")
