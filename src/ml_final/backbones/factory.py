"""Backbone and classifier factories for PEFT experiments.

The formal Scheme 02 path uses timm backbones and PEFT LoRA injection.  The
tiny CNN backend exists only for CPU smoke tests and CI; it is not a candidate
model for the final experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ml_final.utils.hf_cache import prepare_locked_hf_model_name, resolve_locked_hf_cache_dir


@dataclass(frozen=True)
class BackboneBuildResult:
    """Constructed backbone with its output feature dimension."""

    backbone: nn.Module
    feature_dim: int
    backend: str
    input_size: int


class TinyConvBackbone(nn.Module):
    """Small deterministic-ish CNN used for smoke tests."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.num_features = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualImageAdapter(nn.Module):
    """Small residual image-space adapter placed before the frozen backbone."""

    def __init__(self, *, hidden_channels: int = 16, residual_scale: float = 0.1) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(x)


class ImageClassifier(nn.Module):
    """Backbone plus optional image adapter and classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        *,
        classifier_head: dict[str, Any] | None = None,
        image_adapter: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.image_adapter = build_image_adapter(image_adapter)
        self.classifier = build_classifier_head(feature_dim, num_classes, classifier_head)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_adapter(x)
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)


class ConchImageBackbone(nn.Module):
    """Official CONCH visual encoder used as an image-only Scheme02 backbone."""

    def __init__(self, model: nn.Module, *, feature_dim: int, official_preprocess) -> None:
        super().__init__()
        self.model = model
        self.num_features = int(feature_dim)
        self.official_preprocess = official_preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(x, proj_contrast=False, normalize=False)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features


class FeaturePolicyBackbone(nn.Module):
    """Apply model-card feature semantics to a timm backbone output."""

    def __init__(self, backbone: nn.Module, feature_policy: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_policy = feature_policy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return process_timm_output(out, self.feature_policy, torch_module=torch)


def build_backbone(config: dict[str, Any]) -> BackboneBuildResult:
    """Build a backbone from config without adding the classification head."""

    backend = str(config.get("backend", "timm"))
    input_size = int(config.get("input_size", 224))
    if backend == "tiny_cnn":
        feature_dim = int(config.get("feature_dim", 64))
        return BackboneBuildResult(
            backbone=TinyConvBackbone(feature_dim=feature_dim),
            feature_dim=feature_dim,
            backend=backend,
            input_size=input_size,
        )
    if backend == "timm":
        import timm

        raw_model_name = str(config["model_name"])
        lock_key = str(config.get("lock_key", "")) or None
        model_name = prepare_locked_hf_model_name(
            raw_model_name,
            lock_key=lock_key,
        )
        cache_dir = resolve_locked_hf_cache_dir(
            raw_model_name,
            lock_key=lock_key,
        )
        pretrained = bool(config.get("pretrained", True))
        kwargs = resolve_timm_model_kwargs(config.get("model_kwargs", {}))
        feature_policy = str(config.get("feature_policy", "default"))
        if feature_policy == "default":
            kwargs.setdefault("num_classes", 0)
        if cache_dir:
            kwargs.setdefault("cache_dir", cache_dir)
        backbone = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        feature_dim = int(config.get("feature_dim", getattr(backbone, "num_features", 0)))
        if feature_policy != "default":
            backbone = FeaturePolicyBackbone(backbone, feature_policy)
        if feature_dim <= 0:
            feature_dim = infer_feature_dim(backbone, input_size=input_size)
        return BackboneBuildResult(
            backbone=backbone,
            feature_dim=feature_dim,
            backend=backend,
            input_size=input_size,
        )
    if backend == "conch":
        from ml_final.heads.conch_prompt import load_conch_model

        model, preprocess, _zero_shot_classifier = load_conch_model(config)
        feature_dim = int(config.get("feature_dim", getattr(model, "embed_dim", 0) or 512))
        backbone = ConchImageBackbone(
            model,
            feature_dim=feature_dim,
            official_preprocess=preprocess,
        )
        if feature_dim <= 0:
            feature_dim = infer_feature_dim(backbone, input_size=input_size)
        return BackboneBuildResult(
            backbone=backbone,
            feature_dim=feature_dim,
            backend=backend,
            input_size=input_size,
        )
    raise ValueError(f"unsupported backbone backend: {backend}")


def resolve_timm_model_kwargs(raw_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve YAML-safe timm kwargs into Python objects."""

    kwargs = dict(raw_kwargs or {})
    for key in ("mlp_layer", "act_layer"):
        value = kwargs.get(key)
        if isinstance(value, str):
            kwargs[key] = resolve_known_symbol(value)
    return kwargs


def resolve_known_symbol(path: str) -> Any:
    """Resolve a small allowlist of model-card class/function references."""

    if path == "timm.layers.SwiGLUPacked":
        from timm.layers import SwiGLUPacked

        return SwiGLUPacked
    if path == "torch.nn.SiLU":
        return torch.nn.SiLU
    raise ValueError(f"unsupported timm kwarg symbol: {path}")


def process_timm_output(out: Any, feature_policy: str, *, torch_module=torch) -> torch.Tensor:
    """Convert raw timm outputs into official feature vectors."""

    if isinstance(out, (tuple, list)):
        out = out[0]
    if feature_policy == "virchow2_official":
        if out.ndim != 3 or out.shape[1] < 6:
            raise ValueError(
                "virchow2_official expects token output shaped [batch, tokens, dim] "
                "with cls, register, and patch tokens"
            )
        class_token = out[:, 0]
        patch_tokens = out[:, 5:]
        return torch_module.cat([class_token, patch_tokens.mean(1)], dim=-1)
    if feature_policy != "default":
        raise ValueError(f"unsupported feature_policy: {feature_policy}")
    if out.ndim > 2:
        out = torch_module.flatten(out, start_dim=1)
    return out


def build_classifier(config: dict[str, Any], *, num_classes: int) -> tuple[ImageClassifier, int]:
    """Build an image classifier and return it with input size."""

    result = build_backbone(config)
    model = ImageClassifier(
        result.backbone,
        result.feature_dim,
        num_classes,
        classifier_head=config.get("classifier_head"),
        image_adapter=config.get("image_adapter"),
    )
    return model, result.input_size


def merge_model_overrides(backbone_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply opt-in model architecture overrides from an experiment config."""

    merged = dict(backbone_cfg)
    for key in ("classifier_head", "image_adapter"):
        if key in experiment_cfg:
            merged[key] = experiment_cfg[key]
    return merged


def build_classifier_head(
    feature_dim: int,
    num_classes: int,
    config: dict[str, Any] | None = None,
) -> nn.Module:
    """Build the classifier head; empty config preserves the legacy linear head."""

    cfg = dict(config or {})
    head_type = str(cfg.get("type", "linear"))
    if head_type == "linear":
        return nn.Linear(feature_dim, num_classes)
    if head_type == "mlp":
        hidden_dim = int(cfg.get("hidden_dim", 256))
        if hidden_dim <= 0:
            raise ValueError("classifier_head.hidden_dim must be positive")
        dropout = float(cfg.get("dropout", 0.0))
        use_layer_norm = bool(cfg.get("layer_norm", True))
        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(feature_dim))
        layers.extend(
            [
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        return nn.Sequential(*layers)
    raise ValueError(f"unsupported classifier_head.type: {head_type}")


def build_image_adapter(config: dict[str, Any] | None = None) -> nn.Module:
    """Build an optional pre-backbone image adapter."""

    cfg = dict(config or {})
    adapter_type = str(cfg.get("type", "none"))
    if adapter_type in {"none", "identity"}:
        return nn.Identity()
    if adapter_type == "residual_conv":
        hidden_channels = int(cfg.get("hidden_channels", 16))
        if hidden_channels <= 0:
            raise ValueError("image_adapter.hidden_channels must be positive")
        return ResidualImageAdapter(
            hidden_channels=hidden_channels,
            residual_scale=float(cfg.get("residual_scale", 0.1)),
        )
    raise ValueError(f"unsupported image_adapter.type: {adapter_type}")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone parameters while leaving classifier parameters trainable."""

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("model does not expose a backbone attribute")
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for parameter in getattr(model, "image_adapter", nn.Identity()).parameters():
        parameter.requires_grad = True


def freeze_backbone_except_last_blocks(
    model: nn.Module,
    *,
    num_blocks: int = 1,
    train_norm: bool = True,
) -> list[int]:
    """Freeze the backbone except the final N transformer blocks and head."""

    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    block_indices: set[int] = set()
    for name, _parameter in model.named_parameters():
        parts = name.split(".")
        if "blocks" not in parts:
            continue
        block_pos = parts.index("blocks")
        if block_pos + 1 < len(parts) and parts[block_pos + 1].isdigit():
            block_indices.add(int(parts[block_pos + 1]))
    ranked_blocks = sorted(block_indices)
    if not ranked_blocks:
        raise ValueError("last_blocks mode requires a backbone exposing transformer blocks")
    selected = set(ranked_blocks[-num_blocks:])
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        train = name.startswith("classifier.") or name.startswith("image_adapter.")
        if "blocks" in parts:
            block_pos = parts.index("blocks")
            if block_pos + 1 < len(parts) and parts[block_pos + 1].isdigit():
                train = train or int(parts[block_pos + 1]) in selected
        if train_norm and any(part in {"norm", "fc_norm"} for part in parts):
            train = train or name.startswith("backbone.")
        parameter.requires_grad = train
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for parameter in getattr(model, "image_adapter", nn.Identity()).parameters():
        parameter.requires_grad = True
    return sorted(selected)


def freeze_all_except_trainable_adapters_and_head(model: nn.Module) -> None:
    """Freeze base parameters after PEFT injection while preserving adapters and head."""

    for name, parameter in model.named_parameters():
        train = (
            "lora_" in name
            or ".lora_" in name
            or "modules_to_save" in name
            or name.startswith("classifier.")
            or ".classifier." in name
            or name.startswith("image_adapter.")
            or ".image_adapter." in name
        )
        parameter.requires_grad = train


def enable_gradient_checkpointing(module: nn.Module) -> bool:
    """Enable gradient checkpointing for timm/HF-style models when available."""

    candidates = [
        module,
        getattr(module, "backbone", None),
        getattr(getattr(module, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "set_grad_checkpointing"):
            candidate.set_grad_checkpointing(True)
            return True
        if hasattr(candidate, "gradient_checkpointing_enable"):
            candidate.gradient_checkpointing_enable()
            return True
    return False


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """Return total/trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    ratio = float(trainable / total) if total else 0.0
    return {"total": total, "trainable": trainable, "trainable_ratio": ratio}


def infer_feature_dim(backbone: nn.Module, *, input_size: int) -> int:
    """Infer feature dimension with a tiny CPU forward pass."""

    was_training = backbone.training
    backbone.eval()
    device = next(backbone.parameters(), torch.zeros(1)).device
    with torch.no_grad():
        sample = torch.zeros(1, 3, input_size, input_size, device=device)
        out = backbone(sample)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim > 2:
            out = torch.flatten(out, start_dim=1)
    backbone.train(was_training)
    return int(out.shape[1])
