"""Frozen feature extraction for Scheme 01."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from ml_final.features.store import l2_normalize, save_feature_bundle
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.hf_cache import prepare_locked_hf_model_name
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths


def run_feature_extraction(
    config_path: str | Path,
    *,
    tta: str | None = None,
    run_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Extract frozen train/test features according to a Scheme 01 config."""
    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(config, context="Scheme01 feature extraction")
    tta_name = tta or config.get("tta", "none")
    train_expand = str(config.get("train_expand", "none"))
    eval_tta = str(config.get("eval_tta", tta_name))
    color_tta = config.get("color_tta")
    run_name = run_name or config.get("run_name", "scheme01_extract")
    output_dir = resolve_project_path(config.get("output_dir", "artifacts/features/scheme_01"))
    if output_dir is None:
        raise ValueError("output_dir cannot be None")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = resolve_project_path(config["train_manifest"])
    test_manifest = resolve_project_path(config.get("test_manifest"))
    if train_manifest is None:
        raise ValueError("train_manifest cannot be None")

    train_rows = read_manifest(train_manifest)
    test_rows = read_manifest(test_manifest) if test_manifest and test_manifest.exists() else []
    class_names = sorted({row["label"] for row in train_rows})
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}

    backbones = config.get("backbones", [])
    if smoke:
        backbones = [
            {
                "key": "pixel_stats",
                "backend": "pixel_stats",
                "input_size": 32,
                "batch_size": 64,
            }
        ]
    if not backbones:
        raise ValueError("no backbones configured")

    results: dict[str, Any] = {"run_name": run_name, "tta": tta_name, "features": {}}
    for backbone_cfg in backbones:
        key = str(backbone_cfg["key"])
        backend = str(backbone_cfg.get("backend", "timm"))
        input_size = int(backbone_cfg.get("input_size", 224))
        scale_to = optional_positive_int(backbone_cfg.get("scale_to", config.get("scale_to")))
        batch_size = int(backbone_cfg.get("batch_size", config.get("batch_size", 32)))
        variant = build_variant_name(tta_name=tta_name, eval_tta=eval_tta, train_expand=train_expand, color_tta=color_tta)
        bundle_name = f"{key}_{backend}_{variant}_{short_hash(json.dumps(backbone_cfg, sort_keys=True))}.npz"
        bundle_path = output_dir / key / bundle_name
        if bool(config.get("skip_completed", False)) and bundle_path.exists():
            from ml_final.features.store import load_feature_bundle

            bundle = load_feature_bundle(bundle_path)
            train_count = int(np.asarray(bundle["train_features"]).shape[0])
            test_count = int(np.asarray(bundle["test_features"]).shape[0]) if "test_features" in bundle else 0
            results["features"][key] = {
                "path": project_relative(bundle_path),
                "feature_dim": int(np.asarray(bundle["train_features"]).shape[1]),
                "train_count": train_count,
                "test_count": test_count,
                "skipped_completed": True,
            }
            continue

        extractor = build_extractor(backbone_cfg, smoke=smoke)
        train_features, train_origin_filenames, train_origin_indices = extract_train_rows(
            train_rows,
            extractor,
            input_size=input_size,
            scale_to=scale_to,
            tta=tta_name,
            train_expand=train_expand,
            color_tta=color_tta,
            batch_size=batch_size,
        )
        train_eval_features = None
        if train_origin_filenames is not None:
            train_eval_features = extract_rows(
                train_rows,
                extractor,
                input_size=input_size,
                scale_to=scale_to,
                tta=eval_tta,
                color_tta=color_tta,
                batch_size=batch_size,
            )
        test_features = (
            extract_rows(
                test_rows,
                extractor,
                input_size=input_size,
                scale_to=scale_to,
                tta=eval_tta,
                color_tta=color_tta,
                batch_size=batch_size,
            )
            if test_rows
            else None
        )
        train_features = l2_normalize(train_features).astype(np.float32)
        if train_eval_features is not None:
            train_eval_features = l2_normalize(train_eval_features).astype(np.float32)
        if test_features is not None:
            test_features = l2_normalize(test_features).astype(np.float32)

        if train_origin_filenames is None:
            train_labels = np.asarray([label_to_idx[row["label"]] for row in train_rows], dtype=np.int64)
            train_filenames = np.asarray([row["filename"] for row in train_rows], dtype=object)
        else:
            label_by_name = {row["filename"]: label_to_idx[row["label"]] for row in train_rows}
            train_labels = np.asarray([label_by_name[name] for name in train_origin_filenames], dtype=np.int64)
            train_filenames = np.asarray(
                [f"{name}__aug{idx:02d}" for idx, name in enumerate(train_origin_filenames)],
                dtype=object,
            )
        test_filenames = (
            np.asarray([row["filename"] for row in test_rows], dtype=object) if test_rows else None
        )

        metadata = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_name": run_name,
            "backbone": backbone_cfg,
            "backend": backend,
            "input_size": input_size,
            "scale_to": scale_to,
            "batch_size": batch_size,
            "tta": tta_name,
            "eval_tta": eval_tta,
            "tta_view_count": count_tta_views(tta_name, color_tta=color_tta),
            "eval_tta_view_count": count_tta_views(eval_tta, color_tta=color_tta),
            "train_expand": train_expand,
            "train_expand_view_count": count_tta_views(train_expand, color_tta=color_tta)
            if train_expand not in {"", "none", None}
            else 1,
            "color_tta": color_tta,
            "feature_normalization": "l2",
            "feature_mode": backbone_cfg.get("feature_mode", backbone_cfg.get("conch_feature_mode")),
            "train_manifest": project_relative(train_manifest),
            "test_manifest": project_relative(test_manifest) if test_rows else None,
            "feature_dim": int(train_features.shape[1]),
            "train_count": int(train_features.shape[0]),
            "test_count": int(test_features.shape[0]) if test_features is not None else 0,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
            "data_config": getattr(extractor, "data_config", None),
        }
        save_feature_bundle(
            bundle_path,
            train_features=train_features,
            train_labels=train_labels,
            train_filenames=train_filenames,
            test_features=test_features,
            test_filenames=test_filenames,
            class_names=class_names,
            metadata=metadata,
            train_origin_filenames=np.asarray(train_origin_filenames, dtype=object)
            if train_origin_filenames is not None
            else None,
            train_origin_indices=train_origin_indices,
            train_eval_features=train_eval_features,
        )
        results["features"][key] = {
            "path": project_relative(bundle_path),
            "feature_dim": metadata["feature_dim"],
            "train_count": metadata["train_count"],
            "test_count": metadata["test_count"],
        }

    manifest_path = output_dir / f"{run_name}_{tta_name}_feature_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["manifest_path"] = project_relative(manifest_path)
    return results


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read an audit CSV manifest."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_extractor(config: dict[str, Any], *, smoke: bool = False):
    """Build a feature extractor callable."""
    backend = str(config.get("backend", "timm"))
    if smoke or backend == "pixel_stats":
        return PixelStatsExtractor()
    if backend == "cell_stats":
        return CellStatsExtractor()
    if backend == "timm":
        return TimmExtractor(config)
    if backend == "conch":
        return ConchImageExtractor(config)
    raise ValueError(f"unsupported feature backend: {backend}")


def extract_rows(
    rows: list[dict[str, str]],
    extractor,
    *,
    input_size: int,
    scale_to: int | None = None,
    tta: str,
    color_tta: dict[str, Any] | None = None,
    batch_size: int,
) -> np.ndarray:
    """Extract features for manifest rows."""
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    if hasattr(extractor, "extract_images"):
        all_views: list[Image.Image] = []
        view_counts: list[int] = []
        for row in rows:
            image_path = resolve_project_path(row["abs_path"])
            if image_path is None:
                raise ValueError("manifest row path cannot be empty")
            image = Image.open(image_path).convert("RGB")
            if getattr(extractor, "uses_model_transform", False):
                views = [
                    preprocess_feature_view(view, input_size=input_size, scale_to=scale_to)
                    for view in iter_tta_views(image, tta, color_tta=color_tta)
                ]
            else:
                views = [
                    preprocess_feature_view(view, input_size=input_size, scale_to=scale_to)
                    for view in iter_tta_views(image, tta, color_tta=color_tta)
                ]
            all_views.extend(views)
            view_counts.append(len(views))
        all_features = extractor.extract_images(all_views, batch_size=batch_size)
        per_row_features = []
        start = 0
        for count in view_counts:
            stop = start + count
            per_row_features.append(all_features[start:stop].mean(axis=0))
            start = stop
        return np.stack(per_row_features, axis=0).astype(np.float32)

    features = []
    for row in rows:
        image_path = resolve_project_path(row["abs_path"])
        if image_path is None:
            raise ValueError("manifest row path cannot be empty")
        image = Image.open(image_path).convert("RGB")
        view_features = []
        for view in iter_tta_views(image, tta, color_tta=color_tta):
            view = preprocess_feature_view(view, input_size=input_size, scale_to=scale_to)
            view_features.append(extractor(view))
        features.append(np.mean(np.stack(view_features, axis=0), axis=0))
    return np.stack(features, axis=0).astype(np.float32)


def extract_train_rows(
    rows: list[dict[str, str]],
    extractor,
    *,
    input_size: int,
    scale_to: int | None = None,
    tta: str,
    train_expand: str,
    color_tta: dict[str, Any] | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract train features, optionally expanding each sample into D4 views."""

    if train_expand in {"", "none", None}:
        return (
            extract_rows(
                rows,
                extractor,
                input_size=input_size,
                scale_to=scale_to,
                tta=tta,
                color_tta=color_tta,
                batch_size=batch_size,
            ),
            None,
            None,
        )
    if train_expand not in {"d4", "geom6"}:
        raise ValueError(f"unsupported train_expand: {train_expand}")
    all_views: list[Image.Image] = []
    origins = []
    origin_indices = []
    for row_idx, row in enumerate(rows):
        image_path = resolve_project_path(row["abs_path"])
        if image_path is None:
            raise ValueError("manifest row path cannot be empty")
        image = Image.open(image_path).convert("RGB")
        base_views = list(iter_base_tta_views(image, train_expand))
        views = []
        for view in base_views:
            views.extend(iter_color_views(view, color_tta))
        if getattr(extractor, "uses_model_transform", False):
            processed_views = [
                preprocess_feature_view(view, input_size=input_size, scale_to=scale_to)
                for view in views
            ]
        else:
            processed_views = [
                preprocess_feature_view(view, input_size=input_size, scale_to=scale_to)
                for view in views
            ]
        all_views.extend(processed_views)
        origins.extend([row["filename"]] * len(processed_views))
        origin_indices.extend([row_idx] * len(processed_views))
    if getattr(extractor, "uses_model_transform", False):
        features = extractor.extract_images(all_views, batch_size=batch_size)
    else:
        features = np.stack([extractor(view) for view in all_views], axis=0)
    return (
        features.astype(np.float32),
        np.asarray(origins, dtype=object),
        np.asarray(origin_indices, dtype=np.int64),
    )


def optional_positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"expected a positive integer, got {value!r}")
    return result


def preprocess_feature_view(image: Image.Image, *, input_size: int, scale_to: int | None = None) -> Image.Image:
    """Resize normally, or place a smaller scale view on a centered square canvas."""

    if scale_to is None or scale_to == input_size:
        return image.resize((input_size, input_size), resample=Image.Resampling.BICUBIC)
    if scale_to > input_size:
        raise ValueError(f"scale_to must be <= input_size, got scale_to={scale_to}, input_size={input_size}")
    resized = image.resize((scale_to, scale_to), resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (input_size, input_size), color=(255, 255, 255))
    offset = ((input_size - scale_to) // 2, (input_size - scale_to) // 2)
    canvas.paste(resized, offset)
    return canvas


def iter_tta_views(
    image: Image.Image,
    tta: str,
    *,
    color_tta: dict[str, Any] | None = None,
) -> Iterable[Image.Image]:
    """Yield deterministic TTA views."""
    for view in iter_base_tta_views(image, tta):
        yield from iter_color_views(view, color_tta)


def iter_base_tta_views(image: Image.Image, tta: str) -> Iterable[Image.Image]:
    """Yield deterministic geometric TTA views."""

    if tta == "none":
        yield image
        return
    if tta not in {"d4", "geom6"}:
        raise ValueError(f"unsupported tta: {tta}")
    yield image
    yield ImageOps.mirror(image)
    yield ImageOps.flip(image)
    yield image.rotate(90, expand=True)
    yield image.rotate(180, expand=True)
    yield image.rotate(270, expand=True)
    if tta == "d4":
        mirrored = ImageOps.mirror(image)
        yield mirrored.rotate(90, expand=True)
        yield mirrored.rotate(270, expand=True)


def count_tta_views(tta: str, *, color_tta: dict[str, Any] | None = None) -> int:
    """Return the number of deterministic views for metadata."""

    base_counts = {"none": 1, "geom6": 6, "d4": 8}
    if tta not in base_counts:
        raise ValueError(f"unsupported tta: {tta}")
    color_count = 1
    if color_tta and float(color_tta.get("strength", 0.0)) > 0:
        color_count += 2 * len(color_tta.get("channels", ["brightness", "contrast", "saturation"]))
    return base_counts[tta] * color_count


def iter_color_views(image: Image.Image, color_tta: dict[str, Any] | None) -> Iterable[Image.Image]:
    """Yield deterministic mild color views including the original image."""

    yield image
    if not color_tta:
        return
    strength = float(color_tta.get("strength", 0.0))
    if strength <= 0:
        return
    channels = [str(item) for item in color_tta.get("channels", ["brightness", "contrast", "saturation"])]
    factors = [1.0 - strength, 1.0 + strength]
    for channel in channels:
        enhancer_cls = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }.get(channel)
        if enhancer_cls is None:
            raise ValueError(f"unsupported color_tta channel: {channel}")
        enhancer = enhancer_cls(image)
        for factor in factors:
            yield enhancer.enhance(factor)


def build_variant_name(
    *,
    tta_name: str,
    eval_tta: str,
    train_expand: str,
    color_tta: dict[str, Any] | None,
) -> str:
    """Build a stable feature variant name for cache files."""

    parts = [f"tta-{tta_name}"]
    if eval_tta != tta_name:
        parts.append(f"eval-{eval_tta}")
    if train_expand not in {"", "none", None}:
        parts.append(f"trainexpand-{train_expand}")
    if color_tta:
        parts.append(f"color-{float(color_tta.get('strength', 0.0)):.3f}".replace(".", "p"))
    return "_".join(parts)


class PixelStatsExtractor:
    """Small deterministic feature extractor for local smoke tests."""

    def __call__(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image).astype(np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)
        mins = flat.min(axis=0)
        maxs = flat.max(axis=0)
        q25, q50, q75 = np.quantile(flat, [0.25, 0.50, 0.75], axis=0)
        return np.concatenate([means, stds, mins, maxs, q25, q50, q75]).astype(np.float32)


class CellStatsExtractor:
    """Handcrafted color, texture, and coarse morphology features for 32x32 cells."""

    def __call__(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        flat_rgb = arr.reshape(-1, 3)
        features = [
            flat_rgb.mean(axis=0),
            flat_rgb.std(axis=0),
            np.quantile(flat_rgb, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0).reshape(-1),
            color_ratios(flat_rgb),
            gray_stats(gray),
            gradient_stats(gray),
            center_context_stats(gray),
            dark_region_stats(gray),
            histogram(gray, bins=8),
        ]
        return np.concatenate(features).astype(np.float32)


def color_ratios(flat_rgb: np.ndarray) -> np.ndarray:
    eps = 1e-6
    rgb_mean = flat_rgb.mean(axis=0)
    total = rgb_mean.sum() + eps
    return np.asarray(
        [
            rgb_mean[0] / total,
            rgb_mean[1] / total,
            rgb_mean[2] / total,
            rgb_mean[0] / (rgb_mean[1] + eps),
            rgb_mean[2] / (rgb_mean[1] + eps),
        ],
        dtype=np.float32,
    )


def gray_stats(gray: np.ndarray) -> np.ndarray:
    flat = gray.reshape(-1)
    return np.asarray(
        [
            flat.mean(),
            flat.std(),
            flat.min(),
            flat.max(),
            *np.quantile(flat, [0.05, 0.25, 0.50, 0.75, 0.95]).tolist(),
        ],
        dtype=np.float32,
    )


def gradient_stats(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy).reshape(-1)
    return np.asarray(
        [
            mag.mean(),
            mag.std(),
            np.quantile(mag, 0.50),
            np.quantile(mag, 0.90),
            float((mag > np.quantile(mag, 0.75)).mean()),
        ],
        dtype=np.float32,
    )


def center_context_stats(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    center = gray[y0:y1, x0:x1]
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    border = gray[border_mask]
    return np.asarray(
        [
            center.mean(),
            center.std(),
            border.mean(),
            border.std(),
            center.mean() - border.mean(),
            center.std() - border.std(),
        ],
        dtype=np.float32,
    )


def dark_region_stats(gray: np.ndarray) -> np.ndarray:
    mask = gray < np.quantile(gray, 0.35)
    area = float(mask.mean())
    if not mask.any():
        return np.zeros(8, dtype=np.float32)
    ys, xs = np.nonzero(mask)
    height, width = gray.shape
    y_center = float(ys.mean() / max(height - 1, 1))
    x_center = float(xs.mean() / max(width - 1, 1))
    y_span = float((ys.max() - ys.min() + 1) / height)
    x_span = float((xs.max() - xs.min() + 1) / width)
    y_norm = ys / max(height - 1, 1) - y_center
    x_norm = xs / max(width - 1, 1) - x_center
    return np.asarray(
        [
            area,
            x_center,
            y_center,
            x_span,
            y_span,
            float(np.mean(x_norm * x_norm)),
            float(np.mean(y_norm * y_norm)),
            float(np.mean(x_norm * y_norm)),
        ],
        dtype=np.float32,
    )


def histogram(values: np.ndarray, *, bins: int) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    return (hist.astype(np.float32) / max(float(hist.sum()), 1.0)).astype(np.float32)


class TimmExtractor:
    """Timm-based frozen feature extractor for server runs."""

    uses_model_transform = True

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        import timm
        from timm.data import create_transform, resolve_data_config

        from ml_final.backbones.factory import process_timm_output, resolve_timm_model_kwargs

        self.torch = torch
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_policy = str(config.get("feature_policy", "default"))
        self.process_timm_output = process_timm_output
        model_name = prepare_locked_hf_model_name(
            str(config["model_name"]),
            lock_key=str(config.get("lock_key", "")) or None,
        )
        pretrained = bool(config.get("pretrained", True))
        model_kwargs = resolve_timm_model_kwargs(config.get("model_kwargs", {}))
        if self.feature_policy == "default":
            model_kwargs.setdefault("num_classes", 0)
        self.model = timm.create_model(model_name, pretrained=pretrained, **model_kwargs)
        self.model.eval().to(self.device)
        data_config = resolve_data_config(self.model.pretrained_cfg, model=self.model)
        input_size = int(config.get("input_size", data_config.get("input_size", (3, 224, 224))[-1]))
        data_config["input_size"] = (3, input_size, input_size)
        if "mean" in config:
            data_config["mean"] = tuple(float(x) for x in config["mean"])
        if "std" in config:
            data_config["std"] = tuple(float(x) for x in config["std"])
        self.transform = create_transform(**data_config, is_training=False)
        self.data_config = dict(data_config)

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract_images([image], batch_size=1)[0]

    def extract_images(self, images: list[Image.Image], *, batch_size: int) -> np.ndarray:
        """Extract a batch of images with timm model-card preprocessing."""

        outputs = []
        for start in range(0, len(images), max(1, batch_size)):
            batch_images = images[start : start + max(1, batch_size)]
            batch = self.torch.stack([self.transform(image) for image in batch_images], dim=0).to(self.device)
            with self.torch.inference_mode():
                out = self.model(batch)
                out = self.process_timm_output(out, self.feature_policy, torch_module=self.torch)
            outputs.append(out.detach().float().cpu().numpy())
        return np.concatenate(outputs, axis=0)


class ConchImageExtractor:
    """Official CONCH image encoder extractor.

    CONCH is not a standard timm model on the Hub.  It must be loaded through
    the MahmoodLab CONCH package so that both the checkpoint and preprocessing
    match the model card and zero-shot examples.
    """

    uses_model_transform = True

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        import torch.nn.functional as F

        from ml_final.heads.conch_prompt import load_conch_model

        self.torch = torch
        self.F = F
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_mode = str(config.get("feature_mode", config.get("conch_feature_mode", "linear_probe")))
        self.model, self.preprocess, _zero_shot_classifier = load_conch_model(config)
        self.model.eval().to(self.device)
        self.proj_contrast = self.feature_mode in {"contrastive", "zero_shot", "retrieval"}
        self.normalize = self.feature_mode in {"contrastive", "zero_shot", "retrieval"}
        self.data_config = {
            "backend": "conch",
            "model_name": str(config.get("model_name", "conch_ViT-B-16")),
            "feature_mode": self.feature_mode,
            "preprocess": "conch.open_clip_custom.create_model_from_pretrained",
            "encode_image": f"proj_contrast={self.proj_contrast}, normalize={self.normalize}",
        }

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.extract_images([image], batch_size=1)[0]

    def extract_images(self, images: list[Image.Image], *, batch_size: int) -> np.ndarray:
        """Extract CONCH image embeddings according to feature_mode."""

        outputs = []
        for start in range(0, len(images), max(1, batch_size)):
            batch_images = images[start : start + max(1, batch_size)]
            batch = self.torch.stack([self.preprocess(image) for image in batch_images], dim=0).to(self.device)
            with self.torch.inference_mode():
                out = self.model.encode_image(batch, proj_contrast=self.proj_contrast, normalize=self.normalize)
                out = out.float()
                if self.normalize:
                    out = self.F.normalize(out, dim=-1)
            outputs.append(out.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
