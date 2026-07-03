"""Datasets for pseudo-label PEFT training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml_final.training.data import build_label_mapping, read_manifest
from ml_final.utils.config import resolve_project_path


class PseudoImageDataset(Dataset):
    """Mixed true-label and pseudo-label image dataset."""

    def __init__(
        self,
        true_rows: list[dict[str, Any]],
        pseudo_rows: list[dict[str, Any]],
        *,
        class_names: list[str],
        label_to_idx: dict[str, int],
        transform,
        n_classes: int,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.transform = transform
        for row in true_rows:
            label_idx = label_to_idx[row["label"]]
            soft = np.asarray(row.get("soft_label", []), dtype=np.float32)
            if soft.shape != (n_classes,):
                soft = np.zeros(n_classes, dtype=np.float32)
                soft[label_idx] = 1.0
            else:
                soft = soft / np.maximum(soft.sum(), 1e-12)
            self.items.append(
                {
                    "abs_path": row["abs_path"],
                    "filename": row["filename"],
                    "hard_label": label_idx,
                    "soft_label": soft,
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                    "is_pseudo": False,
                }
            )
        for row in pseudo_rows:
            pseudo_label = row["pseudo_label"]
            hard_label = class_names.index(pseudo_label)
            self.items.append(
                {
                    "abs_path": row["abs_path"],
                    "filename": row["filename"],
                    "hard_label": hard_label,
                    "soft_label": row["soft_label"].astype(np.float32),
                    "sample_weight": float(row["sample_weight"]),
                    "is_pseudo": True,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image_path = resolve_project_path(item["abs_path"])
        if image_path is None:
            raise ValueError("image path cannot be empty")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "hard_label": torch.tensor(item["hard_label"], dtype=torch.long),
            "soft_label": torch.tensor(item["soft_label"], dtype=torch.float32),
            "sample_weight": torch.tensor(item["sample_weight"], dtype=torch.float32),
            "is_pseudo": torch.tensor(item["is_pseudo"], dtype=torch.bool),
            "filename": item["filename"],
        }


class MultiTransformPseudoDataset(Dataset):
    """Mixed true/pseudo dataset that returns one tensor per fusion branch."""

    def __init__(
        self,
        true_rows: list[dict[str, Any]],
        pseudo_rows: list[dict[str, Any]],
        *,
        class_names: list[str],
        label_to_idx: dict[str, int],
        transforms_by_key: dict[str, Any],
        n_classes: int,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        self.transforms_by_key = dict(transforms_by_key)
        for row in true_rows:
            label_idx = label_to_idx[row["label"]]
            soft = np.asarray(row.get("soft_label", []), dtype=np.float32)
            if soft.shape != (n_classes,):
                soft = np.zeros(n_classes, dtype=np.float32)
                soft[label_idx] = 1.0
            else:
                soft = soft / np.maximum(soft.sum(), 1e-12)
            self.items.append(
                {
                    "abs_path": row["abs_path"],
                    "filename": row["filename"],
                    "hard_label": label_idx,
                    "soft_label": soft,
                    "sample_weight": float(row.get("sample_weight", 1.0)),
                    "is_pseudo": False,
                }
            )
        for row in pseudo_rows:
            pseudo_label = row["pseudo_label"]
            hard_label = class_names.index(pseudo_label)
            self.items.append(
                {
                    "abs_path": row["abs_path"],
                    "filename": row["filename"],
                    "hard_label": hard_label,
                    "soft_label": row["soft_label"].astype(np.float32),
                    "sample_weight": float(row["sample_weight"]),
                    "is_pseudo": True,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image_path = resolve_project_path(item["abs_path"])
        if image_path is None:
            raise ValueError("image path cannot be empty")
        image = Image.open(image_path).convert("RGB")
        images = {
            key: transform(image.copy()) if transform is not None else image.copy()
            for key, transform in self.transforms_by_key.items()
        }
        return {
            "images_by_branch": images,
            "hard_label": torch.tensor(item["hard_label"], dtype=torch.long),
            "soft_label": torch.tensor(item["soft_label"], dtype=torch.float32),
            "sample_weight": torch.tensor(item["sample_weight"], dtype=torch.float32),
            "is_pseudo": torch.tensor(item["is_pseudo"], dtype=torch.bool),
            "filename": item["filename"],
        }


class UnlabeledManifestDataset(Dataset):
    """Unlabeled image dataset backed by a test manifest."""

    def __init__(self, rows: list[dict[str, str]], *, transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = resolve_project_path(row["abs_path"])
        if image_path is None:
            raise ValueError("image path cannot be empty")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["filename"]


def load_true_and_pseudo_rows(
    *,
    train_manifest: str | Path,
    test_manifest: str | Path,
    pseudolabels: str | Path,
    lambda_pseudo: float,
    min_pseudo_sample_weight: float = 0.0,
    max_pseudo_per_class: int | None = None,
    true_soft_labels: str | Path | None = None,
    pseudo_class_weight: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Load manifests and selected pseudo labels for S03-6."""

    train_path = resolve_project_path(train_manifest)
    test_path = resolve_project_path(test_manifest)
    pseudo_path = resolve_project_path(pseudolabels)
    if train_path is None or test_path is None or pseudo_path is None:
        raise ValueError("train_manifest, test_manifest and pseudolabels are required")
    train_rows = read_manifest(train_path)
    test_rows = read_manifest(test_path)
    class_names, label_to_idx = build_label_mapping(train_rows)
    pseudo_weight_by_class = resolve_pseudo_class_weight(pseudo_class_weight, class_names)
    if true_soft_labels is not None:
        train_rows = attach_true_soft_labels(train_rows, true_soft_labels, class_names)
    test_by_filename = {row["filename"]: row for row in test_rows}
    pseudo_rows: list[dict[str, Any]] = []
    counts = {name: 0 for name in class_names}
    with pseudo_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            filename = raw["filename"]
            test_row = test_by_filename.get(filename)
            if test_row is None:
                continue
            pseudo_label = raw["pseudo_label"]
            if pseudo_label not in counts:
                continue
            if max_pseudo_per_class is not None and counts[pseudo_label] >= max_pseudo_per_class:
                continue
            base_weight = float(raw["sample_weight"])
            if base_weight < min_pseudo_sample_weight:
                continue
            final_weight = base_weight * lambda_pseudo * pseudo_weight_by_class[pseudo_label]
            soft = np.asarray([float(raw[f"soft_label_{idx}"]) for idx in range(len(class_names))], dtype=np.float64)
            soft = soft / np.maximum(soft.sum(), 1e-12)
            pseudo_rows.append(
                {
                    "abs_path": test_row["abs_path"],
                    "filename": filename,
                    "pseudo_label": pseudo_label,
                    "soft_label": soft.astype(np.float32),
                    "sample_weight": final_weight,
                }
            )
            counts[pseudo_label] += 1
    weight_sums = {
        name: float(sum(float(row["sample_weight"]) for row in pseudo_rows if row["pseudo_label"] == name))
        for name in class_names
    }
    return {
        "train_rows": train_rows,
        "pseudo_rows": pseudo_rows,
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "pseudo_counts": counts,
        "pseudo_class_weight": pseudo_weight_by_class,
        "pseudo_effective_weight_sum_by_class": weight_sums,
        "pseudo_effective_weight_sum": float(sum(weight_sums.values())),
    }


def resolve_pseudo_class_weight(
    raw: dict[str, float] | None,
    class_names: list[str],
) -> dict[str, float]:
    """Resolve opt-in pseudo-only class weights."""

    weights = {name: 1.0 for name in class_names}
    if not raw:
        return weights
    unknown = sorted(set(raw) - set(class_names))
    if unknown:
        raise ValueError(f"pseudo_class_weight contains unknown classes: {unknown}")
    for name, value in raw.items():
        weight = float(value)
        if weight < 0:
            raise ValueError(f"pseudo_class_weight must be non-negative for {name}: {weight}")
        weights[name] = weight
    return weights


def attach_true_soft_labels(
    train_rows: list[dict[str, str]],
    true_soft_labels: str | Path,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Attach optional training-time soft labels to manifest rows by filename."""

    soft_path = resolve_project_path(true_soft_labels)
    if soft_path is None or not soft_path.exists():
        raise FileNotFoundError(f"true_soft_labels not found: {true_soft_labels}")
    soft_by_filename: dict[str, np.ndarray] = {}
    weight_by_filename: dict[str, float] = {}
    with soft_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            filename = raw["filename"]
            soft = np.asarray([float(raw[f"soft_label_{idx}"]) for idx in range(len(class_names))], dtype=np.float64)
            soft_by_filename[filename] = (soft / np.maximum(soft.sum(), 1e-12)).astype(np.float32)
            if "sample_weight" in raw and raw["sample_weight"] != "":
                weight_by_filename[filename] = float(raw["sample_weight"])
    out: list[dict[str, Any]] = []
    for row in train_rows:
        item: dict[str, Any] = dict(row)
        soft = soft_by_filename.get(row["filename"])
        if soft is not None:
            item["soft_label"] = soft
        if row["filename"] in weight_by_filename:
            item["sample_weight"] = weight_by_filename[row["filename"]]
        out.append(item)
    return out
