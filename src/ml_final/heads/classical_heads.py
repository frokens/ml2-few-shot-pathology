"""Classical CV heads for frozen feature bundles."""

from __future__ import annotations

import datetime as dt
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_final.features.store import load_feature_bundle
from ml_final.heads.ensemble_search import average_probs, grid_weight_search, weighted_probs
from ml_final.metrics.classification import compute_classification_metrics, write_json
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths


def run_frozen_cv(
    config_path: str | Path,
    *,
    features_dir: str | Path | None = None,
    run_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run CV heads on Scheme 01 feature bundles."""
    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(
        {"config": config, "features_dir": features_dir},
        context="Scheme01 frozen CV",
    )
    run_name = run_name or config.get("run_name", "frozen_cv")
    features_root = resolve_project_path(features_dir or config.get("features_dir"))
    if features_root is None:
        raise ValueError("features_dir is required")
    run_dir_value = f"runs/scheme_01/{run_name}" if run_name else config.get("run_dir")
    if "run_dir" in config and run_name == config.get("run_name"):
        run_dir_value = config["run_dir"]
    run_dir = resolve_project_path(run_dir_value)
    if run_dir is None:
        raise ValueError("run_dir cannot be None")
    metrics_dir = run_dir / "metrics"
    models_dir = run_dir / "models"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "summary.json"
    if bool(config.get("skip_completed", False)) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {"run_dir": project_relative(run_dir), "summary": summary, "skipped": True}

    feature_files = discover_feature_files(features_root, smoke=smoke)
    if not feature_files:
        raise FileNotFoundError(f"no feature bundles found under {features_root}")

    all_results = []
    oof_entries = []
    test_entries = []
    for feature_file in feature_files:
        bundle = load_feature_bundle(feature_file)
        X = np.asarray(bundle["train_features"], dtype=np.float32)
        y = np.asarray(bundle["train_labels"], dtype=np.int64)
        expanded = "train_eval_features" in bundle and "train_origin_indices" in bundle
        if expanded:
            X_eval = np.asarray(bundle["train_eval_features"], dtype=np.float32)
            origin_indices = np.asarray(bundle["train_origin_indices"], dtype=np.int64)
            y_eval = labels_by_origin(y, origin_indices, X_eval.shape[0])
            eval_filenames = filenames_by_origin(bundle, X_eval.shape[0])
        else:
            X_eval = X
            origin_indices = None
            y_eval = y
            eval_filenames = bundle["train_filenames"]
        class_names = [str(x) for x in bundle["class_names"].tolist()]
        model_prefix = Path(feature_file).stem

        head_specs = build_head_specs(config, smoke=smoke)
        for spec in head_specs:
            if expanded:
                result = evaluate_head_cv_expanded(
                    X_train=X,
                    y_train=y,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    origin_indices=origin_indices,
                    class_names=class_names,
                    spec=spec,
                    config=config,
                )
            else:
                result = evaluate_head_cv(X, y, class_names, spec, config=config)
            head_id = f"{model_prefix}__{spec['name']}"
            result["head_id"] = head_id
            result["feature_file"] = project_relative(feature_file)
            all_results.append(result)

            oof_path = predictions_dir / f"oof_{head_id}.npz"
            np.savez_compressed(
                oof_path,
                probs=result["oof_probs"],
                y_true=y_eval,
                y_pred=result["oof_probs"].argmax(axis=1),
                class_names=np.asarray(class_names, dtype=object),
                train_filenames=eval_filenames,
                head_id=np.asarray(head_id, dtype=object),
                feature_file=np.asarray(project_relative(feature_file), dtype=object),
            )
            oof_entries.append(project_relative(oof_path))

            final_model = fit_head(X, y, spec, seed=int(config.get("seed", 2026)) + 10_000)
            model_path = models_dir / f"{head_id}.pkl"
            with model_path.open("wb") as handle:
                pickle.dump({"model": final_model, "spec": spec, "class_names": class_names}, handle)

            if "test_features" in bundle:
                test_probs = predict_proba(final_model, np.asarray(bundle["test_features"], dtype=np.float32))
                test_path = predictions_dir / f"test_{head_id}.npz"
                np.savez_compressed(
                    test_path,
                    probs=test_probs,
                    class_names=np.asarray(class_names, dtype=object),
                    test_filenames=bundle["test_filenames"],
                    head_id=np.asarray(head_id, dtype=object),
                    feature_file=np.asarray(project_relative(feature_file), dtype=object),
                )
                test_entries.append(project_relative(test_path))

    summary = summarize_results(all_results)
    summary.update(
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_name": run_name,
            "feature_files": [project_relative(path) for path in feature_files],
            "oof_prediction_files": oof_entries,
            "test_prediction_files": test_entries,
        }
    )
    write_json(summary, summary_path)
    write_selection_report(summary, metrics_dir / "selection_report.md")
    if bool(config.get("ensemble", {}).get("enabled", True)):
        ensemble = build_oof_ensemble(all_results, metrics_dir, predictions_dir, config=config)
        summary["ensemble"] = ensemble
    else:
        summary["ensemble"] = {}
    write_json(summary, summary_path)
    return {"run_dir": project_relative(run_dir), "summary": summary}


def discover_feature_files(features_root: Path, *, smoke: bool = False) -> list[Path]:
    files = sorted(features_root.rglob("*.npz"))
    if smoke:
        files = [path for path in files if "pixel_stats" in path.name]
    return files


def build_head_specs(config: dict[str, Any], *, smoke: bool = False) -> list[dict[str, Any]]:
    if smoke:
        return [{"name": "logreg_C1", "family": "logreg", "C": 1.0}]
    specs = config.get("heads")
    if specs:
        return list(specs)
    return [
        {"name": "logreg_C0.1", "family": "logreg", "C": 0.1},
        {"name": "logreg_C1", "family": "logreg", "C": 1.0},
        {"name": "svc_rbf_C1", "family": "svc_rbf", "C": 1.0, "gamma": "scale"},
        {"name": "knn_cosine_k3", "family": "knn_cosine", "k": 3},
        {"name": "prototype_cosine", "family": "prototype"},
    ]


def evaluate_head_cv(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    spec: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    n_splits = int(config.get("n_splits", 5))
    n_repeats = int(config.get("n_repeats", 1))
    seed = int(config.get("seed", 2026))
    splitter = (
        RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        if n_repeats > 1
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    oof_sum = np.zeros((len(y), len(class_names)), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.float64)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = fit_head(X[train_idx], y[train_idx], spec, seed=seed + fold_idx)
        probs = predict_proba(model, X[val_idx])
        oof_sum[val_idx] += probs
        oof_count[val_idx] += 1
        pred = probs.argmax(axis=1)
        fold_metrics.append(
            {
                "fold": fold_idx,
                **compute_classification_metrics(y[val_idx], pred, class_names),
            }
        )
    oof_probs = oof_sum / np.maximum(oof_count[:, None], 1.0)
    y_pred = oof_probs.argmax(axis=1)
    metrics = compute_classification_metrics(y, y_pred, class_names)
    return {"spec": spec, "metrics": metrics, "fold_metrics": fold_metrics, "oof_probs": oof_probs}


def evaluate_head_cv_expanded(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    origin_indices: np.ndarray,
    class_names: list[str],
    spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate CV where training rows are augmented views and validation rows are original samples."""

    n_splits = int(config.get("n_splits", 5))
    n_repeats = int(config.get("n_repeats", 1))
    seed = int(config.get("seed", 2026))
    splitter = (
        RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        if n_repeats > 1
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    oof_sum = np.zeros((len(y_eval), len(class_names)), dtype=np.float64)
    oof_count = np.zeros(len(y_eval), dtype=np.float64)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_eval, y_eval)):
        train_mask = np.isin(origin_indices, train_idx)
        model = fit_head(X_train[train_mask], y_train[train_mask], spec, seed=seed + fold_idx)
        probs = predict_proba(model, X_eval[val_idx])
        oof_sum[val_idx] += probs
        oof_count[val_idx] += 1
        pred = probs.argmax(axis=1)
        fold_metrics.append({"fold": fold_idx, **compute_classification_metrics(y_eval[val_idx], pred, class_names)})
    oof_probs = oof_sum / np.maximum(oof_count[:, None], 1.0)
    y_pred = oof_probs.argmax(axis=1)
    metrics = compute_classification_metrics(y_eval, y_pred, class_names)
    return {"spec": spec, "metrics": metrics, "fold_metrics": fold_metrics, "oof_probs": oof_probs}


def labels_by_origin(labels: np.ndarray, origin_indices: np.ndarray, n_origins: int) -> np.ndarray:
    """Recover original-sample labels from expanded rows."""

    out = np.zeros(n_origins, dtype=np.int64)
    seen = np.zeros(n_origins, dtype=bool)
    for label, origin in zip(labels, origin_indices):
        out[int(origin)] = int(label)
        seen[int(origin)] = True
    if not seen.all():
        missing = np.where(~seen)[0].tolist()
        raise ValueError(f"expanded feature bundle missing origins: {missing}")
    return out


def filenames_by_origin(bundle: dict[str, Any], n_origins: int) -> np.ndarray:
    """Recover original filenames from an expanded feature bundle."""

    origins = np.asarray(bundle["train_origin_filenames"]).astype(str)
    origin_indices = np.asarray(bundle["train_origin_indices"], dtype=np.int64)
    out = np.empty(n_origins, dtype=object)
    seen = np.zeros(n_origins, dtype=bool)
    for filename, origin in zip(origins, origin_indices):
        if not seen[int(origin)]:
            out[int(origin)] = filename
            seen[int(origin)] = True
    if not seen.all():
        missing = np.where(~seen)[0].tolist()
        raise ValueError(f"expanded feature bundle missing filenames for origins: {missing}")
    return out


def fit_head(X: np.ndarray, y: np.ndarray, spec: dict[str, Any], *, seed: int = 2026):
    X, y = augment_training_features(X, y, spec, seed=seed)
    family = spec["family"]
    if family == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(spec.get("C", 1.0)),
                        max_iter=2000,
                        class_weight=spec.get("class_weight"),
                    ),
                ),
            ]
        ).fit(X, y)
    if family == "pca_logreg":
        n_components = min(int(spec.get("n_components", 64)), X.shape[0] - 1, X.shape[1])
        if n_components < 1:
            raise ValueError("pca_logreg requires at least one PCA component")
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        whiten=bool(spec.get("whiten", False)),
                        random_state=seed,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=float(spec.get("C", 1.0)),
                        max_iter=2000,
                        class_weight=spec.get("class_weight"),
                        random_state=seed,
                    ),
                ),
            ]
        ).fit(X, y)
    if family == "bias_tuned_logreg":
        return ClassBiasTunedLogisticRegression(
            C=float(spec.get("C", 1.0)),
            target_classes=[int(item) for item in spec.get("target_classes", [2, 3, 4])],
            bias_values=[float(item) for item in spec.get("bias_values", [-0.4, -0.2, 0.0, 0.2, 0.4])],
            inner_splits=int(spec.get("inner_splits", 5)),
            class_weight=spec.get("class_weight"),
            seed=seed,
        ).fit(X, y)
    if family == "svc_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=float(spec.get("C", 1.0)),
                        gamma=spec.get("gamma", "scale"),
                        kernel="rbf",
                        probability=True,
                        class_weight=spec.get("class_weight"),
                    ),
                ),
            ]
        ).fit(X, y)
    if family == "svc_linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=float(spec.get("C", 1.0)),
                        kernel="linear",
                        probability=True,
                        class_weight=spec.get("class_weight"),
                    ),
                ),
            ]
        ).fit(X, y)
    if family == "lda_shrinkage":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage=spec.get("shrinkage", "auto"),
                    ),
                ),
            ]
        ).fit(X, y)
    if family == "knn_cosine":
        return KNeighborsClassifier(
            n_neighbors=int(spec.get("k", 3)),
            metric="cosine",
            weights=spec.get("weights", "distance"),
        ).fit(X, y)
    if family == "prototype":
        return PrototypeClassifier(
            center=bool(spec.get("center", False)),
            normalize=bool(spec.get("normalize", False)),
            temperature=float(spec.get("temperature", 1.0)),
        ).fit(X, y)
    if family == "hierarchical_hard":
        return HierarchicalHardClassifier(
            hard_classes=[int(item) for item in spec.get("hard_classes", [2, 3, 4])],
            base_C=float(spec.get("base_C", spec.get("C", 1.0))),
            gate_C=float(spec.get("gate_C", spec.get("C", 1.0))),
            hard_C=float(spec.get("hard_C", spec.get("C", 1.0))),
            class_weight=spec.get("class_weight"),
            seed=seed,
        ).fit(X, y)
    raise ValueError(f"unsupported head family: {family}")


def augment_training_features(
    X: np.ndarray,
    y: np.ndarray,
    spec: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fold-local frozen-feature augmentation before fitting a head."""

    aug = spec.get("train_augmentation")
    if not aug:
        return X, y
    aug_type = str(aug.get("type", ""))
    if aug_type != "frofa_brightness_c2":
        raise ValueError(f"unsupported train_augmentation.type: {aug_type}")
    copies = int(aug.get("copies", 1))
    level = float(aug.get("level", 0.2))
    if copies <= 0 or level <= 0:
        return X, y

    target_classes = aug.get("target_classes")
    if target_classes is None or target_classes == "all":
        target_mask = np.ones(len(y), dtype=bool)
    else:
        targets = np.asarray([int(item) for item in target_classes], dtype=np.int64)
        target_mask = np.isin(y, targets)
    if not target_mask.any():
        return X, y

    rng = np.random.default_rng(seed)
    X_float = np.asarray(X, dtype=np.float32)
    X_target = X_float[target_mask]
    x_min = X_float.min(axis=0, keepdims=True)
    x_max = X_float.max(axis=0, keepdims=True)
    scale = np.maximum(x_max - x_min, 1e-6)
    scaled = np.clip((X_target - x_min) / scale, 0.0, 1.0)

    augmented = []
    for _ in range(copies):
        delta = rng.uniform(-level, level, size=scaled.shape).astype(np.float32)
        augmented.append((np.clip(scaled + delta, 0.0, 1.0) * scale + x_min).astype(np.float32))
    return (
        np.concatenate([X_float, *augmented], axis=0),
        np.concatenate([y, *([y[target_mask]] * copies)], axis=0),
    )


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    return np.asarray(probs, dtype=np.float64)


class PrototypeClassifier:
    """Cosine nearest-prototype classifier with softmax over negative distances."""

    def __init__(self, *, center: bool = False, normalize: bool = False, temperature: float = 1.0) -> None:
        self.center = center
        self.normalize = normalize
        self.temperature = max(float(temperature), 1e-6)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0, keepdims=True) if self.center else None
        X_fit = self._transform(X)
        self.classes_ = np.unique(y)
        self.prototypes_ = np.stack([X_fit[y == cls].mean(axis=0) for cls in self.classes_], axis=0)
        if self.normalize:
            self.prototypes_ = self._l2(self.prototypes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_eval = self._transform(np.asarray(X, dtype=np.float64))
        distances = pairwise_distances(X_eval, self.prototypes_, metric="cosine")
        logits = -distances / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        out = X - self.mean_ if self.mean_ is not None else X
        return self._l2(out) if self.normalize else out

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-12)


class HierarchicalHardClassifier:
    """Two-level classifier: easy-vs-hard gate plus a specialist for hard classes."""

    def __init__(
        self,
        *,
        hard_classes: list[int],
        base_C: float = 1.0,
        gate_C: float = 1.0,
        hard_C: float = 1.0,
        class_weight: Any = None,
        seed: int = 2026,
    ) -> None:
        self.hard_classes = np.asarray(hard_classes, dtype=np.int64)
        self.base_C = base_C
        self.gate_C = gate_C
        self.hard_C = hard_C
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        self.hard_classes_ = np.asarray([cls for cls in self.classes_ if cls in set(self.hard_classes)], dtype=np.int64)
        self.easy_classes_ = np.asarray([cls for cls in self.classes_ if cls not in set(self.hard_classes_)], dtype=np.int64)
        if len(self.hard_classes_) < 2 or len(self.easy_classes_) < 1:
            raise ValueError("hierarchical_hard requires at least two hard classes and one easy class")

        hard_mask = np.isin(y, self.hard_classes_)
        self.base_model_ = self._logreg(self.base_C, self.class_weight).fit(X, y)
        self.gate_model_ = self._logreg(self.gate_C, "balanced").fit(X, hard_mask.astype(np.int64))
        self.hard_model_ = self._logreg(self.hard_C, self.class_weight).fit(X[hard_mask], y[hard_mask])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        base_probs = self._align_probs(self.base_model_, X, self.classes_)
        gate_probs = self.gate_model_.predict_proba(X)
        gate_classes = np.asarray(self.gate_model_.classes_, dtype=np.int64)
        hard_col = int(np.where(gate_classes == 1)[0][0])
        hard_mass = gate_probs[:, hard_col]
        hard_cond = self._align_probs(self.hard_model_, X, self.hard_classes_)

        probs = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        easy_indices = [int(np.where(self.classes_ == cls)[0][0]) for cls in self.easy_classes_]
        hard_indices = [int(np.where(self.classes_ == cls)[0][0]) for cls in self.hard_classes_]
        easy_base = base_probs[:, easy_indices]
        easy_cond = easy_base / np.maximum(easy_base.sum(axis=1, keepdims=True), 1e-12)
        probs[:, easy_indices] = (1.0 - hard_mass[:, None]) * easy_cond
        probs[:, hard_indices] = hard_mass[:, None] * hard_cond
        return probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)

    def _logreg(self, C: float, class_weight: Any):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        class_weight=class_weight,
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _align_probs(model, X: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        raw = np.asarray(model.predict_proba(X), dtype=np.float64)
        model_classes = np.asarray(model.classes_, dtype=np.int64)
        out = np.zeros((X.shape[0], len(target_classes)), dtype=np.float64)
        for out_idx, cls in enumerate(target_classes):
            match = np.where(model_classes == cls)[0]
            if len(match):
                out[:, out_idx] = raw[:, int(match[0])]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


class ClassBiasTunedLogisticRegression:
    """Logistic regression with class-logit biases selected by inner CV."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        target_classes: list[int],
        bias_values: list[float],
        inner_splits: int = 5,
        class_weight: Any = None,
        seed: int = 2026,
    ) -> None:
        self.C = C
        self.target_classes = np.asarray(target_classes, dtype=np.int64)
        self.bias_values = np.asarray(bias_values, dtype=np.float64)
        self.inner_splits = inner_splits
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        inner_splits = min(self.inner_splits, int(np.bincount(y).min()))
        if inner_splits < 2:
            self.bias_ = np.zeros(len(self.classes_), dtype=np.float64)
        else:
            probs = np.zeros((len(y), len(self.classes_)), dtype=np.float64)
            splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=self.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
                model = self._base_model(seed=self.seed + fold_idx).fit(X[train_idx], y[train_idx])
                probs[val_idx] = self._align_probs(model, X[val_idx], self.classes_)
            self.bias_ = self._select_bias(probs, y)
        self.model_ = self._base_model(seed=self.seed + 10_000).fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._align_probs(self.model_, np.asarray(X, dtype=np.float32), self.classes_)
        return self._apply_bias(probs, self.bias_)

    def _select_bias(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        class_to_idx = {int(cls): idx for idx, cls in enumerate(self.classes_)}
        target_indices = [class_to_idx[int(cls)] for cls in self.target_classes if int(cls) in class_to_idx]
        if not target_indices:
            return np.zeros(len(self.classes_), dtype=np.float64)

        best_score = -np.inf
        best_bias = np.zeros(len(self.classes_), dtype=np.float64)
        grids = np.meshgrid(*([self.bias_values] * len(target_indices)), indexing="ij")
        for values in np.stack([grid.ravel() for grid in grids], axis=1):
            bias = np.zeros(len(self.classes_), dtype=np.float64)
            for idx, value in zip(target_indices, values):
                bias[idx] = float(value)
            pred = self._apply_bias(probs, bias).argmax(axis=1)
            score = 0.5 * f1_score(y, pred, average="macro") + 0.5 * balanced_accuracy_score(y, pred)
            if score > best_score:
                best_score = score
                best_bias = bias
        return best_bias

    def _base_model(self, *, seed: int):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=2000,
                        class_weight=self.class_weight,
                        random_state=seed,
                    ),
                ),
            ]
        )

    @staticmethod
    def _align_probs(model, X: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
        raw = np.asarray(model.predict_proba(X), dtype=np.float64)
        model_classes = np.asarray(model.classes_, dtype=np.int64)
        out = np.zeros((X.shape[0], len(target_classes)), dtype=np.float64)
        for out_idx, cls in enumerate(target_classes):
            match = np.where(model_classes == cls)[0]
            if len(match):
                out[:, out_idx] = raw[:, int(match[0])]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
        logits = np.log(np.maximum(probs, 1e-12)) + bias[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in results:
        metric = item["metrics"]
        rows.append(
            {
                "head_id": item["head_id"],
                "feature_file": item["feature_file"],
                "macro_f1": metric["macro_f1"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "selection_score": metric["selection_score"],
                "spec": item["spec"],
            }
        )
    rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    return {"results": rows, "best": rows[0] if rows else None}


def build_oof_ensemble(
    results: list[dict[str, Any]],
    metrics_dir: Path,
    predictions_dir: Path,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not results:
        return {}
    ensemble_config = dict(config.get("ensemble", {}))
    top_k = int(ensemble_config.get("top_k", min(3, len(results))))
    min_score = ensemble_config.get("min_selection_score")
    ranked = sorted(results, key=lambda item: item["metrics"]["selection_score"], reverse=True)
    selected = []
    for item in ranked:
        if len(selected) >= top_k:
            break
        if min_score is not None and item["metrics"]["selection_score"] < float(min_score):
            continue
        selected.append(item)
    if not selected:
        selected = ranked[:1]
    oof_files = []
    for item in selected:
        head_id = item["head_id"]
        oof_files.append(project_relative(predictions_dir / f"oof_{head_id}.npz"))
    loaded = [np.load(resolve_project_path(path), allow_pickle=True) for path in oof_files]
    try:
        y_true = loaded[0]["y_true"]
        class_names = [str(x) for x in loaded[0]["class_names"].tolist()]
        probs = average_probs([item["probs"] for item in loaded])
        y_pred = probs.argmax(axis=1)
        metrics = compute_classification_metrics(y_true, y_pred, class_names)
        out_path = predictions_dir / "oof_simple_average_ensemble.npz"
        np.savez_compressed(
            out_path,
            probs=probs,
            y_true=y_true,
            y_pred=y_pred,
            class_names=np.asarray(class_names, dtype=object),
            source_files=np.asarray(oof_files, dtype=object),
        )
        write_json(metrics, metrics_dir / "ensemble_simple_average_metrics.json")
        ensemble = {
            "method": "top_k_simple_average",
            "top_k": len(selected),
            "source_files": oof_files,
            "source_head_ids": [item["head_id"] for item in selected],
            "oof_path": project_relative(out_path),
            "metrics": metrics,
        }
        if bool(ensemble_config.get("weighted_search", True)) and len(selected) > 1:
            weighted = grid_weight_search(
                [item["probs"] for item in loaded],
                y_true,
                class_names,
                step=float(ensemble_config.get("weight_step", 0.1)),
                max_sources=int(ensemble_config.get("max_weighted_sources", 5)),
            )
            weighted_path = predictions_dir / "oof_weighted_ensemble.npz"
            np.savez_compressed(
                weighted_path,
                probs=weighted["probs"],
                y_true=y_true,
                y_pred=weighted["probs"].argmax(axis=1),
                class_names=np.asarray(class_names, dtype=object),
                source_files=np.asarray(oof_files[: len(weighted["weights"])], dtype=object),
                source_weights=np.asarray(weighted["weights"], dtype=np.float64),
            )
            write_json(weighted["metrics"], metrics_dir / "ensemble_weighted_metrics.json")
            ensemble["weighted"] = {
                "method": "non_negative_grid",
                "weight_step": float(ensemble_config.get("weight_step", 0.1)),
                "source_files": oof_files[: len(weighted["weights"])],
                "source_head_ids": [item["head_id"] for item in selected[: len(weighted["weights"])]],
                "weights": weighted["weights"],
                "oof_path": project_relative(weighted_path),
                "metrics": weighted["metrics"],
            }
            write_weighted_test_prediction(
                selected[: len(weighted["weights"])],
                predictions_dir=predictions_dir,
                weights=weighted["weights"],
            )
        write_simple_test_prediction(selected, predictions_dir=predictions_dir)
        return ensemble
    finally:
        for item in loaded:
            item.close()


def write_simple_test_prediction(selected: list[dict[str, Any]], *, predictions_dir: Path) -> None:
    """Write a simple-average test prediction when all source test files exist."""

    test_files = [predictions_dir / f"test_{item['head_id']}.npz" for item in selected]
    if not all(path.exists() for path in test_files):
        return
    loaded = [np.load(path, allow_pickle=True) for path in test_files]
    try:
        probs = average_probs([item["probs"] for item in loaded])
        class_names = [str(x) for x in loaded[0]["class_names"].tolist()]
        filenames = np.asarray(loaded[0]["test_filenames"]).astype(str)
        out_path = predictions_dir / "test_simple_average_ensemble.npz"
        np.savez_compressed(
            out_path,
            probs=probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=filenames,
            source_files=np.asarray([project_relative(path) for path in test_files], dtype=object),
        )
    finally:
        for item in loaded:
            item.close()


def write_weighted_test_prediction(
    selected: list[dict[str, Any]],
    *,
    predictions_dir: Path,
    weights: list[float],
) -> None:
    """Write a weighted-average test prediction when all source test files exist."""

    test_files = [predictions_dir / f"test_{item['head_id']}.npz" for item in selected]
    if not all(path.exists() for path in test_files):
        return
    loaded = [np.load(path, allow_pickle=True) for path in test_files]
    try:
        probs = weighted_probs([item["probs"] for item in loaded], weights)
        class_names = [str(x) for x in loaded[0]["class_names"].tolist()]
        filenames = np.asarray(loaded[0]["test_filenames"]).astype(str)
        out_path = predictions_dir / "test_weighted_ensemble.npz"
        np.savez_compressed(
            out_path,
            probs=probs,
            class_names=np.asarray(class_names, dtype=object),
            test_filenames=filenames,
            source_files=np.asarray([project_relative(path) for path in test_files], dtype=object),
            source_weights=np.asarray(weights, dtype=np.float64),
        )
    finally:
        for item in loaded:
            item.close()


def write_selection_report(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Scheme 01 Selection Report", ""]
    for row in summary.get("results", []):
        lines.append(
            f"- `{row['head_id']}`: macro_f1={row['macro_f1']:.6f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.6f}, "
            f"selection_score={row['selection_score']:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
