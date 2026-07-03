"""Frozen-feature pseudo-label head retraining for Scheme 03."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_final.features.store import load_feature_bundle
from ml_final.pseudo.common import DEFAULT_CLASS_NAMES, read_csv_rows, write_json
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths
from ml_final.utils.seed import set_seed


def train_pseudo_heads(
    config_path: str | Path,
    *,
    features: str | Path,
    pseudolabels: str | Path,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train weighted frozen-feature heads on true + pseudo labels."""

    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(
        {"config": config, "features": features, "pseudolabels": pseudolabels},
        context="Scheme03 pseudo head retraining",
    )
    set_seed(int(config.get("seed", 2026)))
    run_name = run_name or str(config.get("run_name", "scheme03_pseudo_heads"))
    features_root = resolve_project_path(features)
    pseudolabel_path = resolve_project_path(pseudolabels)
    if features_root is None or pseudolabel_path is None:
        raise ValueError("features and pseudolabels are required")
    run_dir = resolve_project_path(f"runs/scheme_03/{run_name}")
    if run_dir is None:
        raise ValueError("run_dir cannot be None")
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    models_dir = run_dir / "models"
    for directory in (metrics_dir, predictions_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(features_root.rglob("*.npz")) if features_root.exists() else []
    if not feature_files and not bool(config.get("allow_missing_features", False)):
        raise FileNotFoundError(f"no feature bundles found under {features_root}")

    class_names = [str(item) for item in config.get("class_names", DEFAULT_CLASS_NAMES)]
    pseudo_rows = load_pseudolabel_rows(pseudolabel_path, class_names=class_names)
    heads = list(config.get("heads", [{"name": "logreg_pseudo_C1", "family": "logreg", "C": 1.0}]))
    lambda_pseudo = float(config.get("lambda_pseudo", 0.2))
    if lambda_pseudo < 0:
        raise ValueError("lambda_pseudo must be non-negative")

    results = []
    skipped = []
    for feature_file in feature_files:
        bundle = load_feature_bundle(feature_file)
        if "test_features" not in bundle or "test_filenames" not in bundle:
            skipped.append(
                {
                    "feature_file": project_relative(feature_file),
                    "reason": "feature bundle has no test_features/test_filenames",
                }
            )
            continue
        X_true = np.asarray(bundle["train_features"], dtype=np.float64)
        y_true = np.asarray(bundle["train_labels"], dtype=np.int64)
        train_filenames = np.asarray(bundle["train_filenames"]).astype(str)
        test_filenames = np.asarray(bundle["test_filenames"]).astype(str)
        X_test = np.asarray(bundle["test_features"], dtype=np.float64)
        feature_classes = [str(item) for item in bundle["class_names"].tolist()]
        if feature_classes != class_names:
            raise ValueError(f"class_names mismatch in {feature_file}")
        pseudo_payload = build_pseudo_training_arrays(
            pseudo_rows,
            test_filenames=test_filenames,
            test_features=X_test,
            lambda_pseudo=lambda_pseudo,
            n_classes=len(class_names),
        )
        if pseudo_payload["num_matched"] == 0:
            skipped.append(
                {
                    "feature_file": project_relative(feature_file),
                    "reason": "no pseudo-label filenames matched test_filenames",
                }
            )
            continue
        X_train = np.concatenate([X_true, pseudo_payload["X"]], axis=0)
        y_train = np.concatenate([y_true, pseudo_payload["y"]], axis=0)
        weights = np.concatenate(
            [np.ones(len(y_true), dtype=np.float64), pseudo_payload["sample_weight"]],
            axis=0,
        )
        for head_cfg in heads:
            if str(head_cfg.get("family", "logreg")) != "logreg":
                skipped.append(
                    {
                        "feature_file": project_relative(feature_file),
                        "head": head_cfg.get("name", "unknown"),
                        "reason": "only logreg pseudo retraining is implemented",
                    }
                )
                continue
            model = fit_weighted_logreg(X_train, y_train, weights, head_cfg)
            probs = np.asarray(model.predict_proba(X_test), dtype=np.float64)
            head_id = f"{Path(feature_file).stem}__{head_cfg.get('name', 'logreg_pseudo')}"
            pred_path = predictions_dir / f"test_{head_id}.npz"
            np.savez_compressed(
                pred_path,
                probs=probs,
                class_names=np.asarray(class_names, dtype=object),
                test_filenames=test_filenames,
                head_id=np.asarray(head_id, dtype=object),
                feature_file=np.asarray(project_relative(feature_file), dtype=object),
                pseudo_rows=np.asarray(pseudo_payload["num_matched"], dtype=np.int64),
            )
            results.append(
                {
                    "head_id": head_id,
                    "feature_file": project_relative(feature_file),
                    "prediction_file": project_relative(pred_path),
                    "num_true": int(len(y_true)),
                    "num_pseudo_matched": int(pseudo_payload["num_matched"]),
                    "num_pseudo_expanded": int(len(pseudo_payload["y"])),
                    "lambda_pseudo": lambda_pseudo,
                    "head": dict(head_cfg),
                }
            )

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "features": project_relative(features_root),
        "pseudolabels": project_relative(pseudolabel_path),
        "class_names": class_names,
        "results": results,
        "skipped": skipped,
        "status": "ok" if results else "skipped",
    }
    write_json(metrics_dir / "summary.json", summary)
    write_retrain_report(metrics_dir / "pseudo_retrain_report.md", summary)
    return {"run_dir": project_relative(run_dir), "summary": summary}


def load_pseudolabel_rows(path: Path, *, class_names: list[str]) -> list[dict[str, Any]]:
    """Load selected_pseudolabels.csv rows."""

    if not path.exists():
        raise FileNotFoundError(f"pseudolabel file not found: {path}")
    rows = []
    for raw in read_csv_rows(path):
        soft = np.asarray([float(raw[f"soft_label_{idx}"]) for idx in range(len(class_names))], dtype=np.float64)
        soft = soft / np.maximum(soft.sum(), 1e-12)
        rows.append(
            {
                "filename": raw["filename"],
                "pseudo_label": raw["pseudo_label"],
                "sample_weight": float(raw["sample_weight"]),
                "soft": soft,
            }
        )
    return rows


def build_pseudo_training_arrays(
    rows: list[dict[str, Any]],
    *,
    test_filenames: np.ndarray,
    test_features: np.ndarray,
    lambda_pseudo: float,
    n_classes: int,
) -> dict[str, Any]:
    """Expand soft pseudo labels into weighted hard-label training rows."""

    index = {filename: idx for idx, filename in enumerate(test_filenames)}
    xs = []
    ys = []
    weights = []
    matched = 0
    for row in rows:
        idx = index.get(row["filename"])
        if idx is None:
            continue
        matched += 1
        base_weight = float(row["sample_weight"]) * lambda_pseudo
        for cls_idx in range(n_classes):
            weight = base_weight * float(row["soft"][cls_idx])
            if weight <= 0:
                continue
            xs.append(test_features[idx])
            ys.append(cls_idx)
            weights.append(weight)
    if not xs:
        return {
            "X": np.zeros((0, test_features.shape[1]), dtype=np.float64),
            "y": np.zeros(0, dtype=np.int64),
            "sample_weight": np.zeros(0, dtype=np.float64),
            "num_matched": matched,
        }
    return {
        "X": np.stack(xs, axis=0).astype(np.float64),
        "y": np.asarray(ys, dtype=np.int64),
        "sample_weight": np.asarray(weights, dtype=np.float64),
        "num_matched": matched,
    }


def fit_weighted_logreg(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, spec: dict[str, Any]):
    """Fit a scaled multinomial logistic regression with sample weights."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(spec.get("C", 1.0)),
                    max_iter=int(spec.get("max_iter", 2000)),
                    class_weight=spec.get("class_weight"),
                ),
            ),
        ]
    )
    return model.fit(X, y, clf__sample_weight=sample_weight)


def write_retrain_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a concise retraining report."""

    lines = [
        "# Scheme 03 Pseudo Head Retraining Report",
        "",
        f"- Status: `{summary['status']}`",
        f"- Prediction files: `{len(summary['results'])}`",
        f"- Skipped items: `{len(summary['skipped'])}`",
        "",
        "## Results",
        "",
    ]
    if summary["results"]:
        for row in summary["results"]:
            lines.append(
                f"- `{row['head_id']}`: true={row['num_true']}, "
                f"pseudo_matched={row['num_pseudo_matched']}, "
                f"pseudo_expanded={row['num_pseudo_expanded']}"
            )
    else:
        lines.append("- No pseudo head was trained. This is acceptable for no-test smoke runs.")
    if summary["skipped"]:
        lines.extend(["", "## Skipped", ""])
        for row in summary["skipped"]:
            target = row.get("feature_file", row.get("head", "unknown"))
            lines.append(f"- `{target}`: {row['reason']}")
    lines.extend(
        [
            "",
            "## Transductive Notice",
            "",
            "- Pseudo rows come from unlabeled test predictions when `test_features` are present.",
            "- True labels keep weight 1.0; pseudo labels use bounded sample weights times lambda.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
