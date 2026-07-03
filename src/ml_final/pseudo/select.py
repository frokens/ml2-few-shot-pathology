"""Pseudo-label selection and OOF threshold simulation."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np

from ml_final.pseudo.common import (
    DEFAULT_CLASS_NAMES,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.paths import project_relative
from ml_final.utils.safety import assert_no_forbidden_reference_paths


def select_pseudolabels(
    config_path: str | Path,
    *,
    mode: str,
    teacher: str | Path,
    out: str | Path,
) -> dict[str, Any]:
    """Run OOF simulation or test pseudo-label selection."""

    if mode not in {"simulate", "select-test"}:
        raise ValueError("mode must be 'simulate' or 'select-test'")
    config = load_yaml(config_path)
    assert_no_forbidden_reference_paths(
        {"config": config, "teacher": teacher},
        context="Scheme03 pseudo-label selection",
    )
    teacher_dir = resolve_project_path(teacher)
    out_dir = resolve_project_path(out)
    if teacher_dir is None or out_dir is None:
        raise ValueError("teacher and out are required")
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [str(item) for item in config.get("class_names", DEFAULT_CLASS_NAMES)]
    if mode == "simulate":
        teacher_csv = teacher_dir / str(config.get("teacher_oof_csv", "teacher_oof_predictions.csv"))
        rows = load_teacher_rows(teacher_csv, class_names=class_names, require_labels=True)
        policy = choose_policy_from_oof(config, rows, class_names)
        selection = apply_policy(rows, class_names, policy, has_truth=True)
        policy_path = out_dir / "thresholds_selected.json"
        write_json(policy_path, policy)
    else:
        teacher_csv = teacher_dir / str(config.get("teacher_test_csv", "teacher_test_predictions.csv"))
        rows = load_teacher_rows(teacher_csv, class_names=class_names, require_labels=False)
        policy = load_policy_for_test(config)
        selection = apply_policy(rows, class_names, policy, has_truth=False)
        policy_path = None

    selected_path = out_dir / "selected_pseudolabels.csv"
    rejected_path = out_dir / "rejected_pseudolabels.csv"
    write_csv_rows(selected_path, selection["selected"], pseudo_fieldnames(class_names))
    write_csv_rows(rejected_path, selection["rejected"], rejected_fieldnames(class_names, has_truth=(mode == "simulate")))
    class_distribution = build_class_distribution(selection["selected"], class_names)
    write_json(out_dir / "class_distribution.json", class_distribution)
    effective_weight_estimates = estimate_effective_pseudo_weight(
        selection["selected"],
        class_names,
        lambdas=config.get("effective_weight_lambdas", []),
    )
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "teacher": project_relative(teacher_dir),
        "teacher_csv": project_relative(teacher_csv),
        "config": project_relative(resolve_project_path(config_path) or config_path),
        "policy": policy,
        "policy_path": project_relative(policy_path) if policy_path else None,
        "selected_pseudolabels": project_relative(selected_path),
        "rejected_pseudolabels": project_relative(rejected_path),
        "class_distribution": class_distribution,
        "num_candidates": len(rows),
        "num_selected": len(selection["selected"]),
        "num_rejected": len(selection["rejected"]),
        "simulation_metrics": selection.get("simulation_metrics"),
        "effective_pseudo_weight_estimates": effective_weight_estimates,
    }
    write_json(out_dir / "selection_summary.json", report)
    write_selection_report(out_dir / "selection_report.md", report, class_names)
    return report


def load_teacher_rows(
    path: Path,
    *,
    class_names: list[str],
    require_labels: bool,
) -> list[dict[str, Any]]:
    """Load teacher prediction CSV rows with parsed numeric fields."""

    if not path.exists():
        raise FileNotFoundError(f"teacher predictions not found: {path}")
    rows = []
    for raw in read_csv_rows(path):
        probs = np.asarray([float(raw[f"prob_{idx}"]) for idx in range(len(class_names))], dtype=np.float64)
        probs = probs / np.maximum(probs.sum(), 1e-12)
        row: dict[str, Any] = {
            "filename": raw["filename"],
            "pred_label": raw["pred_label"],
            "prob_top1": float(raw["prob_top1"]),
            "prob_top2": float(raw["prob_top2"]),
            "margin": float(raw["margin"]),
            "entropy": float(raw["entropy"]),
            "teacher_agreement": float(raw.get("teacher_agreement", 1.0) or 1.0),
            "probs": probs,
        }
        if require_labels:
            if not raw.get("true_label"):
                raise ValueError(f"OOF teacher row missing true_label: {raw['filename']}")
            row["true_label"] = raw["true_label"]
            row["correct"] = int(raw.get("correct", "0") or 0)
        rows.append(row)
    return rows


def choose_policy_from_oof(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Choose thresholds from OOF simulation according to configured grid."""

    policy_source = str(config.get("policy_source", "search")).lower()
    if policy_source == "fixed":
        policy = choose_fixed_policy_from_oof(config, rows, class_names)
    elif policy_source in {"thresholds_from", "from_thresholds"}:
        policy = load_policy_for_test(config)
        policy["policy_source"] = "thresholds_from"
        policy["selected_by"] = "thresholds_from_oof_simulation"
    elif bool(config.get("per_class_policy", True)):
        policy = choose_per_class_policy_from_oof(config, rows, class_names)
    else:
        policy = choose_global_policy_from_oof(config, rows, class_names)
    policy = apply_policy_overrides(policy, config)
    if policy.get("balanced_soft_expansion", {}).get("enabled"):
        selection = apply_policy(rows, class_names, policy, has_truth=True)
        policy["selection_metrics"] = selection["simulation_metrics"]
        policy["class_policy_metrics"] = compute_class_policy_metrics(selection["selected"], rows, class_names)
    return policy


def choose_fixed_policy_from_oof(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Evaluate an explicitly configured policy on OOF rows."""

    policy = build_fixed_policy(config, class_names)
    policy["selected_by"] = "fixed_oof_simulation"
    selection = apply_policy(rows, class_names, policy, has_truth=True)
    policy["class_policy_metrics"] = compute_class_policy_metrics(selection["selected"], rows, class_names)
    policy["selection_metrics"] = selection["simulation_metrics"]
    policy["pruned_classes"] = []
    if bool(config.get("prune_failed_classes", False)):
        pruned = prune_policy_classes(policy, config, selection["selected"], rows, class_names)
        if pruned:
            policy["pruned_classes"] = pruned
            selection = apply_policy(rows, class_names, policy, has_truth=True)
            policy["class_policy_metrics"] = compute_class_policy_metrics(selection["selected"], rows, class_names)
            policy["selection_metrics"] = selection["simulation_metrics"]
    return policy


def build_fixed_policy(config: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    """Normalize a fixed pseudo-label policy from config."""

    raw_policy = config.get("policy")
    if not isinstance(raw_policy, dict):
        raise ValueError("policy_source=fixed requires a policy mapping")
    policy = {
        "prob_top1": class_float_mapping(raw_policy.get("prob_top1", 1.0), class_names, default=1.0),
        "margin": class_float_mapping(raw_policy.get("margin", 0.0), class_names, default=0.0),
        "entropy_max": class_float_mapping(raw_policy.get("entropy_max", 10.0), class_names, default=10.0),
        "quota": class_int_mapping(raw_policy.get("quota", 10**9), class_names, default=10**9),
        "min_teacher_agreement": class_float_mapping(
            raw_policy.get("min_teacher_agreement", config.get("min_teacher_agreement", 0.0)),
            class_names,
            default=float(config.get("min_teacher_agreement", 0.0)),
        ),
        "sample_weight": dict(
            raw_policy.get(
                "sample_weight",
                config.get("sample_weight", {"min_weight": 0.1, "max_weight": 0.5, "weight_function": "linear"}),
            )
        ),
        "policy_source": "fixed",
    }
    return policy


def class_float_mapping(value: Any, class_names: list[str], *, default: float) -> dict[str, float]:
    """Return a per-class float mapping."""

    if isinstance(value, dict):
        return {name: float(value.get(name, default)) for name in class_names}
    return {name: float(value) for name in class_names}


def class_int_mapping(value: Any, class_names: list[str], *, default: int) -> dict[str, int]:
    """Return a per-class integer mapping."""

    if isinstance(value, dict):
        return {name: int(value.get(name, default)) for name in class_names}
    return {name: int(value) for name in class_names}


def compute_class_policy_metrics(
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute per-predicted-class OOF precision for the current policy."""

    metrics: dict[str, Any] = {}
    for class_name in class_names:
        selected_rows = [row for row in selected if row["pseudo_label"] == class_name]
        correct = sum(int(row.get("correct", 0)) for row in selected_rows)
        precision = correct / len(selected_rows) if selected_rows else 0.0
        metrics[class_name] = {
            "candidates": sum(1 for row in rows if str(row["pred_label"]) == class_name),
            "selected": len(selected_rows),
            "correct": correct,
            "precision": float(precision),
            "precision_lower_bound": wilson_lower_bound(correct, len(selected_rows)),
        }
    return metrics


def prune_policy_classes(
    policy: dict[str, Any],
    config: dict[str, Any],
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> list[str]:
    """Set quota=0 for fixed-policy classes that fail configured OOF gates."""

    raw_gates = config.get("pruning", {})
    if not isinstance(raw_gates, dict):
        raw_gates = {}
    metrics = compute_class_policy_metrics(selected, rows, class_names)
    pruned = []
    for class_name in class_names:
        gate = raw_gates.get(class_name, {})
        if not isinstance(gate, dict):
            gate = {}
        min_precision = float(gate.get("min_precision", config.get("prune_min_precision", 1.0)))
        min_selected = int(gate.get("min_selected", config.get("prune_min_selected", 1)))
        class_metrics = metrics[class_name]
        if class_metrics["selected"] < min_selected or class_metrics["precision"] < min_precision:
            policy["quota"][class_name] = 0
            pruned.append(class_name)
    return pruned


def choose_global_policy_from_oof(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Choose one global threshold tuple and copy it to all classes."""

    grid = config.get("policy_grid", {})
    prob_grid = [float(item) for item in grid.get("prob_top1", [0.9])]
    margin_grid = [float(item) for item in grid.get("margin", [0.0])]
    entropy_grid = [float(item) for item in grid.get("entropy_max", [10.0])]
    quota_grid = [int(item) for item in grid.get("quota_per_class", [10**9])]
    agreement_grid = [float(item) for item in grid.get("min_teacher_agreement", [config.get("min_teacher_agreement", 0.0)])]
    precision_target = float(config.get("precision_target", 0.95))
    lower_bound_target = float(config.get("precision_lower_bound_target", 0.0))
    min_coverage = int(config.get("min_selected", 0))
    best: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    for prob in prob_grid:
        for margin in margin_grid:
            for entropy_max in entropy_grid:
                for quota in quota_grid:
                    for agreement in agreement_grid:
                        policy = build_global_policy(
                            class_names,
                            prob_top1=prob,
                            margin=margin,
                            entropy_max=entropy_max,
                            quota_per_class=quota,
                            min_teacher_agreement=agreement,
                            config=config,
                        )
                        selection = apply_policy(rows, class_names, policy, has_truth=True)
                        metrics = selection["simulation_metrics"]
                        candidate = {
                            "policy": policy,
                            "metrics": metrics,
                            "score": simulation_score(metrics),
                        }
                        feasible = (
                            metrics["selected"] >= min_coverage
                            and metrics["macro_precision"] >= precision_target
                        )
                        if best is None or candidate["score"] > best["score"]:
                            best = candidate
                        if feasible and (best_feasible is None or candidate["score"] > best_feasible["score"]):
                            best_feasible = candidate
    chosen = best_feasible or best
    if chosen is None:
        raise ValueError("policy grid produced no candidates")
    policy = dict(chosen["policy"])
    policy["selected_by"] = "oof_simulation"
    policy["selection_metrics"] = chosen["metrics"]
    return policy


def choose_per_class_policy_from_oof(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Choose class-specific thresholds using OOF precision simulation.

    Macro-F1 and balanced accuracy are sensitive to minority-class recall.  A
    single global threshold often over-selects easy/common classes while
    discarding rare classes.  This routine searches the same threshold grid per
    predicted class, keeps precision as the first constraint, and then maximizes
    accepted coverage within that class.
    """

    grid = config.get("policy_grid", {})
    prob_grid = [float(item) for item in grid.get("prob_top1", [0.9])]
    margin_grid = [float(item) for item in grid.get("margin", [0.0])]
    entropy_grid = [float(item) for item in grid.get("entropy_max", [10.0])]
    quota_grid = [int(item) for item in grid.get("quota_per_class", [10**9])]
    agreement_grid = [float(item) for item in grid.get("min_teacher_agreement", [config.get("min_teacher_agreement", 0.0)])]
    precision_target = float(config.get("precision_target", 0.95))
    lower_bound_target = float(config.get("precision_lower_bound_target", 0.0))
    min_selected_per_class = int(config.get("min_selected_per_class", 0))
    class_choices: dict[str, Any] = {}
    policy = {
        "prob_top1": {},
        "margin": {},
        "entropy_max": {},
        "quota": {},
        "min_teacher_agreement": {},
        "sample_weight": dict(
            config.get(
                "sample_weight",
                {"min_weight": 0.1, "max_weight": 0.5, "weight_function": "linear"},
            )
        ),
    }
    for class_name in class_names:
        class_rows = [row for row in rows if str(row["pred_label"]) == class_name]
        best: dict[str, Any] | None = None
        best_feasible: dict[str, Any] | None = None
        for prob in prob_grid:
            for margin in margin_grid:
                for entropy_max in entropy_grid:
                    for quota in quota_grid:
                        for agreement in agreement_grid:
                            metrics = evaluate_class_policy(
                                class_rows,
                                class_name=class_name,
                                prob_top1=prob,
                                margin=margin,
                                entropy_max=entropy_max,
                                quota_per_class=quota,
                                min_teacher_agreement=agreement,
                            )
                            candidate = {
                                "prob_top1": prob,
                                "margin": margin,
                                "entropy_max": entropy_max,
                                "quota": quota,
                                "min_teacher_agreement": agreement,
                                "metrics": metrics,
                                "score": class_policy_score(metrics),
                            }
                            feasible = (
                                metrics["selected"] >= min_selected_per_class
                                and metrics["precision"] >= precision_target
                                and metrics["precision_lower_bound"] >= lower_bound_target
                            )
                            if best is None or candidate["score"] > best["score"]:
                                best = candidate
                            if feasible and (best_feasible is None or candidate["score"] > best_feasible["score"]):
                                best_feasible = candidate
        chosen = best_feasible or best
        if chosen is None:
            chosen = {
                "prob_top1": 1.0,
                "margin": 1.0,
                "entropy_max": 0.0,
                "quota": 0,
                "min_teacher_agreement": 1.0,
                "metrics": {
                    "selected": 0,
                    "correct": 0,
                    "precision": 0.0,
                    "precision_lower_bound": 0.0,
                    "candidates": 0,
                },
                "score": 0.0,
            }
        policy["prob_top1"][class_name] = float(chosen["prob_top1"])
        policy["margin"][class_name] = float(chosen["margin"])
        policy["entropy_max"][class_name] = float(chosen["entropy_max"])
        policy["quota"][class_name] = int(chosen["quota"])
        policy["min_teacher_agreement"][class_name] = float(chosen["min_teacher_agreement"])
        class_choices[class_name] = {
            "selected": int(chosen["metrics"]["selected"]),
            "correct": int(chosen["metrics"]["correct"]),
            "precision": float(chosen["metrics"]["precision"]),
            "precision_lower_bound": float(chosen["metrics"]["precision_lower_bound"]),
            "candidates": int(chosen["metrics"]["candidates"]),
            "prob_top1": float(chosen["prob_top1"]),
            "margin": float(chosen["margin"]),
            "entropy_max": float(chosen["entropy_max"]),
            "quota": int(chosen["quota"]),
            "min_teacher_agreement": float(chosen["min_teacher_agreement"]),
            "feasible": bool(
                chosen["metrics"]["selected"] >= min_selected_per_class
                and chosen["metrics"]["precision"] >= precision_target
                and chosen["metrics"]["precision_lower_bound"] >= lower_bound_target
            ),
        }
    selection = apply_policy(rows, class_names, policy, has_truth=True)
    policy["selected_by"] = "per_class_oof_simulation"
    policy["class_policy_metrics"] = class_choices
    policy["selection_metrics"] = selection["simulation_metrics"]
    return policy


def evaluate_class_policy(
    rows: list[dict[str, Any]],
    *,
    class_name: str,
    prob_top1: float,
    margin: float,
    entropy_max: float,
    quota_per_class: int,
    min_teacher_agreement: float,
) -> dict[str, Any]:
    """Evaluate one class-specific threshold tuple on OOF rows."""

    accepted = []
    sorted_rows = sorted(rows, key=lambda row: (row["prob_top1"], row["margin"]), reverse=True)
    for row in sorted_rows:
        if len(accepted) >= quota_per_class:
            break
        if row["prob_top1"] < prob_top1:
            continue
        if row["margin"] < margin:
            continue
        if row["entropy"] > entropy_max:
            continue
        if row["teacher_agreement"] < min_teacher_agreement:
            continue
        accepted.append(row)
    correct = sum(int(row.get("true_label") == class_name) for row in accepted)
    precision = correct / len(accepted) if accepted else 0.0
    return {
        "selected": len(accepted),
        "correct": correct,
        "precision": float(precision),
        "precision_lower_bound": wilson_lower_bound(correct, len(accepted)),
        "candidates": len(rows),
    }


def class_policy_score(metrics: dict[str, Any]) -> float:
    """Precision-first class policy score with a small coverage reward."""

    coverage = metrics["selected"] / max(metrics["candidates"], 1)
    return float(metrics["precision_lower_bound"] + 0.05 * coverage)


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    """Return the Wilson-score lower confidence bound for a binomial rate."""

    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = proportion + (z * z / (2.0 * total))
    radius = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total)
    return float(max(0.0, (centre - radius) / denominator))


def simulation_score(metrics: dict[str, Any]) -> float:
    """Rank OOF simulation policies by precision first, then coverage."""

    coverage = metrics["selected"] / max(metrics["candidates"], 1)
    return float(metrics["macro_precision"] + 0.05 * coverage)


def build_global_policy(
    class_names: list[str],
    *,
    prob_top1: float,
    margin: float,
    entropy_max: float,
    quota_per_class: int,
    min_teacher_agreement: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a class-balanced global threshold policy."""

    return {
        "prob_top1": {name: float(prob_top1) for name in class_names},
        "margin": {name: float(margin) for name in class_names},
        "entropy_max": float(entropy_max),
        "quota": {name: int(quota_per_class) for name in class_names},
        "min_teacher_agreement": {name: float(min_teacher_agreement) for name in class_names},
        "sample_weight": dict(
            config.get(
                "sample_weight",
                {"min_weight": 0.1, "max_weight": 0.5, "weight_function": "linear"},
            )
        ),
    }


def load_policy_for_test(config: dict[str, Any]) -> dict[str, Any]:
    """Load a selection policy from simulation output or config."""

    from_path = config.get("thresholds_from")
    if from_path:
        path = resolve_project_path(from_path)
        if path is None or not path.exists():
            raise FileNotFoundError(f"thresholds_from not found: {from_path}")
        import json

        return apply_policy_overrides(json.loads(path.read_text(encoding="utf-8")), config)
    policy = config.get("policy")
    if not policy:
        raise ValueError("select-test requires thresholds_from or policy in config")
    return apply_policy_overrides(dict(policy), config)


def apply_policy_overrides(policy: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Apply opt-in policy overlays that should not affect legacy configs."""

    out = dict(policy)
    expansion = config.get("balanced_soft_expansion")
    if isinstance(expansion, dict):
        out["balanced_soft_expansion"] = dict(expansion)
    return out


def apply_policy(
    rows: list[dict[str, Any]],
    class_names: list[str],
    policy: dict[str, Any],
    *,
    has_truth: bool,
) -> dict[str, Any]:
    """Apply threshold/margin/entropy/agreement/quota policy."""

    if policy.get("balanced_soft_expansion", {}).get("enabled"):
        return apply_balanced_soft_expansion_policy(rows, class_names, policy, has_truth=has_truth)

    accepted = []
    rejected = []
    quota = {name: int(policy.get("quota", {}).get(name, 10**9)) for name in class_names}
    counts = {name: 0 for name in class_names}
    sorted_rows = sorted(rows, key=lambda row: (row["prob_top1"], row["margin"]), reverse=True)
    for row in sorted_rows:
        label = str(row["pred_label"])
        reasons = reject_reasons(row, label, policy)
        if not reasons and counts.get(label, 0) >= quota.get(label, 10**9):
            reasons.append("quota_exceeded")
        pseudo = to_pseudo_row(row, class_names, policy)
        if has_truth:
            pseudo["true_label"] = row["true_label"]
            pseudo["correct"] = int(row["pred_label"] == row["true_label"])
        if reasons:
            pseudo["reject_reason"] = ";".join(reasons)
            rejected.append(pseudo)
        else:
            counts[label] += 1
            accepted.append(pseudo)
    metrics = compute_simulation_metrics(accepted, class_names, total_candidates=len(rows)) if has_truth else None
    return {"selected": accepted, "rejected": rejected, "simulation_metrics": metrics}


def apply_balanced_soft_expansion_policy(
    rows: list[dict[str, Any]],
    class_names: list[str],
    policy: dict[str, Any],
    *,
    has_truth: bool,
) -> dict[str, Any]:
    """Select a class-balanced ranked pseudo pool with tiered confidence weights."""

    expansion = policy.get("balanced_soft_expansion", {}) or {}
    expanded_quota = int(expansion.get("expanded_quota", expansion.get("quota", 10**9)))
    accepted = []
    rejected = []
    selected_filenames = set()
    for label in class_names:
        class_rows = [row for row in rows if str(row["pred_label"]) == label]
        class_rows = sorted(class_rows, key=lambda row: (row["prob_top1"], row["margin"]), reverse=True)
        for rank, row in enumerate(class_rows, start=1):
            multiplier = rank_weight_multiplier(rank, expansion)
            pseudo = to_pseudo_row(row, class_names, policy, weight_multiplier=multiplier)
            if has_truth:
                pseudo["true_label"] = row["true_label"]
                pseudo["correct"] = int(row["pred_label"] == row["true_label"])
            if rank <= expanded_quota:
                accepted.append(pseudo)
                selected_filenames.add(row["filename"])
            else:
                pseudo["reject_reason"] = "quota_exceeded"
                rejected.append(pseudo)
    for row in rows:
        if row["filename"] in selected_filenames:
            continue
        if str(row["pred_label"]) in class_names:
            continue
        pseudo = to_pseudo_row(row, class_names, policy)
        if has_truth:
            pseudo["true_label"] = row["true_label"]
            pseudo["correct"] = int(row["pred_label"] == row["true_label"])
        pseudo["reject_reason"] = "unknown_pred_label"
        rejected.append(pseudo)
    accepted = sorted(accepted, key=lambda row: (float(row["prob_top1"]), float(row["margin"])), reverse=True)
    rejected = sorted(rejected, key=lambda row: (float(row["prob_top1"]), float(row["margin"])), reverse=True)
    metrics = compute_simulation_metrics(accepted, class_names, total_candidates=len(rows)) if has_truth else None
    return {"selected": accepted, "rejected": rejected, "simulation_metrics": metrics}


def rank_weight_multiplier(rank: int, expansion: dict[str, Any]) -> float:
    """Return the configured sample-weight multiplier for a 1-based per-class rank."""

    tiers = expansion.get("rank_weight_multipliers", [])
    for tier in tiers:
        start = int(tier.get("start_rank", tier.get("start", 1)))
        end = int(tier.get("end_rank", tier.get("end", start)))
        if start <= rank <= end:
            return float(tier.get("multiplier", 1.0))
    core_quota = int(expansion.get("core_quota", 0))
    if core_quota > 0 and rank <= core_quota:
        return 1.0
    return float(expansion.get("tail_multiplier", 1.0))


def reject_reasons(row: dict[str, Any], label: str, policy: dict[str, Any]) -> list[str]:
    """Return human-readable rejection reasons for a candidate."""

    reasons = []
    prob_threshold = float(policy.get("prob_top1", {}).get(label, 1.0))
    margin_threshold = float(policy.get("margin", {}).get(label, 0.0))
    entropy_policy = policy.get("entropy_max", 10.0)
    if isinstance(entropy_policy, dict):
        entropy_max = float(entropy_policy.get(label, 10.0))
    else:
        entropy_max = float(entropy_policy)
    agreement_policy = policy.get("min_teacher_agreement", 0.0)
    if isinstance(agreement_policy, dict):
        min_agreement = float(agreement_policy.get(label, 0.0))
    else:
        min_agreement = float(agreement_policy)
    if row["prob_top1"] < prob_threshold:
        reasons.append("prob_top1_below_threshold")
    if row["margin"] < margin_threshold:
        reasons.append("margin_below_threshold")
    if row["entropy"] > entropy_max:
        reasons.append("entropy_above_threshold")
    if row["teacher_agreement"] < min_agreement:
        reasons.append("agreement_below_threshold")
    return reasons


def to_pseudo_row(
    row: dict[str, Any],
    class_names: list[str],
    policy: dict[str, Any],
    *,
    weight_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Convert a teacher row to the required pseudo-label CSV schema."""

    weight = compute_sample_weight(row, policy) * float(weight_multiplier)
    result: dict[str, Any] = {
        "filename": row["filename"],
        "pseudo_label": row["pred_label"],
        "prob_top1": f"{row['prob_top1']:.10f}",
        "prob_top2": f"{row['prob_top2']:.10f}",
        "margin": f"{row['margin']:.10f}",
        "entropy": f"{row['entropy']:.10f}",
        "teacher_agreement": f"{row['teacher_agreement']:.10f}",
        "sample_weight": f"{weight:.10f}",
    }
    for cls_idx, _name in enumerate(class_names):
        result[f"soft_label_{cls_idx}"] = f"{float(row['probs'][cls_idx]):.10f}"
    return result


def compute_sample_weight(row: dict[str, Any], policy: dict[str, Any]) -> float:
    """Map confidence to bounded pseudo-label sample weight."""

    config = policy.get("sample_weight", {})
    min_weight = float(config.get("min_weight", 0.1))
    max_weight = float(config.get("max_weight", 0.5))
    if max_weight < min_weight:
        raise ValueError("sample_weight.max_weight must be >= min_weight")
    label = str(row["pred_label"])
    threshold = float(policy.get("prob_top1", {}).get(label, 0.0))
    denom = max(1.0 - threshold, 1e-12)
    scaled = (row["prob_top1"] - threshold) / denom
    scaled = min(max(float(scaled), 0.0), 1.0)
    return min_weight + scaled * (max_weight - min_weight)


def compute_simulation_metrics(
    selected: list[dict[str, Any]],
    class_names: list[str],
    *,
    total_candidates: int,
) -> dict[str, Any]:
    """Compute pseudo-label precision metrics on OOF simulation rows."""

    per_class = {}
    precisions = []
    for label in class_names:
        rows = [row for row in selected if row["pseudo_label"] == label]
        correct = sum(int(row.get("correct", 0)) for row in rows)
        precision = float(correct / len(rows)) if rows else 0.0
        per_class[label] = {"selected": len(rows), "correct": correct, "precision": precision}
        if rows:
            precisions.append(precision)
    return {
        "selected": len(selected),
        "candidates": total_candidates,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "per_class": per_class,
    }


def build_class_distribution(
    selected: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any]:
    """Return selected pseudo-label counts by class."""

    counts = {label: 0 for label in class_names}
    for row in selected:
        counts[row["pseudo_label"]] += 1
    total = sum(counts.values())
    fractions = {key: (value / total if total else 0.0) for key, value in counts.items()}
    return {"total": total, "counts": counts, "fractions": fractions}


def estimate_effective_pseudo_weight(
    selected: list[dict[str, Any]],
    class_names: list[str],
    *,
    lambdas: list[Any],
) -> dict[str, Any]:
    """Estimate pseudo loss weight after multiplying CSV weights by lambda."""

    sample_weights = [float(row.get("sample_weight", 0.0)) for row in selected]
    raw_total = float(sum(sample_weights))
    by_class_raw = {
        label: float(sum(float(row.get("sample_weight", 0.0)) for row in selected if row["pseudo_label"] == label))
        for label in class_names
    }
    estimates: dict[str, Any] = {
        "raw_sample_weight_sum": raw_total,
        "raw_sample_weight_mean": float(raw_total / len(sample_weights)) if sample_weights else 0.0,
        "raw_sample_weight_by_class": by_class_raw,
        "lambda_scaled": {},
    }
    for item in lambdas or []:
        lam = float(item)
        estimates["lambda_scaled"][f"{lam:g}"] = {
            "effective_weight_sum": raw_total * lam,
            "effective_weight_by_class": {label: value * lam for label, value in by_class_raw.items()},
        }
    return estimates


def pseudo_fieldnames(class_names: list[str]) -> list[str]:
    return [
        "filename",
        "pseudo_label",
        "prob_top1",
        "prob_top2",
        "margin",
        "entropy",
        "teacher_agreement",
        "sample_weight",
    ] + [f"soft_label_{idx}" for idx in range(len(class_names))]


def rejected_fieldnames(class_names: list[str], *, has_truth: bool) -> list[str]:
    fields = pseudo_fieldnames(class_names) + ["reject_reason"]
    if has_truth:
        fields += ["true_label", "correct"]
    return fields


def write_selection_report(path: Path, report: dict[str, Any], class_names: list[str]) -> None:
    """Write a concise selection report."""

    lines = [
        "# Scheme 03 Pseudo-Label Selection Report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Candidates: `{report['num_candidates']}`",
        f"- Selected: `{report['num_selected']}`",
        f"- Rejected: `{report['num_rejected']}`",
        f"- Policy source: `{report['policy'].get('policy_source', 'search')}`",
        f"- Selected by: `{report['policy'].get('selected_by', 'unknown')}`",
        "",
        "## Class Distribution",
        "",
    ]
    if report["policy"].get("pruned_classes"):
        lines.insert(-3, f"- Pruned classes: `{', '.join(report['policy']['pruned_classes'])}`")
    for label in class_names:
        count = report["class_distribution"]["counts"][label]
        frac = report["class_distribution"]["fractions"][label]
        lines.append(f"- `{label}`: {count} ({frac:.4f})")
    if report.get("simulation_metrics"):
        lines.extend(["", "## OOF Simulation Precision", ""])
        metrics = report["simulation_metrics"]
        lines.append(f"- macro precision: `{metrics['macro_precision']:.6f}`")
        for label in class_names:
            item = metrics["per_class"][label]
            lines.append(
                f"- `{label}`: selected={item['selected']}, precision={item['precision']:.6f}"
            )
    if report.get("effective_pseudo_weight_estimates"):
        estimates = report["effective_pseudo_weight_estimates"]
        lines.extend(["", "## Effective Pseudo Weight Estimate", ""])
        lines.append(f"- raw sample-weight sum: `{estimates['raw_sample_weight_sum']:.6f}`")
        for lam, item in estimates.get("lambda_scaled", {}).items():
            lines.append(f"- lambda `{lam}` effective-weight sum: `{item['effective_weight_sum']:.6f}`")
    lines.extend(
        [
            "",
            "## Transductive Notice",
            "",
            "- `select-test` uses unlabeled test images through teacher predictions only.",
            "- Hidden labels are not used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
