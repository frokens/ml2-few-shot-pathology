"""Selection file writer for Scheme 01 frozen-feature experiments."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from ml_final.metrics.classification import write_json
from ml_final.utils.config import resolve_project_path
from ml_final.utils.paths import project_relative


PEFT_ELIGIBLE_BACKBONES = ("uni2_h", "virchow2", "h_optimus_0")


def select_scheme01(
    *,
    runs: str | Path,
    out: str | Path,
    max_peft_backbones: int = 2,
    min_delta_for_second: float = 0.005,
    peft_eligible_backbones: tuple[str, ...] = PEFT_ELIGIBLE_BACKBONES,
) -> dict[str, Any]:
    """Build Scheme01 selection artifacts consumed by Scheme02 and Scheme03."""

    runs_dir = resolve_project_path(runs)
    out_dir = resolve_project_path(out)
    if runs_dir is None or out_dir is None:
        raise ValueError("runs and out are required")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_scheme01_summaries(runs_dir)
    if not summaries:
        raise FileNotFoundError(f"no Scheme01 summaries found under {runs_dir}")
    rows = collect_candidate_rows(summaries)
    if not rows:
        raise ValueError("Scheme01 summaries contain no candidates")
    rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    best = rows[0]
    selected_backbones = choose_backbones(
        rows,
        max_backbones=max_peft_backbones,
        min_delta_for_second=min_delta_for_second,
    )
    peft_eligible = set(peft_eligible_backbones)
    peft_rows = [
        row
        for row in rows
        if str(row.get("backbone", "")) in peft_eligible
    ]
    selected_peft_backbones = choose_backbones(
        peft_rows,
        max_backbones=max_peft_backbones,
        min_delta_for_second=min_delta_for_second,
    )
    best_peft = peft_rows[0] if peft_rows else None
    teacher_predictions = collect_teacher_prediction_files(summaries, rows)
    best_settings = build_best_feature_settings(rows)
    best_heads = build_best_heads(rows)

    (out_dir / "scheme01_best_backbones.txt").write_text(
        "\n".join(selected_backbones) + "\n",
        encoding="utf-8",
    )
    (out_dir / "scheme01_top1_backbone.txt").write_text(f"{best['backbone']}\n", encoding="utf-8")
    (out_dir / "scheme01_best_peft_backbones.txt").write_text(
        "\n".join(selected_peft_backbones) + ("\n" if selected_peft_backbones else ""),
        encoding="utf-8",
    )
    (out_dir / "scheme01_top1_peft_backbone.txt").write_text(
        f"{best_peft['backbone']}\n" if best_peft else "",
        encoding="utf-8",
    )
    (out_dir / "scheme01_teacher_predictions.txt").write_text(
        "\n".join(teacher_predictions) + ("\n" if teacher_predictions else ""),
        encoding="utf-8",
    )
    write_json(best_settings, out_dir / "scheme01_best_feature_settings.json")
    write_json(best_heads, out_dir / "scheme01_best_heads.json")
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": project_relative(runs_dir),
        "best": best,
        "best_peft": best_peft,
        "selected_backbones": selected_backbones,
        "peft_eligible_backbones": list(peft_eligible_backbones),
        "selected_peft_backbones": selected_peft_backbones,
        "teacher_predictions": teacher_predictions,
        "results": rows,
    }
    write_json(report, out_dir / "scheme01_selection.json")
    write_scheme01_report(out_dir / "scheme01_report.md", report)
    return report


def load_scheme01_summaries(runs_dir: Path) -> list[dict[str, Any]]:
    """Load all Scheme01 summary files under a runs directory."""

    summaries = []
    for path in sorted(runs_dir.rglob("metrics/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_summary_path"] = project_relative(path)
        data["_run_dir"] = project_relative(path.parent.parent)
        summaries.append(data)
    return summaries


def collect_candidate_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect per-head and ensemble rows from Scheme01 summaries."""

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        for row in summary.get("results", []):
            rows.append(
                {
                    "candidate_id": row["head_id"],
                    "kind": "head",
                    "backbone": infer_backbone(row.get("feature_file", ""), row["head_id"]),
                    "macro_f1": float(row["macro_f1"]),
                    "balanced_accuracy": float(row["balanced_accuracy"]),
                    "selection_score": float(row["selection_score"]),
                    "feature_file": row.get("feature_file"),
                    "spec": row.get("spec", {}),
                    "oof_path": find_oof_for_head(summary, row["head_id"]),
                    "test_path": find_test_for_head(summary, row["head_id"]),
                    "summary_path": summary["_summary_path"],
                    "run_dir": summary["_run_dir"],
                }
            )
        ensemble = summary.get("ensemble", {})
        if ensemble:
            metrics = ensemble.get("metrics", {})
            rows.append(
                {
                    "candidate_id": f"{summary.get('run_name', 'scheme01')}::simple_average",
                    "kind": "ensemble",
                    "backbone": "ensemble",
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                    "selection_score": float(metrics.get("selection_score", 0.0)),
                    "source_head_ids": ensemble.get("source_head_ids", []),
                    "oof_path": ensemble.get("oof_path"),
                    "test_path": infer_test_from_oof(ensemble.get("oof_path")),
                    "summary_path": summary["_summary_path"],
                    "run_dir": summary["_run_dir"],
                }
            )
        weighted = ensemble.get("weighted", {}) if ensemble else {}
        if weighted:
            metrics = weighted.get("metrics", {})
            rows.append(
                {
                    "candidate_id": f"{summary.get('run_name', 'scheme01')}::weighted",
                    "kind": "weighted_ensemble",
                    "backbone": "ensemble",
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                    "selection_score": float(metrics.get("selection_score", 0.0)),
                    "source_head_ids": weighted.get("source_head_ids", []),
                    "weights": weighted.get("weights", []),
                    "oof_path": weighted.get("oof_path"),
                    "test_path": infer_test_from_oof(weighted.get("oof_path")),
                    "summary_path": summary["_summary_path"],
                    "run_dir": summary["_run_dir"],
                }
            )
    return rows


def choose_backbones(rows: list[dict[str, Any]], *, max_backbones: int, min_delta_for_second: float) -> list[str]:
    """Choose at most N non-ensemble backbones for PEFT."""

    best_by_backbone: dict[str, dict[str, Any]] = {}
    for row in rows:
        backbone = row.get("backbone")
        if not backbone or backbone == "ensemble":
            continue
        if backbone not in best_by_backbone or row["selection_score"] > best_by_backbone[backbone]["selection_score"]:
            best_by_backbone[backbone] = row
    ranked = sorted(best_by_backbone.values(), key=lambda row: row["selection_score"], reverse=True)
    if not ranked:
        return []
    selected = [ranked[0]["backbone"]]
    for row in ranked[1:]:
        if len(selected) >= max_backbones:
            break
        delta = ranked[0]["selection_score"] - row["selection_score"]
        if delta <= min_delta_for_second:
            selected.append(row["backbone"])
    return selected


def collect_teacher_prediction_files(summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    """Return the single best non-ensemble OOF/test pair for teacher use."""

    for row in rows:
        backbone = str(row.get("backbone", ""))
        if not backbone or backbone == "ensemble":
            continue
        oof_path = row.get("oof_path")
        test_path = row.get("test_path")
        if not oof_path:
            continue
        selected = [str(oof_path)]
        if test_path:
            selected.append(str(test_path))
        return selected
    fallback = []
    for summary in summaries:
        oof_files = [str(path) for path in summary.get("oof_prediction_files", [])]
        if not oof_files:
            continue
        fallback.append(oof_files[0])
        test_files = [str(path) for path in summary.get("test_prediction_files", [])]
        if test_files:
            fallback.append(test_files[0])
        break
    return fallback


def build_best_feature_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize best row per backbone."""

    out: dict[str, Any] = {}
    for row in rows:
        backbone = row.get("backbone")
        if not backbone or backbone == "ensemble":
            continue
        if backbone not in out:
            out[backbone] = row
    return out


def build_best_heads(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize best head rows by family."""

    out: dict[str, Any] = {}
    for row in rows:
        spec = row.get("spec") or {}
        family = spec.get("family")
        if not family:
            continue
        if family not in out:
            out[family] = row
    return out


def find_oof_for_head(summary: dict[str, Any], head_id: str) -> str | None:
    """Find an OOF file by head id."""

    suffix = f"oof_{head_id}.npz"
    for path in summary.get("oof_prediction_files", []):
        if str(path).endswith(suffix):
            return str(path)
    run_dir = resolve_project_path(summary["_run_dir"])
    candidate = run_dir / "predictions" / suffix if run_dir else None
    return project_relative(candidate) if candidate and candidate.exists() else None


def find_test_for_head(summary: dict[str, Any], head_id: str) -> str | None:
    """Find a test prediction file by head id."""

    suffix = f"test_{head_id}.npz"
    for path in summary.get("test_prediction_files", []):
        if str(path).endswith(suffix):
            return str(path)
    run_dir = resolve_project_path(summary["_run_dir"])
    candidate = run_dir / "predictions" / suffix if run_dir else None
    return project_relative(candidate) if candidate and candidate.exists() else None


def infer_test_from_oof(oof_path: str | None) -> str | None:
    """Infer conventional ensemble test path from an OOF path."""

    if not oof_path:
        return None
    path = resolve_project_path(oof_path)
    if path is None:
        return None
    name = path.name.replace("oof_", "test_", 1)
    candidate = path.with_name(name)
    return project_relative(candidate) if candidate.exists() else None


def infer_backbone(feature_file: str, head_id: str) -> str:
    """Infer backbone key from feature path/head id."""

    parts = Path(feature_file).parts
    if len(parts) >= 2:
        return parts[-2]
    return head_id.split("_", 1)[0]


def write_scheme01_report(path: Path, report: dict[str, Any]) -> None:
    """Write a concise human-readable Scheme01 selection report."""

    lines = [
        "# Scheme01 Selection",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Selected backbones overall: `{', '.join(report['selected_backbones'])}`",
        f"- PEFT-eligible backbones: `{', '.join(report['peft_eligible_backbones'])}`",
        f"- Selected PEFT backbones: `{', '.join(report['selected_peft_backbones'])}`",
        f"- Teacher prediction files: `{len(report['teacher_predictions'])}`",
        "",
        "## Top Candidates",
        "",
        "| rank | candidate | kind | backbone | macro-F1 | balanced acc | selection |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(report["results"][:20], start=1):
        lines.append(
            f"| {idx} | `{row['candidate_id']}` | `{row['kind']}` | `{row['backbone']}` | "
            f"{row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | {row['selection_score']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
