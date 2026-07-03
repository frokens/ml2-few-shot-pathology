"""Selection helpers for Scheme 02 PEFT experiments."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from ml_final.metrics.classification import write_json
from ml_final.utils.config import resolve_project_path
from ml_final.utils.paths import project_relative


def select_scheme02(
    *,
    runs: str | Path,
    out: str | Path,
    scheme01: str | Path | None = None,
    min_improvement: float = 0.005,
) -> dict[str, Any]:
    """Select Scheme02 PEFT candidates and write downstream artifacts."""

    runs_dir = resolve_project_path(runs)
    out_dir = resolve_project_path(out)
    if runs_dir is None or out_dir is None:
        raise ValueError("runs and out are required")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_scheme02_rows(runs_dir)
    if not rows:
        raise FileNotFoundError(f"no Scheme02 summary rows found under {runs_dir}")
    rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    scheme01_score = read_scheme01_score(scheme01)
    best = rows[0]
    passed_gate = scheme01_score is None or best["selection_score"] >= scheme01_score + min_improvement
    best_txt = out_dir / "scheme02_best_peft.txt"
    best_txt.write_text(best["experiment_id"] + "\n", encoding="utf-8")
    write_json(best, out_dir / "scheme02_best_peft.json")
    teacher_files = []
    if best.get("test_prediction_kind") == "single_refit" and best.get("test_path"):
        for key in ("oof_path", "test_path"):
            if best.get(key):
                teacher_files.append(best[key])
    if not passed_gate:
        teacher_files = []
    (out_dir / "scheme02_teacher_predictions.txt").write_text(
        "\n".join(teacher_files) + ("\n" if teacher_files else ""),
        encoding="utf-8",
    )
    fusion = {
        "passed_gate": passed_gate,
        "scheme01_score": scheme01_score,
        "min_improvement": min_improvement,
        "best_scheme02": best,
        "teacher_prediction_files": teacher_files,
    }
    write_json(fusion, out_dir / "scheme02_fusion_candidates.json")
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": project_relative(runs_dir),
        "scheme01": project_relative(resolve_project_path(scheme01)) if scheme01 else None,
        "scheme01_score": scheme01_score,
        "min_improvement": min_improvement,
        "passed_gate": passed_gate,
        "best": best,
        "results": rows,
    }
    write_json(report, out_dir / "scheme02_selection.json")
    write_scheme02_report(out_dir / "scheme02_report.md", report)
    return report


def collect_scheme02_rows(runs_dir: Path) -> list[dict[str, Any]]:
    """Collect compact Scheme02 rows from summary files."""

    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.rglob("metrics/summary.json")):
        run_dir = path.parent.parent
        if is_nonformal_scheme02_run(run_dir):
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if is_nonformal_scheme02_summary(summary):
            continue
        for row in summary.get("results", []):
            rows.append(
                {
                    **row,
                    "resolved_config": summary.get("resolved_config"),
                    "summary_path": project_relative(path),
                    "run_dir": project_relative(path.parent.parent),
                }
            )
    return rows


def is_nonformal_scheme02_run(run_dir: Path) -> bool:
    """Return True for smoke/debug/incomplete Scheme02 runs."""

    name = run_dir.name.lower()
    return any(token in name for token in ("smoke", "debug", "incomplete", "tmp"))


def is_nonformal_scheme02_summary(summary: dict[str, Any]) -> bool:
    """Return True when summary metadata marks a run as non-formal."""

    run_name = str(summary.get("run_name", "")).lower()
    if any(token in run_name for token in ("smoke", "debug", "incomplete", "tmp")):
        return True
    return bool(summary.get("smoke") or summary.get("debug") or summary.get("incomplete"))


def read_scheme01_score(path: str | Path | None) -> float | None:
    """Read best Scheme01 selection score from report JSON or Markdown path."""

    if path is None:
        candidate = resolve_project_path("artifacts/selection/scheme01_selection.json")
    else:
        candidate = resolve_project_path(path)
    if candidate is None or not candidate.exists():
        return None
    if candidate.suffix.lower() == ".json":
        data = json.loads(candidate.read_text(encoding="utf-8"))
        best = data.get("best", {})
        return float(best["selection_score"]) if "selection_score" in best else None
    json_sibling = candidate.with_name("scheme01_selection.json")
    if json_sibling.exists():
        return read_scheme01_score(json_sibling)
    return None


def write_scheme02_report(path: Path, report: dict[str, Any]) -> None:
    """Write a concise Scheme02 selection report."""

    lines = [
        "# Scheme02 Selection",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Passed gate: `{report['passed_gate']}`",
        f"- Scheme01 score: `{report['scheme01_score']}`",
        f"- Best: `{report['best']['experiment_id']}`",
        "",
        "| rank | experiment | mode | macro-F1 | balanced acc | selection |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(report["results"][:20], start=1):
        lines.append(
            f"| {idx} | `{row['experiment_id']}` | `{row['mode']}` | "
            f"{row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} | {row['selection_score']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
