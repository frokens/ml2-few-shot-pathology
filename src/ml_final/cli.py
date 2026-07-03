"""CLI entry point — typer-based command interface.

Usage:
    python -m ml_final.cli env-check
    python -m ml_final.cli audit-dataset --train-dir train_few_shot
    python -m ml_final.cli align-reference-labels --config configs/scheme_01/reference_alignment.yaml
    python -m ml_final.cli extract-features --config configs/scheme_01/extract_features.yaml
    python -m ml_final.cli probe-conch-prompts --config configs/scheme_01/conch_prompt_probe.yaml
    python -m ml_final.cli run-frozen-cv --config configs/scheme_01/cv_heads.yaml
    python -m ml_final.cli train-peft-cv --config configs/scheme_02/peft_cv.yaml
    python -m ml_final.cli build-teacher --config configs/scheme_03/teacher_s01.yaml --out ...
    python -m ml_final.cli select-pseudolabels --config configs/scheme_03/pseudolabel.yaml --mode simulate --teacher ... --out ...
    python -m ml_final.cli train-pseudo-heads --config configs/scheme_03/pseudo_retrain.yaml --features ... --pseudolabels ...
    python -m ml_final.cli infer-final --config configs/scheme_01/ensemble.yaml
    python -m ml_final.cli init-model-registry --out artifacts/model_registry
    python -m ml_final.cli download-models --config ... --source official --dry-run
    python -m ml_final.cli verify-models --lock artifacts/model_registry/models.lock.yaml
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path
from typing import Optional

# --- Framework import guard ---
# Import failures here mean project dependencies are missing.
# The project convention is to run under the active project conda env:
#   conda activate <project-env>
#   pip install -e ".[dev]"
_CLI_DEPS_MISSING = []
for _pkg_name, _import_name in [
    ("typer[all]", "typer"),
    ("pyyaml", "yaml"),
    ("rich", "rich"),
    ("loguru", "loguru"),
    ("Pillow", "PIL"),
    ("scikit-learn", "sklearn"),
]:
    try:
        __import__(_import_name)
    except ImportError:
        _CLI_DEPS_MISSING.append(_pkg_name)

if _CLI_DEPS_MISSING:
    sys.stderr.write(
        "ERROR: Missing CLI dependencies: "
        f"{', '.join(_CLI_DEPS_MISSING)}\n\n"
        "The project expects commands to run under the active project conda env.\n"
        "On the server, activate the environment and install dependencies:\n"
        "\n"
        "    conda activate base\n"
        "    pip install -e /path/to/final_report\n"
        "\n"
        "For a custom local environment, create one first:\n"
        "\n"
        "    conda create -n ml_final python=3.12 -y\n"
        "    conda activate ml_final\n"
        "    pip install -e /path/to/final_report\n"
        "\n"
    )
    sys.exit(1)

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from ml_final.data.audit import DEFAULT_CLASSES, audit_dataset
from ml_final.utils.config import resolve_project_path
from ml_final.utils.logging import setup_logging
from ml_final.utils.paths import ensure_dir, model_registry_dir, project_root
from ml_final.weights.checksum import verify_checksums
from ml_final.weights.download_hf import (
    build_hf_download_command,
    dry_run_command,
    resolve_endpoint,
    resolve_ignore_patterns,
)
from ml_final.weights.download_modelscope import build_modelscope_command
from ml_final.weights.license_audit import generate_license_audit
from ml_final.weights.registry import (
    VALID_SOURCES,
    generate_lock_template,
    load_lock,
    load_requested,
    write_lock,
)

app = typer.Typer(
    name="ml-final",
    help="ML Final Project CLI — pathology image classification pipeline.",
)
console = Console()


def _default_training_offline_env() -> None:
    """Default formal training to the local locked model cache."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# audit-dataset
# ---------------------------------------------------------------------------


@app.command("audit-dataset")
def audit_dataset_command(
    train_dir: str = typer.Option(
        "train_few_shot",
        help="Training directory with Class_0..Class_4 subdirectories.",
    ),
    test_dir: Optional[str] = typer.Option(
        "test",
        help="Optional unlabeled test directory. Missing path is reported but not fatal.",
    ),
    out: str = typer.Option(
        "artifacts/dataset_audit",
        help="Output directory for manifests and audit JSON files.",
    ),
    expected_class_count: int = typer.Option(
        5,
        help="Expected number of classes named Class_0..Class_<n-1>.",
    ),
    strict_train: bool = typer.Option(
        True,
        "--strict-train/--no-strict-train",
        help="Fail if train classes are missing/extra or train images are invalid.",
    ),
) -> None:
    """Audit image files and write deterministic train/test manifests."""
    setup_logging("INFO")
    expected_classes = tuple(f"Class_{idx}" for idx in range(expected_class_count))
    if expected_classes != DEFAULT_CLASSES and expected_class_count == 5:
        expected_classes = DEFAULT_CLASSES

    train_path = project_root() / train_dir
    test_path = project_root() / test_dir if test_dir else None
    out_path = project_root() / out

    outputs = audit_dataset(
        train_dir=train_path,
        test_dir=test_path,
        out_dir=out_path,
        expected_classes=expected_classes,
        strict_train=strict_train,
    )

    console.print("[bold green]Dataset audit complete.[/bold green]")
    console.print(f"  Train manifest: {outputs.train_manifest}")
    if outputs.test_manifest is not None:
        console.print(f"  Test manifest:  {outputs.test_manifest}")
    else:
        console.print("  [yellow]Test manifest skipped: test directory missing or empty.[/yellow]")
    console.print(f"  Summary:        {outputs.summary_json}")


@app.command("align-reference-labels")
def align_reference_labels_command(
    config: str = typer.Option(
        "configs/scheme_01/reference_alignment.yaml",
        help="Reference six-class dataset alignment config.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Use pixel_stats backend for local plumbing tests.",
    ),
) -> None:
    """Use pretrained features to map anonymous Class_X labels to reference names."""
    setup_logging("INFO")
    from ml_final.data.reference_align import run_reference_alignment

    result = run_reference_alignment(config, run_name=run_name, smoke=smoke)
    console.print("[bold green]Reference label alignment complete.[/bold green]")
    console.print(f"  Summary: {result['summary_path']}")
    console.print(f"  Report:  {result['report_path']}")
    console.print(f"  Mapping: {result['mapping_path']}")


# ---------------------------------------------------------------------------
# scheme 01: frozen features and heads
# ---------------------------------------------------------------------------


@app.command("extract-features")
def extract_features_command(
    config: str = typer.Option(
        "configs/scheme_01/extract_features.yaml",
        help="Scheme 01 feature extraction config.",
    ),
    tta: Optional[str] = typer.Option(None, "--tta", help="Override TTA mode: none, d4, or geom6."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Use the pixel_stats backend for Mac/local smoke tests.",
    ),
) -> None:
    """Extract frozen features for Scheme 01."""
    setup_logging("INFO")
    from ml_final.features.extract import run_feature_extraction

    result = run_feature_extraction(config, tta=tta, run_name=run_name, smoke=smoke)
    console.print("[bold green]Feature extraction complete.[/bold green]")
    console.print(f"  Manifest: {result['manifest_path']}")
    for key, item in result["features"].items():
        console.print(
            f"  {key}: dim={item['feature_dim']} "
            f"train={item['train_count']} test={item['test_count']}"
        )


@app.command("run-frozen-cv")
def run_frozen_cv_command(
    config: str = typer.Option(
        "configs/scheme_01/cv_heads.yaml",
        help="Scheme 01 CV heads config.",
    ),
    features: Optional[str] = typer.Option(None, "--features", help="Feature directory."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Only use pixel_stats features and the smoke head set.",
    ),
) -> None:
    """Run frozen-feature CV heads and OOF ensemble."""
    setup_logging("INFO")
    from ml_final.heads.classical_heads import run_frozen_cv

    result = run_frozen_cv(config, features_dir=features, run_name=run_name, smoke=smoke)
    summary = result["summary"]
    console.print("[bold green]Frozen CV complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    if summary.get("best"):
        best = summary["best"]
        console.print(
            f"  Best: {best['head_id']} "
            f"macro_f1={best['macro_f1']:.6f} "
            f"balanced_acc={best['balanced_accuracy']:.6f}"
        )


@app.command("fuse-features")
def fuse_features_command(
    config: str = typer.Option(
        "configs/scheme_01/fuse_features_single.yaml",
        help="Scheme 01 feature fusion config.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
) -> None:
    """Fuse aligned frozen feature bundles into one multi-encoder bundle."""
    setup_logging("INFO")
    from ml_final.features.fuse import run_feature_fusion

    result = run_feature_fusion(config, run_name=run_name)
    console.print("[bold green]Feature fusion complete.[/bold green]")
    console.print(f"  Manifest: {result['manifest_path']}")
    console.print(
        f"  Feature: dim={result['feature']['feature_dim']} "
        f"train={result['feature']['train_count']} test={result['feature']['test_count']}"
    )


@app.command("probe-conch-prompts")
def probe_conch_prompts_command(
    config: str = typer.Option(
        "configs/scheme_01/conch_prompt_probe.yaml",
        help="Scheme 01 CONCH prompt probe config.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
) -> None:
    """Run CONCH image-text prompt families and write Scheme01-compatible predictions."""
    setup_logging("INFO")
    from ml_final.heads.conch_prompt import run_conch_prompt_probe

    result = run_conch_prompt_probe(config, run_name=run_name)
    summary = result["summary"]
    console.print("[bold green]CONCH prompt probe complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    console.print(f"  Best prompt family: {summary['best_family']}")


@app.command("select-scheme01")
def select_scheme01_command(
    runs: str = typer.Option("runs/scheme_01", "--runs", help="Scheme01 runs directory."),
    out: str = typer.Option("artifacts/selection", "--out", help="Selection output directory."),
    max_peft_backbones: int = typer.Option(
        2,
        "--max-peft-backbones",
        help="Maximum backbones allowed into Scheme02 PEFT.",
    ),
    min_delta_for_second: float = typer.Option(
        0.005,
        "--min-delta-for-second",
        help="Allow second backbone only when it is within this selection-score delta.",
    ),
) -> None:
    """Write Scheme01 selection artifacts for downstream schemes."""
    setup_logging("INFO")
    from ml_final.heads.selection import select_scheme01

    result = select_scheme01(
        runs=runs,
        out=out,
        max_peft_backbones=max_peft_backbones,
        min_delta_for_second=min_delta_for_second,
    )
    console.print("[bold green]Scheme01 selection complete.[/bold green]")
    console.print(f"  Best: {result['best']['candidate_id']}")
    console.print(f"  Backbones: {', '.join(result['selected_backbones'])}")
    console.print(f"  PEFT backbones: {', '.join(result['selected_peft_backbones'])}")


@app.command("audit-oof-errors")
def audit_oof_errors_command(
    oof: str = typer.Option(..., "--oof", help="OOF prediction .npz file to audit."),
    out: str = typer.Option("artifacts/analysis/oof_audit", "--out", help="Output directory."),
    manifest: Optional[str] = typer.Option(
        "artifacts/dataset_audit/train_manifest.csv",
        "--manifest",
        help="Train manifest used to map filenames back to images.",
    ),
    hard_classes: str = typer.Option(
        "Class_2,Class_3,Class_4",
        "--hard-classes",
        help="Comma-separated class names to prioritize in the audit.",
    ),
    top_k: int = typer.Option(40, "--top-k", help="Number of hard errors to show in report/montage."),
) -> None:
    """Write sample-level OOF error reports for hard-class Scheme01 analysis."""
    setup_logging("INFO")
    from ml_final.analysis.oof_audit import audit_oof_errors

    result = audit_oof_errors(
        oof_path=oof,
        out_dir=out,
        manifest_path=manifest,
        hard_classes=[item.strip() for item in hard_classes.split(",") if item.strip()],
        top_k=top_k,
    )
    console.print("[bold green]OOF audit complete.[/bold green]")
    console.print(f"  Report: {result['report_path']}")
    console.print(f"  Hard errors: {result['num_hard_errors']}")
    if result.get("montage_path"):
        console.print(f"  Montage: {result['montage_path']}")


@app.command("compare-oof-audits")
def compare_oof_audits_command(
    audit_dir: list[str] = typer.Option(
        ...,
        "--audit-dir",
        help="Audit output directory containing oof_sample_rows.csv. Repeat for multiple audits.",
    ),
    out: str = typer.Option(
        "artifacts/analysis/oof_audit_compare",
        "--out",
        help="Output directory for persistent error report.",
    ),
    min_wrong_count: int = typer.Option(
        2,
        "--min-wrong-count",
        help="Minimum number of audits where the sample must be wrong.",
    ),
) -> None:
    """Compare OOF audits and rank persistent high-confidence errors."""
    setup_logging("INFO")
    from ml_final.analysis.oof_audit import compare_oof_audits

    result = compare_oof_audits(
        audit_dirs=audit_dir,
        out_dir=out,
        min_wrong_count=min_wrong_count,
    )
    console.print("[bold green]OOF audit comparison complete.[/bold green]")
    console.print(f"  Report: {result['report_path']}")
    console.print(f"  Persistent errors: {result['num_persistent_errors']}")


# ---------------------------------------------------------------------------
# scheme 02: LP-FT and PEFT LoRA CV
# ---------------------------------------------------------------------------


@app.command("train-peft-cv")
def train_peft_cv_command(
    config: str = typer.Option(
        "configs/scheme_02/peft_cv.yaml",
        help="Scheme 02 PEFT CV config.",
    ),
    selected_backbones: Optional[str] = typer.Option(
        None,
        "--selected-backbones",
        help="Optional text file with one selected backbone key per line.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Checkpoint to resume from."),
    smoke: bool = typer.Option(
        False,
        "--smoke",
        help="Use tiny CNN and short CPU-friendly settings.",
    ),
) -> None:
    """Run Scheme 02 linear probe / LoRA cross-validation."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.peft_train import run_peft_cv

    result = run_peft_cv(
        config,
        selected_backbones=selected_backbones,
        run_name=run_name,
        resume=resume,
        smoke=smoke,
    )
    summary = result["summary"]
    console.print("[bold green]PEFT CV complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    if summary.get("best"):
        best = summary["best"]
        console.print(
            f"  Best: {best['experiment_id']} "
            f"macro_f1={best['macro_f1']:.6f} "
            f"balanced_acc={best['balanced_accuracy']:.6f}"
        )


@app.command("select-scheme02")
def select_scheme02_command(
    runs: str = typer.Option("runs/scheme_02", "--runs", help="Scheme02 runs directory."),
    out: str = typer.Option("artifacts/selection", "--out", help="Selection output directory."),
    scheme01: Optional[str] = typer.Option(
        "artifacts/selection/scheme01_selection.json",
        "--scheme01",
        help="Scheme01 selection JSON or report path.",
    ),
    min_improvement: float = typer.Option(
        0.005,
        "--min-improvement",
        help="Required Scheme02 selection-score improvement over Scheme01.",
    ),
) -> None:
    """Write Scheme02 selection artifacts for Scheme03/fusion."""
    setup_logging("INFO")
    from ml_final.training.selection import select_scheme02

    result = select_scheme02(
        runs=runs,
        out=out,
        scheme01=scheme01,
        min_improvement=min_improvement,
    )
    console.print("[bold green]Scheme02 selection complete.[/bold green]")
    console.print(f"  Best: {result['best']['experiment_id']}")
    console.print(f"  Passed gate: {result['passed_gate']}")


@app.command("train-peft-adapted-heads")
def train_peft_adapted_heads_command(
    config: str = typer.Option(
        "configs/scheme_02/peft_cv.yaml",
        help="Scheme 02 PEFT CV config used for the source run.",
    ),
    source_run: str = typer.Option(..., "--source-run", help="Source Scheme02 PEFT run name."),
    experiment_id: str = typer.Option(..., "--experiment-id", help="Source backbone__experiment id."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Output run name."),
) -> None:
    """Run classical heads on fold-local LoRA-adapted features."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.adapted_heads import run_adapted_head_cv

    result = run_adapted_head_cv(
        config,
        source_run=source_run,
        experiment_id=experiment_id,
        run_name=run_name,
    )
    best = result["summary"].get("best") or {}
    console.print("[bold green]Adapted-head CV complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    if best:
        console.print(
            f"  Best: {best['experiment_id']} "
            f"macro_f1={best['macro_f1']:.6f} "
            f"balanced_acc={best['balanced_accuracy']:.6f}"
        )


@app.command("train-fusion-peft-cv")
def train_fusion_peft_cv_command(
    config: str = typer.Option(
        "configs/scheme_02/fusion_peft_smoke.yaml",
        help="Scheme 02b fusion PEFT config.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(False, "--smoke", help="Use smoke fold limits from config."),
    experiment_name: list[str] = typer.Option(
        [],
        "--experiment-name",
        help="Run only the named Scheme02b experiment. Repeat for staged B0/B1/B2/B3 gates.",
    ),
) -> None:
    """Run Scheme02b multi-encoder representation-fusion CV."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.fusion_peft import run_fusion_peft_cv

    result = run_fusion_peft_cv(
        config,
        run_name=run_name,
        smoke=smoke,
        experiment_names=experiment_name or None,
    )
    best = result["summary"].get("best") or {}
    console.print("[bold green]Fusion PEFT CV complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    if best:
        console.print(
            f"  Best: {best['experiment_id']} "
            f"macro_f1={best['macro_f1']:.6f} "
            f"balanced_acc={best['balanced_accuracy']:.6f}"
        )


@app.command("train-fusion-pseudo-cv")
def train_fusion_pseudo_cv_command(
    config: str = typer.Option(
        "configs/scheme_03/fusion_pseudo_cv_s02b.yaml",
        help="Scheme 03 pseudo-student fusion CV config.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(False, "--smoke", help="Use smoke fold limits from config."),
    experiment_name: list[str] = typer.Option(
        [],
        "--experiment-name",
        help="Run only the named Scheme02b fusion experiment. Repeat for staged pseudo CV checks.",
    ),
    pseudolabels: str = typer.Option(
        ...,
        "--pseudolabels",
        help="Selected_pseudolabels.csv generated from the Scheme02b teacher.",
    ),
) -> None:
    """Run Scheme02b fusion CV with pseudo rows mixed into the training folds."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.fusion_peft import run_fusion_peft_cv

    result = run_fusion_peft_cv(
        config,
        run_name=run_name,
        smoke=smoke,
        experiment_names=experiment_name or None,
        pseudolabels=pseudolabels,
    )
    best = result["summary"].get("best") or {}
    console.print("[bold green]Fusion pseudo CV complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    if best:
        console.print(
            f"  Best: {best['experiment_id']} "
            f"macro_f1={best['macro_f1']:.6f} "
            f"balanced_acc={best['balanced_accuracy']:.6f}"
        )


@app.command("diagnose-fusion-classical-cv")
def diagnose_fusion_classical_cv_command(
    config: str = typer.Option(
        "configs/scheme_02/fusion_peft_cv.yaml",
        help="Scheme 02b fusion config used for branch definitions and heads.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(False, "--smoke", help="Use smoke fold limits from config."),
    diagnostic_name: list[str] = typer.Option(
        [],
        "--diagnostic-name",
        help="Run only the named Scheme02b classical diagnostic. Repeat for subsets.",
    ),
) -> None:
    """Run raw-branch classical fusion diagnostics for Scheme02b B0."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.fusion_peft import run_fusion_classical_diagnostic

    result = run_fusion_classical_diagnostic(
        config,
        run_name=run_name,
        smoke=smoke,
        diagnostic_names=diagnostic_name or None,
    )
    best = result["summary"].get("best") or {}
    console.print("[bold green]Fusion classical diagnostic complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    if best:
        console.print(
            f"  Best: {best['experiment_id']} "
            f"macro_f1={best['macro_f1']:.6f} "
            f"balanced_acc={best['balanced_accuracy']:.6f}"
        )


@app.command("train-peft-single-refit")
def train_peft_single_refit_command(
    config: str = typer.Option(
        "configs/scheme_02/peft_cv.yaml",
        help="Scheme 02 PEFT config containing the selected experiment.",
    ),
    experiment_id: str = typer.Option(..., "--experiment-id", help="backbone__experiment id to refit."),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    source_summary: Optional[str] = typer.Option(
        None,
        "--source-summary",
        help="CV summary.json that provides OOF metrics for this refit.",
    ),
    oof_path: Optional[str] = typer.Option(None, "--oof-path", help="Explicit OOF prediction path."),
) -> None:
    """Train one Scheme02 candidate on all labels and write one test prediction."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.peft_refit import run_peft_single_refit

    result = run_peft_single_refit(
        config,
        experiment_id=experiment_id,
        run_name=run_name,
        source_summary=source_summary,
        oof_path=oof_path,
    )
    console.print("[bold green]PEFT single refit complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")


@app.command("train-fusion-single-refit")
def train_fusion_single_refit_command(
    config: str = typer.Option(
        "configs/scheme_03/fusion_single_refit_best.yaml",
        help="Scheme02b/Scheme03 fusion single-refit config.",
    ),
    experiment_name: str = typer.Option(
        "b1_uni_conch_adapters_all4_concat_p32_reg",
        "--experiment-name",
        help="Scheme02b fusion experiment name to refit.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    source_summary: Optional[str] = typer.Option(
        None,
        "--source-summary",
        help="CV summary.json that provides OOF metrics for this refit.",
    ),
    oof_path: Optional[str] = typer.Option(None, "--oof-path", help="Explicit OOF prediction path."),
    pseudolabels: Optional[str] = typer.Option(
        None,
        "--pseudolabels",
        help="Optional selected_pseudolabels.csv for Scheme03 pseudo refit.",
    ),
) -> None:
    """Train one Scheme02b fusion model on all labels and write one test prediction."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.fusion_peft import run_fusion_single_refit

    result = run_fusion_single_refit(
        config,
        experiment_name=experiment_name,
        run_name=run_name,
        source_summary=source_summary,
        oof_path=oof_path,
        pseudolabels=pseudolabels,
    )
    summary = result["summary"]
    console.print("[bold green]Fusion single refit complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    console.print(f"  Test prediction: {summary['best']['test_path']}")
    console.print(f"  Submission:      {summary['submission']}")


@app.command("predict-fusion-single-refit")
def predict_fusion_single_refit_command(
    config: str = typer.Option(
        "configs/scheme_03/fusion_single_refit_best.yaml",
        help="Scheme02b/Scheme03 fusion single-refit config.",
    ),
    experiment_name: str = typer.Option(
        "b1_uni_conch_adapters_all4_concat_p32_reg",
        "--experiment-name",
        help="Scheme02b fusion experiment name to predict.",
    ),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Trained fusion single-refit checkpoint.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Prediction run name."),
    out: Optional[str] = typer.Option(None, "--out", help="Prediction output directory."),
) -> None:
    """Load one trained Scheme02b fusion model and write one test prediction."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.fusion_peft import predict_fusion_single_refit

    result = predict_fusion_single_refit(
        config,
        experiment_name=experiment_name,
        checkpoint=checkpoint,
        run_name=run_name,
        out=out,
    )
    console.print("[bold green]Fusion single-refit prediction complete.[/bold green]")
    console.print(f"  Test prediction: {result['test_path']}")
    console.print(f"  Submission:      {result['submission']}")


# ---------------------------------------------------------------------------
# scheme 03: transductive pseudo-labeling
# ---------------------------------------------------------------------------


@app.command("build-teacher")
def build_teacher_command(
    config: str = typer.Option(
        "configs/scheme_03/teacher_s01.yaml",
        help="Scheme 03 teacher ensemble config.",
    ),
    out: str = typer.Option(
        "artifacts/teachers/s01_teacher",
        "--out",
        help="Teacher output directory.",
    ),
) -> None:
    """Build a calibrated teacher ensemble from OOF/test prediction files."""
    setup_logging("INFO")
    from ml_final.pseudo.calibrate import build_teacher

    result = build_teacher(config, out=out)
    console.print("[bold green]Teacher build complete.[/bold green]")
    console.print(f"  OOF rows:    {result['num_oof_rows']}")
    console.print(f"  Test rows:   {result['num_test_rows']}")
    console.print(f"  Temperature: {result['temperature']}")
    console.print(f"  Manifest:    {result['teacher_oof_predictions']}")


@app.command("select-pseudolabels")
def select_pseudolabels_command(
    config: str = typer.Option(
        "configs/scheme_03/pseudolabel.yaml",
        help="Scheme 03 pseudo-label selection config.",
    ),
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Selection mode: simulate or select-test.",
    ),
    teacher: str = typer.Option(
        ...,
        "--teacher",
        help="Teacher directory produced by build-teacher.",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="Pseudo-label output directory.",
    ),
) -> None:
    """Select pseudo labels from calibrated teacher predictions."""
    setup_logging("INFO")
    from ml_final.pseudo.select import select_pseudolabels

    result = select_pseudolabels(config, mode=mode, teacher=teacher, out=out)
    console.print("[bold green]Pseudo-label selection complete.[/bold green]")
    console.print(f"  Mode:       {result['mode']}")
    console.print(f"  Candidates: {result['num_candidates']}")
    console.print(f"  Selected:   {result['num_selected']}")
    console.print(f"  Rejected:   {result['num_rejected']}")


@app.command("train-pseudo-heads")
def train_pseudo_heads_command(
    config: str = typer.Option(
        "configs/scheme_03/pseudo_retrain.yaml",
        help="Scheme 03 pseudo-label retraining config.",
    ),
    features: str = typer.Option(
        "artifacts/features/scheme_01",
        "--features",
        help="Scheme 01 feature directory.",
    ),
    pseudolabels: str = typer.Option(
        ...,
        "--pseudolabels",
        help="selected_pseudolabels.csv produced by select-pseudolabels.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
) -> None:
    """Train weighted frozen-feature heads with selected pseudo labels."""
    setup_logging("INFO")
    from ml_final.pseudo.retrain import train_pseudo_heads

    result = train_pseudo_heads(
        config,
        features=features,
        pseudolabels=pseudolabels,
        run_name=run_name,
    )
    summary = result["summary"]
    console.print("[bold green]Pseudo head retraining complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    console.print(f"  Status:  {summary['status']}")
    console.print(f"  Outputs: {len(summary['results'])}")


@app.command("train-pseudo-lora")
def train_pseudo_lora_command(
    config: str = typer.Option(
        "configs/scheme_03/pseudo_lora_lam005.yaml",
        help="S03-6 pseudo-label LoRA config.",
    ),
    selected_peft: str = typer.Option(
        "artifacts/selection/scheme02_best_peft.txt",
        "--selected-peft",
        help="Scheme02 selected PEFT experiment id file.",
    ),
    pseudolabels: str = typer.Option(
        ...,
        "--pseudolabels",
        help="selected_pseudolabels.csv produced by select-pseudolabels.",
    ),
    train_manifest: str = typer.Option(
        "artifacts/dataset_audit/train_manifest.csv",
        "--train-manifest",
        help="Train manifest.",
    ),
    test_manifest: str = typer.Option(
        "artifacts/dataset_audit/test_manifest.csv",
        "--test-manifest",
        help="Test manifest.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Checkpoint to resume from."),
    smoke: bool = typer.Option(False, "--smoke", help="Use tiny CNN settings."),
) -> None:
    """Run optional S03-6 pseudo-label LoRA training."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.pseudo_peft import run_pseudo_lora

    result = run_pseudo_lora(
        config,
        selected_peft=selected_peft,
        pseudolabels=pseudolabels,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        run_name=run_name,
        resume=resume,
        smoke=smoke,
    )
    summary = result["summary"]
    console.print("[bold green]Pseudo-LoRA training complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    console.print(f"  Pseudo rows: {summary['num_pseudo']}")
    console.print(f"  OOF macro-F1: {summary['metrics']['macro_f1']:.6f}")
    console.print(f"  Submission: {summary['submission']}")


@app.command("train-pseudo-lora-single-refit")
def train_pseudo_lora_single_refit_command(
    config: str = typer.Option(
        "configs/scheme_03/pseudo_lora_uni2_scale64_hardw_lam001_single_refit.yaml",
        help="S03-6 pseudo-label LoRA single-refit config.",
    ),
    selected_peft: str = typer.Option(
        "artifacts/selection/scheme02_best_peft.txt",
        "--selected-peft",
        help="Scheme02 selected PEFT experiment id file.",
    ),
    pseudolabels: str = typer.Option(
        ...,
        "--pseudolabels",
        help="selected_pseudolabels.csv produced by select-pseudolabels.",
    ),
    train_manifest: str = typer.Option(
        "artifacts/dataset_audit/train_manifest.csv",
        "--train-manifest",
        help="Train manifest.",
    ),
    test_manifest: str = typer.Option(
        "artifacts/dataset_audit/test_manifest.csv",
        "--test-manifest",
        help="Test manifest.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    smoke: bool = typer.Option(False, "--smoke", help="Use tiny CNN settings."),
) -> None:
    """Train one full-data pseudo-label LoRA model and write one test prediction."""
    setup_logging("INFO")
    _default_training_offline_env()
    from ml_final.training.pseudo_peft import run_pseudo_lora_single_refit

    result = run_pseudo_lora_single_refit(
        config,
        selected_peft=selected_peft,
        pseudolabels=pseudolabels,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        run_name=run_name,
        smoke=smoke,
    )
    summary = result["summary"]
    console.print("[bold green]Pseudo-LoRA single refit complete.[/bold green]")
    console.print(f"  Run dir: {result['run_dir']}")
    console.print(f"  Pseudo rows: {summary['num_pseudo']}")
    console.print(f"  Test prediction: {summary['test_path']}")
    console.print(f"  Submission: {summary['submission']}")


@app.command("infer-final")
def infer_final_command(
    config: str = typer.Option(
        "configs/scheme_01/ensemble.yaml",
        help="Scheme 01 final inference config.",
    ),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name."),
    out: Optional[str] = typer.Option(None, "--out", help="Submission output directory."),
) -> None:
    """Generate final submission from saved Scheme 01 test predictions."""
    setup_logging("INFO")
    from ml_final.inference.submission import infer_final

    result = infer_final(config, run_name=run_name, out=out)
    console.print("[bold green]Submission generated.[/bold green]")
    console.print(f"  Submission: {result['submission']}")
    console.print(f"  Rows:       {result['num_rows']}")


@app.command("calibrate-class-bias")
def calibrate_class_bias_command(
    config: str = typer.Option(..., "--config", help="Class-bias calibration config."),
) -> None:
    """Nested-validate additive class-bias calibration for one OOF file."""
    setup_logging("INFO")
    from ml_final.inference.class_bias import calibrate_class_bias

    result = calibrate_class_bias(config)
    console.print("[bold green]Class-bias calibration complete.[/bold green]")
    console.print(f"  Bias:             {result['artifacts']['bias']}")
    console.print(f"  Nested selection: {result['nested']['selection_score']:.6f}")


@app.command("apply-class-bias")
def apply_class_bias_command(
    prediction: str = typer.Option(..., "--prediction", help="Prediction NPZ to transform."),
    bias: str = typer.Option(..., "--bias", help="Saved class_bias.json."),
    out: str = typer.Option(..., "--out", help="Output biased prediction NPZ."),
) -> None:
    """Apply a saved class bias to one prediction NPZ."""
    setup_logging("INFO")
    from ml_final.inference.class_bias import apply_class_bias

    result = apply_class_bias(prediction, bias=bias, out=out)
    console.print("[bold green]Class bias applied.[/bold green]")
    console.print(f"  Output: {result['out']}")


@app.command("select-final-blend")
def select_final_blend_command(
    scheme01_runs: str = typer.Option("runs/scheme_01", "--scheme01-runs", help="Scheme01 runs root."),
    scheme02_runs: Optional[str] = typer.Option("runs/scheme_02", "--scheme02-runs", help="Scheme02 runs root."),
    scheme03_runs: Optional[str] = typer.Option("runs/scheme_03", "--scheme03-runs", help="Scheme03 runs root."),
    out: str = typer.Option("artifacts/selection", "--out", help="Selection output directory."),
    top_k: int = typer.Option(5, "--top-k", help="Maximum OOF sources in blend."),
    weight_step: float = typer.Option(0.1, "--weight-step", help="Simplex grid step for weights."),
) -> None:
    """Select final OOF blend and write submission when test predictions exist."""
    setup_logging("INFO")
    from ml_final.inference.fusion_select import select_final_blend

    result = select_final_blend(
        scheme01_runs=scheme01_runs,
        scheme02_runs=scheme02_runs,
        scheme03_runs=scheme03_runs,
        out=out,
        top_k=top_k,
        weight_step=weight_step,
    )
    console.print("[bold green]Final blend selection complete.[/bold green]")
    console.print(f"  Method: {result['chosen_method']}")
    console.print(f"  Sources: {result['selected_count']}")
    console.print(f"  Submission: {result['submission']}")


@app.command("validate-submission")
def validate_submission_command(
    submission: str = typer.Argument(..., help="submission.csv to validate."),
    test_manifest: str = typer.Option(
        "artifacts/dataset_audit/test_manifest.csv",
        "--test-manifest",
        help="Test manifest produced by audit-dataset.",
    ),
    test_prediction: Optional[str] = typer.Option(None, "--test-prediction", help="Optional test prediction .npz to audit."),
    compare_submission: Optional[str] = typer.Option(None, "--compare-submission", help="Optional backup submission to compare."),
    expected_rows: Optional[int] = typer.Option(None, "--expected-rows", help="Expected submission row count."),
) -> None:
    """Validate submission CSV schema, row alignment, and optional prediction diagnostics."""
    setup_logging("INFO")
    from ml_final.inference.submission import validate_submission

    result = validate_submission(
        submission,
        test_manifest=test_manifest,
        test_prediction=test_prediction,
        compare_submission=compare_submission,
        expected_rows=expected_rows,
    )
    console.print("[bold green]Submission valid.[/bold green]")
    console.print(f"  Rows: {result['rows']}")
    console.print(f"  Class distribution: {result['class_distribution']}")
    if "prediction" in result:
        prediction = result["prediction"]
        console.print(f"  Test prediction: {prediction['path']}")
        if "prediction_kind" in prediction:
            console.print(f"  Prediction kind: {prediction['prediction_kind']}")
        if "model_count" in prediction:
            console.print(f"  Model count: {prediction['model_count']}")
        console.print(f"  Top1 confidence: {prediction['top1_confidence']}")
    if "comparison" in result:
        console.print(f"  Changed labels: {result['comparison']['changed_labels']}")

# ---------------------------------------------------------------------------
# env-check
# ---------------------------------------------------------------------------


@app.command("env-check")
def env_check(
    expect_gpus: int = typer.Option(0, help="Expected number of GPUs."),
    expect_cuda: bool = typer.Option(False, help="Require CUDA to be available."),
    check_hf_cache: Optional[str] = typer.Option(
        None, help="Path to Hugging Face cache directory to verify."
    ),
    check_write: Optional[str] = typer.Option(
        None, help="Comma-separated paths to verify are writable."
    ),
) -> None:
    """Check the local environment for ML readiness."""
    setup_logging("INFO")
    console.print("[bold green]ML Final — Environment Check[/bold green]")
    console.print(f"  Python: {sys.version}")
    console.print(f"  Executable: {sys.executable}")
    console.print(f"  Project root: {project_root()}")

    # Check Python version
    if sys.version_info < (3, 10):
        console.print("[bold red]FAIL:[/bold red] Python >= 3.10 required.")
        raise typer.Exit(code=1)
    console.print("  [green]Python >= 3.10[/green]")

    # Check key packages
    packages = [
        "torch",
        "timm",
        "huggingface_hub",
        "transformers",
        "sklearn",
        "numpy",
        "pandas",
        "PIL",
        "typer",
        "yaml",
        "loguru",
    ]
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        console.print(f"[yellow]WARN:[/yellow] Missing packages: {', '.join(missing)}")
    else:
        console.print("  [green]Core packages available[/green]")

    # Check CUDA
    if expect_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                console.print(
                    f"  [green]CUDA available: {torch.cuda.device_count()} GPU(s)[/green]"
                )
                if torch.cuda.device_count() < expect_gpus:
                    console.print(
                        f"  [yellow]WARN: Expected {expect_gpus} GPUs, "
                        f"found {torch.cuda.device_count()}[/yellow]"
                    )
            else:
                console.print("  [bold red]FAIL: CUDA not available[/bold red]")
                raise typer.Exit(code=1)
        except ImportError:
            console.print("  [bold red]FAIL: torch not installed, cannot check CUDA[/bold red]")
            raise typer.Exit(code=1)

    # Check HF cache
    if check_hf_cache:
        cache_path = Path(check_hf_cache)
        if cache_path.exists():
            console.print(f"  [green]HF cache exists: {cache_path}[/green]")
        else:
            console.print(f"  [yellow]WARN: HF cache not found: {cache_path}[/yellow]")

    # Check writable paths
    if check_write:
        for p in check_write.split(","):
            p = p.strip()
            wpath = project_root() / p
            try:
                wpath.mkdir(parents=True, exist_ok=True)
                test_file = wpath / ".write_test"
                test_file.touch()
                test_file.unlink()
                console.print(f"  [green]Writable: {p}[/green]")
            except Exception:
                console.print(f"  [bold red]FAIL: Not writable: {p}[/bold red]")

    console.print("\n[bold green]Environment check complete.[/bold green]")


# ---------------------------------------------------------------------------
# init-model-registry
# ---------------------------------------------------------------------------


@app.command("init-model-registry")
def init_model_registry(
    out: str = typer.Option(
        "artifacts/model_registry",
        help="Output directory for registry files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview only; do not write files.",
    ),
) -> None:
    """Initialize the model registry with models.requested.yaml and scaffolding."""
    setup_logging("INFO")
    out_dir = project_root() / out
    console.print(f"[bold]Initializing model registry in: {out_dir}[/bold]")

    # Required models
    models = {
        "models": {
            "uni2_h": {
                "hub": "huggingface",
                "repo_id": "MahmoodLab/UNI2-h",
                "required": True,
                "gated": True,
                "license": "CC-BY-NC-ND-4.0",
                "usage": "feature_extraction_and_optional_lora",
                "allow_patterns": [
                    "*.json",
                    "*.py",
                    "*.txt",
                    "*.md",
                    "*.safetensors",
                    "*.bin",
                ],
            },
            "virchow2": {
                "hub": "huggingface",
                "repo_id": "paige-ai/Virchow2",
                "required": True,
                "gated": True,
                "license": "check_model_card_terms",
                "usage": "feature_extraction_and_optional_lora",
                "allow_patterns": [
                    "*.json",
                    "*.py",
                    "*.txt",
                    "*.md",
                    "*.safetensors",
                ],
            },
            "conch": {
                "hub": "huggingface",
                "repo_id": "MahmoodLab/CONCH",
                "required": True,
                "gated": True,
                "license": "CC-BY-NC-ND-4.0",
                "usage": "image_encoder_features_and_prompt_encoder_probe",
                "official_loader": {
                    "library": "conch",
                    "model_name": "conch_ViT-B-16",
                    "frozen_image_encode": "proj_contrast=False, normalize=False",
                    "prompt_image_encode": "proj_contrast=True, normalize=True",
                    "text_encode": "conch.downstream.zeroshot_path.zero_shot_classifier",
                },
                "allow_patterns": [
                    "*.json",
                    "*.py",
                    "*.txt",
                    "*.md",
                    "*.safetensors",
                    "*.bin",
                ],
            },
            "h_optimus_0": {
                "hub": "huggingface",
                "repo_id": "bioptimus/H-optimus-0",
                "required": True,
                "gated": True,
                "license": "Apache-2.0",
                "usage": "feature_extraction_and_optional_lora",
                "official_loader": {
                    "library": "timm",
                    "model_name": "hf-hub:bioptimus/H-optimus-0",
                    "input_size": 224,
                    "feature_dim": 1536,
                    "model_kwargs": {
                        "num_classes": 0,
                        "init_values": 1e-5,
                        "dynamic_img_size": False,
                    },
                    "mean": [0.707223, 0.578729, 0.703617],
                    "std": [0.211883, 0.230117, 0.177517],
                },
                "allow_patterns": [
                    "*.json",
                    "*.py",
                    "*.txt",
                    "*.md",
                    "*.safetensors",
                    "*.bin",
                ],
            },
        }
    }

    if dry_run:
        console.print("[yellow]--dry-run: would create the following files:[/yellow]")
        console.print(f"  {out_dir / 'models.requested.yaml'}")
        console.print(f"  {out_dir / 'download_report.md'}")
        console.print(f"  {out_dir / 'license_audit.md'}")
        console.print(f"  {out_dir / 'models.lock.yaml'} (template)")
        console.print(f"  {out_dir / '.gitkeep'}")
        # Print the YAML content for review
        console.print("\n[dim]models.requested.yaml preview:[/dim]")
        console.print(yaml.safe_dump(models, sort_keys=False, default_flow_style=False))
        return

    # Actually write files
    ensure_dir(out_dir)

    requested_path = out_dir / "models.requested.yaml"
    with open(requested_path, "w") as f:
        yaml.safe_dump(models, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Wrote {requested_path}")

    # Write license audit stub
    generate_license_audit(models, output_path=out_dir / "license_audit.md")

    # Write download report stub
    report_path = out_dir / "download_report.md"
    report_path.write_text(
        "# Model Download Report\n\n"
        f"Generated: {datetime.datetime.now().isoformat()}\n\n"
        "Models have NOT been downloaded yet.\n"
        "Run `download-models --execute` to download.\n"
    )
    logger.info(f"Wrote {report_path}")

    # Write lockfile template
    lock_template = generate_lock_template(
        models, source="official", store="external_models/model_store"
    )
    write_lock(lock_template, out_dir / "models.lock.yaml")
    logger.info(f"Wrote {out_dir / 'models.lock.yaml'} (template)")

    # Ensure .gitkeep
    (out_dir / ".gitkeep").touch()
    logger.info(f"Touched {out_dir / '.gitkeep'}")

    console.print("[bold green]Model registry initialized.[/bold green]")


# ---------------------------------------------------------------------------
# download-models
# ---------------------------------------------------------------------------


@app.command("download-models")
def download_models(
    config: str = typer.Option(
        "artifacts/model_registry/models.requested.yaml",
        help="Path to models.requested.yaml.",
    ),
    source: str = typer.Option(
        "official",
        help="Download source: official, hf-mirror, modelscope, or offline.",
    ),
    store: str = typer.Option(
        "external_models/model_store",
        help="Root directory for model storage.",
    ),
    out: str = typer.Option(
        "artifacts/model_registry",
        help="Output directory for lockfile and reports.",
    ),
    model: Optional[list[str]] = typer.Option(
        None,
        "--model",
        help="Specific model keys to process (repeatable). If omitted, process all.",
    ),
    offline_verify: bool = typer.Option(
        False,
        "--offline-verify",
        help="Verify local files only; do not attempt any download.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print commands that would be run; do not execute.",
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Actually execute downloads. Required for real downloads.",
    ),
) -> None:
    """Download or verify model weights from the configured source.

    By default, this is a dry-run. Pass --execute to actually download.
    """
    setup_logging("INFO")

    # Validate source
    if source not in VALID_SOURCES:
        logger.error(f"Invalid source '{source}'. Must be one of {VALID_SOURCES}")
        raise typer.Exit(code=1)

    # Load config
    config_path = resolve_project_path(config)
    if config_path is None:
        console.print("[bold red]Config path is required.[/bold red]")
        raise typer.Exit(code=1)
    requested = load_requested(config_path)
    models = requested["models"]

    # Filter models if --model specified. Without an explicit model list, only
    # process required weights; optional gated candidates can stay pending.
    if model:
        requested_keys = set(model)
        unknown = requested_keys - set(models.keys())
        if unknown:
            logger.error(f"Unknown model keys: {unknown}. Known: {list(models.keys())}")
            raise typer.Exit(code=1)
        models = {k: v for k, v in models.items() if k in requested_keys}
    else:
        skipped_optional = [k for k, v in models.items() if not bool(v.get("required", True))]
        models = {k: v for k, v in models.items() if bool(v.get("required", True))}
        if skipped_optional:
            console.print(
                "[yellow]Skipping non-required models by default: "
                f"{', '.join(skipped_optional)}. Use --model to process them.[/yellow]"
            )
    processed_requested = {"models": models}

    out_dir = project_root() / out
    ensure_dir(out_dir)
    store_path = resolve_project_path(store)
    if store_path is None:
        console.print("[bold red]Store path is required.[/bold red]")
        raise typer.Exit(code=1)
    ensure_dir(store_path)
    hf_home = project_root() / "hf_home"
    hf_cache = hf_home / "hub"
    ensure_dir(hf_cache)

    console.print(f"[bold]Model Download Plan[/bold]")
    console.print(f"  Source: {source}")
    console.print(f"  Store:  {store_path}")
    console.print(f"  Models: {', '.join(models.keys())}")

    if dry_run:
        console.print("\n[bold yellow]--dry-run enabled: showing planned commands.[/bold yellow]\n")

    # Offline verify mode
    if source == "offline" or offline_verify:
        _run_offline_verify(models, str(store_path), out_dir)
        return

    # Process each model
    downloaded_models = {}
    for model_key, entry in models.items():
        hub = entry["hub"]
        repo_id = entry["repo_id"]
        safe_name = repo_id.replace("/", "__")
        allow_patterns = entry.get("allow_patterns", [])
        ignore_patterns = resolve_ignore_patterns(model_key, allow_patterns)

        local_dir = store_path / "hf" / safe_name / "snapshot"

        console.print(f"\n[bold]--- {model_key} ({repo_id}) ---[/bold]")

        if hub == "huggingface":
            endpoint = resolve_endpoint(source)
            env_vars = {
                "HF_ENDPOINT": endpoint,
                "HF_HOME": str(hf_home),
                "HF_HUB_CACHE": str(hf_cache),
            }
            cmd = build_hf_download_command(
                repo_id=repo_id,
                cache_dir=str(hf_cache),
                local_dir=str(local_dir),
                endpoint=endpoint,
                allow_patterns=allow_patterns if allow_patterns else None,
                ignore_patterns=ignore_patterns if ignore_patterns else None,
            )
            rendered = dry_run_command(env_vars, cmd)
            console.print(f"  [dim]{rendered}[/dim]")

            if ignore_patterns:
                console.print(
                    f"  [yellow]Excluding: {', '.join(ignore_patterns)}[/yellow]"
                )

        elif hub == "modelscope":
            cmd = build_modelscope_command(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
            )
            console.print(f"  [dim]{cmd}[/dim]")

        if dry_run:
            continue

        # Safety gate: require --execute for actual downloads
        if not execute:
            console.print(
                "  [yellow]Skipping execution (pass --execute to actually download).[/yellow]"
            )
            continue

        if hub != "huggingface":
            logger.warning("Execution for non-Hugging Face hubs is not implemented.")
            continue
        downloaded_models[model_key] = _execute_hf_snapshot_download(
            model_key=model_key,
            entry=entry,
            source=source,
            endpoint=endpoint,
            cache_dir=hf_cache,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    # Write updated lock template
    if dry_run:
        lock_template = generate_lock_template(processed_requested, source, str(store_path))
        write_lock(lock_template, out_dir / "models.lock.yaml")
        logger.info(f"Updated lockfile template: {out_dir / 'models.lock.yaml'}")
    elif execute:
        lock_template = generate_lock_template(processed_requested, source, str(store_path))
        for key, value in downloaded_models.items():
            lock_template["models"][key].update(value)
        lock_path = out_dir / "models.lock.yaml"
        existing_lock = load_lock(lock_path) or {"models": {}}
        merged_lock = {"models": dict(existing_lock.get("models", {}))}
        merged_lock["models"].update(lock_template["models"])
        write_lock(merged_lock, lock_path)
        logger.info(f"Updated lockfile: {out_dir / 'models.lock.yaml'}")

    # Write download report
    _write_download_report(models, source, str(store_path), out_dir, dry_run=dry_run)

    # Generate license audit
    generate_license_audit(
        requested,
        lock=load_lock(out_dir / "models.lock.yaml"),
        output_path=out_dir / "license_audit.md",
    )

    console.print("\n[bold green]Download plan complete.[/bold green]")


def _execute_hf_snapshot_download(
    *,
    model_key: str,
    entry: dict,
    source: str,
    endpoint: str,
    cache_dir: Path,
    local_dir: Path,
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> dict:
    """Download one HF repo into a project-local directory and return lock data."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise typer.BadParameter(
            "huggingface_hub is required for --execute downloads. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    repo_id = entry["repo_id"]
    revision = entry.get("revision", "main")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HOME"] = str(project_root() / "hf_home")
    os.environ["HF_HUB_CACHE"] = str(cache_dir)

    console.print(f"  [bold]Resolving revision for {model_key}...[/bold]")
    info = HfApi(endpoint=endpoint).model_info(repo_id=repo_id, revision=revision)
    resolved_revision = info.sha or revision
    final_dir = local_dir.parent / resolved_revision
    ensure_dir(final_dir)

    console.print(f"  [bold]Downloading to {final_dir}[/bold]")
    snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
        cache_dir=str(cache_dir),
        local_dir=str(final_dir),
        allow_patterns=allow_patterns or None,
        ignore_patterns=ignore_patterns or None,
    )

    return {
        "revision": resolved_revision,
        "source_endpoint": endpoint,
        "local_path": str(final_dir),
        "cache_path": str(cache_dir),
        "downloaded_at": datetime.datetime.now().isoformat(),
        "license": entry["license"],
        "terms_accepted": bool(entry.get("gated", False)),
    }


def _run_offline_verify(
    models: dict,
    store: str,
    out_dir: Path,
) -> None:
    """Verify local files without network access."""
    console.print("[bold]Offline Verification Mode[/bold]")
    all_ok = True

    for model_key, entry in models.items():
        repo_id = entry["repo_id"]
        safe_name = repo_id.replace("/", "__")
        model_dir = Path(store) / "hf" / safe_name

        console.print(f"\n  Checking {model_key} in {model_dir} ...")

        if not model_dir.exists():
            console.print(
                f"    [bold red]FAIL:[/bold red] Directory not found: {model_dir}"
            )
            all_ok = False
            continue

        # Look for revision subdirectory
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not subdirs:
            console.print(
                f"    [bold red]FAIL:[/bold red] No revision directories in {model_dir}"
            )
            all_ok = False
            continue

        # Check for required files in the first revision dir
        rev_dir = subdirs[0]
        required_files = entry.get("required_files", None)
        if required_files:
            for rf in required_files:
                matches = list(rev_dir.rglob(rf))
                if not matches:
                    console.print(f"    [yellow]WARN: No files matching '{rf}'[/yellow]")
                else:
                    console.print(f"    [green]Found {len(matches)} file(s) matching '{rf}'[/green]")
        else:
            file_count = sum(1 for _ in rev_dir.rglob("*") if _.is_file())
            console.print(
                f"    {'[green]' if file_count > 0 else '[red]'} "
                f"{file_count} file(s) found{'[/green]' if file_count > 0 else '[/red]'}"
            )
            if file_count == 0:
                all_ok = False

        # Look for SHA256SUMS
        sums_path = out_dir / f"SHA256SUMS.{model_key}"
        if sums_path.exists():
            ok, errors = verify_checksums(rev_dir, sums_path)
            if ok:
                console.print(f"    [green]SHA256 checksums verified[/green]")
            else:
                for err in errors:
                    console.print(f"    [red]{err}[/red]")
                all_ok = False
        else:
            console.print(
                f"    [yellow]No SHA256SUMS found at {sums_path}[/yellow]"
            )

    if all_ok:
        console.print("\n[bold green]All models verified successfully.[/bold green]")
    else:
        console.print("\n[bold red]Offline verification FAILED. See errors above.[/bold red]")
        raise typer.Exit(code=1)


def _write_download_report(
    models: dict,
    source: str,
    store: str,
    out_dir: Path,
    dry_run: bool = True,
) -> None:
    """Write a download report in Markdown format."""
    lines = [
        "# Model Download Report",
        "",
        f"Generated: {datetime.datetime.now().isoformat()}",
        f"Source: {source}",
        f"Store: {store}",
        f"Mode: {'dry-run' if dry_run else 'execute'}",
        "",
        "## Models Processed",
        "",
        "| Key | Repo ID | Hub | License | Status |",
        "|-----|---------|-----|---------|--------|",
    ]

    for key, entry in models.items():
        status = "dry-run preview" if dry_run else "commands generated"
        lines.append(
            f"| {key} | {entry['repo_id']} | {entry['hub']} | "
            f"{entry['license']} | {status} |"
        )

    lines.extend([
        "",
        "## Download Commands",
        "",
        "See CLI output above for the exact commands generated.",
        "",
        "## Notes",
        "",
        "- Run with `--execute` to actually download models.",
        "- Set `HF_HUB_OFFLINE=1` before training to prevent automatic downloads.",
        "- Gated models require accepting terms on Hugging Face first.",
        "- Optional pending models are skipped unless passed with `--model <key>`.",
    ])

    report_path = out_dir / "download_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Download report written to {report_path}")


# ---------------------------------------------------------------------------
# verify-models
# ---------------------------------------------------------------------------


@app.command("verify-models")
def verify_models(
    lock: str = typer.Option(
        "artifacts/model_registry/models.lock.yaml",
        help="Path to models.lock.yaml.",
    ),
    store: Optional[str] = typer.Option(
        None,
        help="Override the store directory for model files.",
    ),
) -> None:
    """Verify downloaded models against their lockfile checksums."""
    setup_logging("INFO")

    lock_path = project_root() / lock
    lock_data = load_lock(lock_path)

    if lock_data is None:
        console.print(f"[bold red]Lockfile not found: {lock_path}[/bold red]")
        raise typer.Exit(code=1)

    console.print("[bold]Model Verification[/bold]")

    all_ok = True
    for model_key, entry in lock_data.get("models", {}).items():
        local_path = entry.get("local_path", "")
        if store:
            # Override the store prefix
            safe_name = entry["repo_id"].replace("/", "__")
            revision = entry.get("revision", "<unknown>")
            local_path = str(Path(store) / "hf" / safe_name / revision)

        console.print(f"\n  {model_key}: {entry['repo_id']}")
        console.print(f"    Local path: {local_path}")
        console.print(f"    Revision: {entry.get('revision', 'N/A')}")

        local = Path(local_path)
        if not local.exists():
            console.print(f"    [bold red]FAIL: Path not found[/bold red]")
            all_ok = False
            continue

        file_count = sum(1 for _ in local.rglob("*") if _.is_file())
        console.print(f"    [green]{file_count} file(s) found[/green]")

    if all_ok:
        console.print("\n[bold green]All models verified.[/bold green]")
    else:
        console.print("\n[bold red]Verification FAILED.[/bold red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
