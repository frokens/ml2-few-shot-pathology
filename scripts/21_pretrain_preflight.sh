#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 21_pretrain_preflight.sh — final no-training checks before formal runs.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODE="${1:-no-card}"

find_python() {
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
        return
    fi
    for candidate in python python3 python3.12 python3.11 python3.10; do
        local resolved
        resolved="$(command -v "$candidate" 2>/dev/null)" || continue
        if [ -n "$resolved" ]; then
            echo "$resolved"
            return
        fi
    done
    echo ""
}

PYTHON_BIN="$(find_python)"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python >= 3.10 interpreter found." >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

echo "preflight_mode=$MODE"
echo "project_root=$PROJECT_ROOT"
echo "python=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
bash scripts/19_check_disk_budget.sh

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

from pathlib import Path

import yaml

from ml_final.features.extract import count_tta_views
from ml_final.utils.config import load_yaml, resolve_project_path
from ml_final.utils.safety import assert_no_forbidden_reference_paths

required_configs = [
    "configs/scheme_01/extract_features_single.yaml",
    "configs/scheme_01/extract_features_d4.yaml",
    "configs/scheme_01/extract_features_d4_expand.yaml",
    "configs/scheme_01/conch_prompt_probe.yaml",
    "configs/scheme_01/cv_heads.yaml",
    "configs/scheme_02/peft_cv.yaml",
    "configs/scheme_02/peft_gpu_smoke_all4.yaml",
    "configs/scheme_02/fusion_peft_smoke.yaml",
    "configs/scheme_03/teacher_s01.yaml",
    "configs/scheme_03/pseudolabel.yaml",
    "configs/scheme_03/pseudo_retrain_lam01.yaml",
    "configs/scheme_03/pseudo_lora_lam005.yaml",
    "configs/scheme_01/ensemble.yaml",
]

print("SECTION:config_parse")
for path in required_configs:
    cfg = load_yaml(path)
    assert_no_forbidden_reference_paths(cfg, context=f"preflight {path}")
    print(f"ok {path}")

print("SECTION:required_files")
for path in [
    "artifacts/dataset_audit/train_manifest.csv",
    "artifacts/model_registry/models.lock.yaml",
    "scripts/20_run_background.sh",
    "scripts/19_check_disk_budget.sh",
]:
    resolved = resolve_project_path(path)
    print(f"{path} exists={bool(resolved and resolved.exists())}")

print("SECTION:selection_dependencies")
for path in [
    "artifacts/selection/reference_class_mapping.yaml",
    "artifacts/selection/scheme01_teacher_predictions.txt",
    "artifacts/selection/scheme01_best_peft_backbones.txt",
    "artifacts/selection/scheme01_top1_peft_backbone.txt",
]:
    resolved = resolve_project_path(path)
    print(f"{path} exists={bool(resolved and resolved.exists())}")

print("SECTION:parameter_chain")
peft = load_yaml("configs/scheme_02/peft_cv.yaml")
print(f"scheme02_ema_decay={peft.get('ema_decay')}")
print(f"scheme02_max_grad_norm={peft.get('max_grad_norm')}")
print(f"scheme02_gradient_checkpointing={peft.get('gradient_checkpointing')}")
print(f"scheme02_test_prediction_mode={peft.get('test_prediction_mode')}")
print(f"scheme02_experiments={','.join(str(item.get('name')) for item in peft.get('experiments', []))}")

teacher = load_yaml("configs/scheme_03/teacher_s01.yaml")
print(f"teacher_sources_file={teacher.get('teacher_sources_file')}")
print(f"teacher_allow_prediction_search={teacher.get('allow_prediction_search')}")

ensemble = load_yaml("configs/scheme_01/ensemble.yaml")
print(f"final_prediction_sources_file={ensemble.get('test_prediction_sources_file')}")
print(f"final_allow_prediction_search={ensemble.get('allow_prediction_search')}")

print("SECTION:tta")
print(f"d4_views={count_tta_views('d4')}")
print(f"geom6_views={count_tta_views('geom6')}")

print("SECTION:model_registry")
lock_path = resolve_project_path("artifacts/model_registry/models.lock.yaml")
if lock_path and lock_path.exists():
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    for key in ["uni2_h", "virchow2", "conch", "h_optimus_0"]:
        item = (payload.get("models") or {}).get(key)
        local_path = resolve_project_path(item.get("local_path")) if item else None
        print(f"{key} locked={bool(item)} local_exists={bool(local_path and local_path.exists())}")
else:
    print("models.lock.yaml missing")
PY

if [ "$MODE" = "gpu" ]; then
    "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: gpu preflight requires at least one CUDA device")
print("SECTION:cuda")
print("cuda_available=True")
print("cuda_count", torch.cuda.device_count())
print("cuda_names", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
else
    "$PYTHON_BIN" - <<'PY'
import torch
print("SECTION:cuda")
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
PY
fi

echo "preflight_status=ok"
