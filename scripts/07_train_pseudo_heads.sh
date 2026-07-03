#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 07_train_pseudo_heads.sh — thin wrapper for `python -m ml_final.cli train-pseudo-heads`
#
# Usage:
#   bash scripts/07_train_pseudo_heads.sh --config configs/scheme_03/pseudo_retrain.yaml --features artifacts/features/scheme_01 --pseudolabels artifacts/pseudolabels/scheme_03/selected_pseudolabels.csv --run-name pseudo_heads_main4
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_python() {
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
        return
    fi
    for candidate in python python3 python3.10; do
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
exec "$PYTHON_BIN" -m ml_final.cli train-pseudo-heads "$@"

