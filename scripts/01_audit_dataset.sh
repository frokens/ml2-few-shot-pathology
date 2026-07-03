#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01_audit_dataset.sh — thin wrapper for `python -m ml_final.cli audit-dataset`
#
# Usage:
#   bash scripts/01_audit_dataset.sh \
#     --train-dir train_few_shot \
#     --test-dir test \
#     --out artifacts/dataset_audit
#
# Notes:
#   - Prefers $CONDA_PREFIX/bin/python if set, then python, python3, python3.10.
#   - Rejects Python < 3.10.
#   - This stage is offline-only and never downloads model weights or data.
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
    echo "Activate a conda environment first, for example: conda activate base" >&2
    exit 1
fi

PYTHON_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required. Found $("$PYTHON_BIN" --version 2>&1)." >&2
    echo "Activate a conda environment with Python >= 3.10 first." >&2
    exit 1
fi

PYTHON_EXECUTABLE="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
echo "Using Python: $PYTHON_EXECUTABLE ($PYTHON_VERSION)"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m ml_final.cli audit-dataset "$@"
