#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 02_download_models.sh — thin wrapper for `python -m ml_final.cli download-models`
#
# Usage:
#   # Dry-run (default, safe):
#   bash scripts/02_download_models.sh --config artifacts/model_registry/models.requested.yaml
#
#   # Official Hugging Face:
#   bash scripts/02_download_models.sh --source official --store external_models/model_store --execute
#
#   # HF-Mirror (China fallback):
#   bash scripts/02_download_models.sh --source hf-mirror --store external_models/model_store --execute
#
#   # Offline verify:
#   bash scripts/02_download_models.sh --source offline --offline-verify
#
# Notes:
#   - Prefers $CONDA_PREFIX/bin/python if set, then python, python3, python3.10.
#   - Rejects Python < 3.10.
#   - Default behavior is safe (dry-run). Pass --execute to actually download.
#   - Neural-network training/evaluation should run under an active project
#     conda environment.
#   - Before executing, ensure you have accepted model terms on Hugging Face.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Find a suitable Python interpreter ---
# Priority:
#   1. $CONDA_PREFIX/bin/python  if CONDA_PREFIX is set and the binary exists
#   2. python, python3, python3.10  in PATH (conda-activated env wins via PATH)
find_python() {
    # 1. Prefer active conda environment Python
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
        return
    fi
    # 2. Fall back to PATH scanning (python first so conda activates win)
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

# --- Version check ---
PYTHON_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required. Found $("$PYTHON_BIN" --version 2>&1)." >&2
    echo "Activate a conda environment with Python >= 3.10 first." >&2
    exit 1
fi

# Resolve the full executable path using sys.executable for accuracy
PYTHON_EXECUTABLE="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Using Python: $PYTHON_EXECUTABLE ($("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'))"

# --- Set Python path ---
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# --- Run the CLI command ---
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m ml_final.cli download-models "$@"
