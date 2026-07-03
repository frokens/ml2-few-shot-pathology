#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
if [ -z "${PYTHON_BIN:-}" ] || [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m ml_final.cli infer-final "$@"

