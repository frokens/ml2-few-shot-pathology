#!/usr/bin/env bash
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

PYTHON_BIN="${PYTHON_BIN:-$(find_python)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python >= 3.10 interpreter found." >&2
    exit 1
fi

PYTHON_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python >= 3.10 required. Found $("$PYTHON_BIN" --version 2>&1)." >&2
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT_ROOT/hf_home/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m ml_final.cli probe-conch-prompts "$@"
