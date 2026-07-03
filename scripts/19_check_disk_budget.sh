#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 19_check_disk_budget.sh — fail early when the experiment disk is too full.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MIN_FREE_GB="${ML_FINAL_MIN_FREE_GB:-10}"

if [ "${ML_FINAL_DISABLE_DISK_GUARD:-0}" = "1" ]; then
    echo "disk_guard=disabled"
    exit 0
fi

case "$MIN_FREE_GB" in
    ''|*[!0-9]*)
        echo "ERROR: ML_FINAL_MIN_FREE_GB must be a positive integer, got '$MIN_FREE_GB'" >&2
        exit 2
        ;;
esac

AVAILABLE_KB="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
REQUIRED_KB="$((MIN_FREE_GB * 1024 * 1024))"

echo "disk_guard=enabled"
echo "disk_guard_path=$PROJECT_ROOT"
echo "disk_guard_min_free_gb=$MIN_FREE_GB"
echo "disk_guard_available_kb=$AVAILABLE_KB"

if [ "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]; then
    echo "ERROR: free disk is below ${MIN_FREE_GB}G for $PROJECT_ROOT" >&2
    df -h "$PROJECT_ROOT" >&2
    exit 1
fi

df -h "$PROJECT_ROOT"
du -sh "$PROJECT_ROOT"/runs "$PROJECT_ROOT"/artifacts "$PROJECT_ROOT"/logs "$PROJECT_ROOT"/external_models 2>/dev/null || true
