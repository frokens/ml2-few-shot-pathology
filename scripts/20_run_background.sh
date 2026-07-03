#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 20_run_background.sh — start one project command with nohup logging.
#
# Usage:
#   bash scripts/20_run_background.sh s02_aug_weak_geom -- \
#     bash scripts/08_train_peft_cv.sh --config configs/scheme_02/peft_aug_weak_geom.yaml
#
# Monitor:
#   tail -f logs/s02_aug_weak_geom.out
#   cat logs/s02_aug_weak_geom.pid
# ---------------------------------------------------------------------------

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    echo "Usage: bash scripts/20_run_background.sh RUN_NAME -- COMMAND [ARGS...]" >&2
    exit 2
fi

RUN_NAME="$1"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/${RUN_NAME}.out"
PID_PATH="$LOG_DIR/${RUN_NAME}.pid"
STATUS_PATH="$LOG_DIR/${RUN_NAME}.status"
DISK_SNAPSHOT_PATH="$LOG_DIR/${RUN_NAME}.disk.txt"

if [ -f "$PID_PATH" ]; then
    OLD_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "ERROR: run '$RUN_NAME' is already active with pid $OLD_PID" >&2
        echo "Log: $LOG_PATH" >&2
        exit 1
    fi
fi

{
    echo "run_name=$RUN_NAME"
    echo "started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "cwd=$PROJECT_ROOT"
    printf "command="
    printf "%q " "$@"
    echo
    echo "git_commit=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "python=$(command -v python 2>/dev/null || true)"
    echo "conda_prefix=${CONDA_PREFIX:-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "disk_snapshot=$DISK_SNAPSHOT_PATH"
} > "$STATUS_PATH"

cd "$PROJECT_ROOT"
bash scripts/19_check_disk_budget.sh > "$DISK_SNAPSHOT_PATH"
nohup env PYTHONUNBUFFERED=1 bash -c '
    status_path="$1"
    shift
    set +e
    "$@"
    exit_code=$?
    {
        echo "finished_at=$(date -u "+%Y-%m-%dT%H:%M:%SZ")"
        echo "exit_code=$exit_code"
    } >> "$status_path"
    exit "$exit_code"
' _ "$STATUS_PATH" "$@" > "$LOG_PATH" 2>&1 &
PID="$!"
echo "$PID" > "$PID_PATH"
echo "pid=$PID" >> "$STATUS_PATH"

echo "Started $RUN_NAME"
echo "  PID: $PID"
echo "  Log: $LOG_PATH"
echo "  Tail: tail -f $LOG_PATH"
echo "  Status: cat $STATUS_PATH"
