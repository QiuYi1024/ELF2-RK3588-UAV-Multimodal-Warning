#!/usr/bin/env bash
set -Eeuo pipefail
DATA_ROOT="${ANTI_UAV_DATA_ROOT:-/home/elf/AntiUAV_Data}"
PID_FILE="$DATA_ROOT/runtime/pid/export_run_all.pids"
if [ -f "$PID_FILE" ]; then
  for pid in $(cat "$PID_FILE"); do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
  done
  rm -f "$PID_FILE"
fi
echo "Stopped export-layout background services."