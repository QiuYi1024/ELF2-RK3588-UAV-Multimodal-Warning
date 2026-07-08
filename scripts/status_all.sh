#!/usr/bin/env bash
set -Eeuo pipefail
DATA_ROOT="${ANTI_UAV_DATA_ROOT:-/home/elf/AntiUAV_Data}"
PID_FILE="$DATA_ROOT/runtime/pid/export_run_all.pids"
if [ ! -f "$PID_FILE" ]; then echo "No export-layout PID file found."; exit 0; fi
for pid in $(cat "$PID_FILE"); do
  if kill -0 "$pid" 2>/dev/null; then echo "running pid=$pid"; else echo "not running pid=$pid"; fi
done