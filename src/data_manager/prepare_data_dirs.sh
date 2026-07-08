#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${ANTI_UAV_DATA_ROOT:-${ANTIUAV_DATA_ROOT:-/home/elf/AntiUAV_Data}}"
export ANTI_UAV_DATA_ROOT="$DATA_ROOT"
mkdir -p \
  "$DATA_ROOT/runtime/pid" \
  "$DATA_ROOT/runtime/status" \
  "$DATA_ROOT/runtime/logs" \
  "$DATA_ROOT/diagnostics/runtime/yolo" \
  "$DATA_ROOT/diagnostics/runtime/audio" \
  "$DATA_ROOT/diagnostics/runtime/rid900" \
  "$DATA_ROOT/diagnostics/runtime/qt" \
  "$DATA_ROOT/sessions"
printf '%s\n' "$DATA_ROOT"