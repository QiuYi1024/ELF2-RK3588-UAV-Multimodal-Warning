#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SESSION_TYPE="${2:-${ANTI_UAV_SESSION_TYPE:-realtime_camera}}"
DATA_ROOT="${ANTI_UAV_DATA_ROOT:-${ANTIUAV_DATA_ROOT:-/home/elf/AntiUAV_Data}}"
export ANTI_UAV_DATA_ROOT="$DATA_ROOT"
"$ROOT/src/data_manager/prepare_data_dirs.sh" "$ROOT" >/dev/null
SESSION_ID="$(date +%Y%m%d_%H%M%S)_${SESSION_TYPE}"
SESSION_DIR="$DATA_ROOT/sessions/$SESSION_ID"
mkdir -p \
  "$SESSION_DIR/logs/yolo" \
  "$SESSION_DIR/logs/audio" \
  "$SESSION_DIR/logs/rid900" \
  "$SESSION_DIR/logs/qt" \
  "$SESSION_DIR/config_snapshots" \
  "$SESSION_DIR/manifests"
printf '%s\n' "$SESSION_ID" >"$DATA_ROOT/runtime/pid/current_session_id"
printf '%s\n' "$SESSION_ID"