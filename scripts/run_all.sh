#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ANTI_UAV_DATA_ROOT="${ANTI_UAV_DATA_ROOT:-/home/elf/AntiUAV_Data}"
export ANTI_UAV_YOLO_MODEL="${ANTI_UAV_YOLO_MODEL:-$ROOT/models/best_uav_headless_i8.rknn}"
export ANTI_UAV_YAMNET_RKNN="${ANTI_UAV_YAMNET_RKNN:-$ROOT/models/drone_yamnet.rknn}"
"$ROOT/src/data_manager/prepare_data_dirs.sh" "$ROOT" >/dev/null
"$ROOT/scripts/run_rid900.sh" &
RID_PID=$!
"$ROOT/scripts/run_audio.sh" &
AUDIO_PID=$!
"$ROOT/scripts/run_yolo.sh" &
YOLO_PID=$!
echo "$RID_PID $AUDIO_PID $YOLO_PID" >"${ANTI_UAV_DATA_ROOT}/runtime/pid/export_run_all.pids"
echo "Started AntiUAV export-layout services. Qt/YOLO binaries must be built from CMakeLists.txt before use."