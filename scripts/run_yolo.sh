#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_MODEL="${ANTI_UAV_YOLO_MODEL:-$ROOT/models/best_uav_headless_i8.rknn}"
BIN="${ANTI_UAV_YOLO_BIN:-$ROOT/build/vision/uav_hikvision_tracker_int/uav_hikvision_tracker_int}"
if [ ! -x "$BIN" ]; then
  echo "YOLO executable not found: $BIN" >&2
  echo "Build src/vision/uav_hikvision_tracker_int/CMakeLists.txt first, then set ANTI_UAV_YOLO_BIN if needed." >&2
  exit 1
fi
exec "$BIN" --model "$YOLO_MODEL"