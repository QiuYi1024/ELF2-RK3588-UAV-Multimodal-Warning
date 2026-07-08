#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENTRY="$ROOT/src/audio/audio_beamforming_detector/run_realtime_respeaker.py"
CONFIG="${ANTI_UAV_AUDIO_CONFIG:-$ROOT/src/audio/audio_beamforming_detector/configs/respeaker4.yaml}"
MODEL="${ANTI_UAV_YAMNET_RKNN:-$ROOT/models/drone_yamnet.rknn}"
export ANTI_UAV_YAMNET_RKNN="$MODEL"
exec python3 "$ENTRY" --config "$CONFIG"