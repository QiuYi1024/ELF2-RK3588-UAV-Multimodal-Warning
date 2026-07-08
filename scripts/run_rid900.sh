#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/src/rid/rid900_reader.py"
exec python3 "$SCRIPT" --device "${ANTI_UAV_RID900_DEVICE:-auto}" --host "${ANTI_UAV_RID900_HOST:-127.0.0.1}" --port "${ANTI_UAV_RID900_PORT:-5009}"