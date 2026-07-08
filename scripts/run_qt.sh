#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QT_BIN="${ANTI_UAV_QT_BIN:-$ROOT/build/ui/qt_dashboard/qt_dashboard}"
if [ ! -x "$QT_BIN" ]; then
  echo "Qt executable not found: $QT_BIN" >&2
  echo "Build src/ui/qt_dashboard/CMakeLists.txt first, then set ANTI_UAV_QT_BIN if needed." >&2
  exit 1
fi
exec "$QT_BIN"