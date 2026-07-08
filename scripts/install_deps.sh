#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m pip install -r "$ROOT/requirements.txt"
echo "Build C++/Qt components from their CMakeLists.txt files before running vision or Qt binaries."