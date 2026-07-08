#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m compileall -q "$ROOT/src"
(cd "$ROOT/src/rid" && python3 -m unittest discover -p "test_*.py")
for f in "$ROOT"/scripts/*.sh "$ROOT"/src/data_manager/*.sh; do
  [ -f "$f" ] && bash -n "$f"
done
echo "Verification passed."