#!/usr/bin/env bash
set -Eeuo pipefail
DATA_ROOT="${ANTI_UAV_DATA_ROOT:-/home/elf/AntiUAV_Data}"
SESSION_ID="${1:-${ANTI_UAV_SESSION_ID:-}}"
if [ -z "$SESSION_ID" ]; then echo "Usage: $0 SESSION_ID" >&2; exit 2; fi
SESSION_DIR="$DATA_ROOT/sessions/$SESSION_ID"
if [ ! -d "$SESSION_DIR" ]; then echo "Session not found: $SESSION_DIR" >&2; exit 1; fi
tar -czf "${SESSION_ID}_metadata_only.tar.gz" -C "$DATA_ROOT/sessions" "$SESSION_ID" --exclude='*.mp4' --exclude='*.avi' --exclude='*.wav' --exclude='*.pcm' --exclude='*.raw' --exclude='raw_*' --exclude='*.log'
echo "Exported metadata package: ${SESSION_ID}_metadata_only.tar.gz"