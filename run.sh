#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/build/pilar"

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found. Building..."
  "${ROOT_DIR}/build.sh"
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <youtube_url> <output_dir> [interval_seconds] [bottom_percent]"
  echo "Example: $0 https://www.youtube.com/watch?v=dQw4w9WgXcQ out 1.0 0.2"
  exit 1
fi

"$BIN" "$@"
