#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for Web UI
# Usage: ./run-web.sh [PORT]

PORT="${1:-8000}"

# Activate local venv if present
if [ -f "./pilar-venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ./pilar-venv/bin/activate
fi

URL="http://localhost:${PORT}"
echo "Starting Web UI on ${URL}"

# Note: Do not auto-open a browser

exec python3 main.py --web --no-gui --host 0.0.0.0 --port "${PORT}"
