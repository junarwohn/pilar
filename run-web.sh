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

# Use a single worker because in-memory state is not shared across workers.
gunicorn -w 1 -k gthread --threads 8 --timeout 0 --bind 0.0.0.0:8000 pilar.web.wsgi:app
