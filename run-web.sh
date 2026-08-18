#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for Web UI
# Usage: ./run-web.sh [PORT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT="${1:-${PORT:-8000}}"
RESULT_RETENTION_DAYS="${RESULT_RETENTION_DAYS:-7}"
PILAR_CLEANUP_OLD_RESULTS="${PILAR_CLEANUP_OLD_RESULTS:-1}"

cleanup_old_results() {
  if [ "${PILAR_CLEANUP_OLD_RESULTS}" != "1" ]; then
    echo "Skipping old result cleanup"
    return
  fi

  if ! [[ "${RESULT_RETENTION_DAYS}" =~ ^[0-9]+$ ]] || [ "${RESULT_RETENTION_DAYS}" -lt 1 ]; then
    echo "Invalid RESULT_RETENTION_DAYS: ${RESULT_RETENTION_DAYS}" >&2
    exit 1
  fi

  local results_dir="out"
  local mtime_days=$((RESULT_RETENTION_DAYS - 1))

  if [ ! -d "${results_dir}" ]; then
    return
  fi

  local file_count
  file_count="$(find "${results_dir}" -mindepth 2 -type f -mtime +"${mtime_days}" -print | wc -l)"
  find "${results_dir}" -mindepth 2 -type f -mtime +"${mtime_days}" -delete

  local dir_count
  dir_count="$(find "${results_dir}" -mindepth 1 -type d -empty -mtime +"${mtime_days}" -print | wc -l)"
  find "${results_dir}" -mindepth 1 -type d -empty -mtime +"${mtime_days}" -delete

  echo "Cleaned ${file_count} old result file(s) and ${dir_count} empty result folder(s) older than ${RESULT_RETENTION_DAYS} day(s)"
}

# Activate local venv if present
if [ -f "./pilar-venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ./pilar-venv/bin/activate
fi

cleanup_old_results

URL="http://localhost:${PORT}"
echo "Starting Web UI on ${URL}"

# Note: Do not auto-open a browser

# Use a single worker because in-memory state is not shared across workers.
gunicorn -w 1 -k gthread --threads 8 --timeout 0 --bind "0.0.0.0:${PORT}" pilar.web.wsgi:app
