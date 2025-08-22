#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

for pidf in pids/*.pid; do
  [ -f "$pidf" ] || continue
  pid=$(cat "$pidf")
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping pid $pid"
    kill "$pid" || true
    sleep 0.1
  fi
  rm -f "$pidf"
done

echo "Stopped all and removed pid files."
