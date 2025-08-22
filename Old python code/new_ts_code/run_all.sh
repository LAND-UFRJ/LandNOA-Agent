#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Building TypeScript..."
npm run build

mkdir -p logs pids

start_service() {
  name=$1; file=$2; port=$3
  logf="logs/${name}.log"
  pidf="pids/${name}.pid"
  if [ -f "$pidf" ] && kill -0 "$(cat $pidf)" 2>/dev/null; then
    echo "$name already running (pid=$(cat $pidf))"
    return
  fi
  echo "Starting $name -> $file (port $port)"
  nohup node dist/$file.js > "$logf" 2>&1 &
  echo $! > "$pidf"
  sleep 0.2
}

# start registry, ai guide, biologist, host
start_service registry registry 8080
start_service ai_guide_agent ai_guide_agent 8010
start_service biologist_agent biologist_agent 8006
start_service host_agent host_agent 8000

echo "Started. PIDs:" && ls -l pids || true

echo "Tails (last 5 lines) of logs:"
for f in logs/*.log; do echo "---- $f"; tail -n 5 "$f" || true; done
