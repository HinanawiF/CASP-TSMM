#!/usr/bin/env bash
# Serve the web dashboard locally (no docker). Auto-refreshes results.json.
set -e
cd "$(dirname "$0")"
PORT="${1:-8000}"
echo "TSMM dashboard:  http://localhost:${PORT}/"
echo "(serving ./web ; Ctrl-C to stop)"
cd web
python3 -m http.server "${PORT}"
