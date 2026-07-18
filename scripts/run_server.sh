#!/usr/bin/env bash
# run_server.sh — start llama-server by hand, OUTSIDE FreeCAD.
# For testing the model/client (e.g. pdf2cad/pipeline.py) from a terminal.
# Inside FreeCAD the workbench manages the server itself — don't run both
# at once on the same port.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$REPO/models/gemma-4-E2B-it-Q8_0.gguf}"
MMPROJ="${2:-$REPO/models/mmproj-gemma-4-E2B-it-Q8_0.gguf}"
PORT="${PORT:-8735}"

exec llama-server \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --host 127.0.0.1 --port "$PORT" \
    -c 8192 -ngl 99 --keep -1
