#!/usr/bin/env bash
# One-off generation with any model family:
#
#   ./gen.sh "a marble rolls off a wooden table"
#   ./gen.sh "the marble bounces" --image start.png
#   ./gen.sh "..." --model wan:t2v-a14b --steps 50
#
# For folder batches use ./run.sh instead. Env overrides: PY=
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PY:-/data/ssagar6/miniforge3/envs/physicslens/bin/python}"
[ -x "$PY" ] || { echo "ERROR: python not found at $PY (set PY=)" >&2; exit 1; }
exec "$PY" generate.py "$@"
