#!/usr/bin/env bash
# PhysicsLENS launcher — starts the FastAPI backend (which also serves the UI).
#
#   ./run.sh                # listen on 0.0.0.0:8000
#   PORT=9000 ./run.sh      # custom port
#
# Then access the UI. Because the frontend is hardcoded to talk to
# http://localhost:8000, the simplest path from your laptop is an SSH tunnel:
#
#   ssh -L 8000:localhost:8000 <user>@<this-server>
#
# ...and open http://localhost:8000 in your local browser.
set -euo pipefail

ENV_NAME="physicslens"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# shellcheck disable=SC1091
source /data/ssagar6/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

cd "$(dirname "$0")/backend"
echo "PhysicsLENS → http://localhost:${PORT}  (Ctrl-C to stop)"
exec python -m uvicorn main:app --host "$HOST" --port "$PORT"
