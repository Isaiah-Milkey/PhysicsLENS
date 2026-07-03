#!/usr/bin/env bash
# Launch the PhysicsLENS server (backend + statically-served frontend).
#
# Usage: ./launch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="physicslens"
PORT="${PORT:-8000}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$SCRIPT_DIR/backend"
echo "Starting PhysicsLENS on http://localhost:${PORT}"
exec uvicorn main:app --reload --port "$PORT"
