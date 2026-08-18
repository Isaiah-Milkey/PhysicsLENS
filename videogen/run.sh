#!/usr/bin/env bash
#
# Generate videos from whatever is in inputs/.
#
#   1. put one prompt per line in   inputs/prompts.txt
#   2. (optional) drop images in    inputs/images/
#   3. ./run.sh
#
# Results land in outputs/ as <NNN>_<name>.mp4 with a matching .json recording
# the prompt, seed, model and settings used.
#
# Mode is auto-detected: no images -> t2v, images present -> i2v.
# Override by passing it explicitly:
#
#   ./run.sh t2v          text only
#   ./run.sh i2v          text + 1 image each (pairs images with prompts in order)
#   ./run.sh flf          text + first & last frame (needs exactly 2 images)
#   ./run.sh ref          text + all images as reference subjects
#
# Anything after the mode is passed straight to wan_generate.py:
#
#   ./run.sh t2v --model t2v-a14b --steps 50
#   ./run.sh --dry-run                      # validate, download nothing
#
# Env overrides:  MODEL=... STEPS=... SEED=... OUTDIR=... PY=...
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PY:-/data/ssagar6/miniforge3/envs/physicslens/bin/python}"
PROMPTS="inputs/prompts.txt"
IMAGES="inputs/images"
OUTDIR="${OUTDIR:-outputs}"
STEPS="${STEPS:-40}"
SEED="${SEED:-0}"

if [ ! -x "$PY" ]; then
  echo "ERROR: python not found at $PY" >&2
  echo "       set PY=/path/to/python, or create the env (see README)." >&2
  exit 1
fi

if [ ! -f "$PROMPTS" ]; then
  echo "ERROR: $PROMPTS not found. Put one prompt per line in it." >&2
  exit 1
fi

# Count real prompts (skip blanks and # comments) so an all-commented file is
# reported as empty rather than silently generating nothing.
n_prompts=$(grep -cvE '^\s*(#|$)' "$PROMPTS" || true)
n_images=0
[ -d "$IMAGES" ] && n_images=$(find "$IMAGES" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \) \
  | wc -l)

if [ "$n_prompts" -eq 0 ]; then
  echo "ERROR: no prompts in $PROMPTS (blank lines and # comments don't count)." >&2
  exit 1
fi

# First arg is the mode only if it looks like one; otherwise it's a passthrough
# flag and we auto-detect.
MODE=""
case "${1:-}" in
  t2v|i2v|flf|ref) MODE="$1"; shift ;;
esac
if [ -z "$MODE" ]; then
  if [ "$n_images" -gt 0 ]; then MODE="i2v"; else MODE="t2v"; fi
  echo "[run] mode auto-detected: $MODE  ($n_images image(s) in $IMAGES/)"
fi

ARGS=(--mode "$MODE" --prompt-file "$PROMPTS" --out-dir "$OUTDIR"
      --steps "$STEPS" --seed "$SEED")
[ -n "${MODEL:-}" ] && ARGS+=(--model "$MODEL")
# t2v takes no images; every other mode reads the whole folder.
if [ "$MODE" != "t2v" ] && [ "$n_images" -gt 0 ]; then
  ARGS+=(--image-dir "$IMAGES")
fi

echo "[run] $n_prompts prompt(s), $n_images image(s) -> $OUTDIR/"
exec "$PY" wan_generate.py "${ARGS[@]}" "$@"
