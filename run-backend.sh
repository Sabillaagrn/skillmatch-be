#!/usr/bin/env bash
# Start the SkillMatch API locally — always from the repo root so the `ml`
# package is importable.
#
#   ./run-backend.sh         # real model (needs TensorFlow + Keras installed)
#   ./run-backend.sh mock    # mock mode (no TensorFlow needed)
#
set -euo pipefail
cd "$(dirname "$0")"          # repo root

PORT="${PORT:-8000}"

# Quiet TensorFlow's startup logging (cosmetic INFO/oneDNN messages).
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

if [[ "${1:-}" == "mock" ]]; then
  export SKILLMATCH_MOCK=true
  echo "▶ Starting in MOCK mode (no TensorFlow) on http://localhost:${PORT}  — instant startup"
else
  echo "▶ Starting with the real model on http://localhost:${PORT}"
  echo "  (first boot takes ~10-20s: importing TensorFlow + loading the model — this is normal)"
fi

exec uvicorn backend.app.main:app --reload --port "${PORT}"
