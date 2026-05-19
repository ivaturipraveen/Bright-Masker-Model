#!/usr/bin/env bash
set -euo pipefail

# Read a single value from .env without sourcing it (avoids () issues in values).
_env_val() {
  grep -E "^${1}=" .env 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/^["'\'']//; s/["'\'']$//'
}

VENV=".venv"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
if [ -f .env ]; then
  _p="$(_env_val PORT)"; [ -n "$_p" ] && PORT="$_p"
  _h="$(_env_val HOST)"; [ -n "$_h" ] && HOST="$_h"
fi

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi

# ── 2. Dependencies ───────────────────────────────────────────────────────────
echo "Installing dependencies..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements.txt

# ── 3. spaCy model ────────────────────────────────────────────────────────────
SPACY_MODEL="${SPACY_MODEL_NAME:-en_core_web_sm}"
if ! "$VENV/bin/python" -c "import spacy; spacy.load('$SPACY_MODEL')" 2>/dev/null; then
  echo "Downloading spaCy model: $SPACY_MODEL"
  "$VENV/bin/pip" install -q "https://github.com/explosion/spacy-models/releases/download/${SPACY_MODEL}-3.8.0/${SPACY_MODEL}-3.8.0-py3-none-any.whl"
fi

# ── 4. .env check ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo "  ERROR: No .env file found. Copy .env.example to .env and set your values."
  exit 1
fi

# ── 5. S3 model weights ───────────────────────────────────────────────────────
S3_BUCKET="$(_env_val S3_BUCKET)"
S3_MODEL_PREFIX="$(_env_val S3_MODEL_PREFIX)"
MODEL_DIR="$(_env_val FINE_TUNED_MODEL_PATH)"
MODEL_DIR="${MODEL_DIR:-models/pii_gliner}"
S3_BUCKET="${S3_BUCKET:-brightmasker}"
S3_MODEL_PREFIX="${S3_MODEL_PREFIX:-models/pii_gliner}"

MODEL_BIN="$MODEL_DIR/pytorch_model.bin"

if [ ! -f "$MODEL_BIN" ]; then
  echo ""
  echo "  Model weights not found at $MODEL_DIR — downloading from S3..."

  # Install awscli into venv if not present
  if ! "$VENV/bin/python" -c "import awscli" 2>/dev/null; then
    "$VENV/bin/pip" install -q awscli
  fi

  mkdir -p "$MODEL_DIR"

  AWS_ACCESS_KEY_ID="$(_env_val AWS_ACCESS_KEY_ID)" \
  AWS_SECRET_ACCESS_KEY="$(_env_val AWS_SECRET_ACCESS_KEY)" \
  AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
  "$VENV/bin/aws" s3 sync "s3://${S3_BUCKET}/${S3_MODEL_PREFIX}/" "$MODEL_DIR/" \
    --exclude "*" \
    --include "pytorch_model.bin" \
    --include "gliner_config.json" \
    --include "tokenizer.json" \
    --include "tokenizer_config.json" \
    --no-progress

  echo "  Model downloaded: $MODEL_DIR"
else
  echo "  Model weights found at $MODEL_DIR — skipping S3 download."
fi

# ── 6. NVIDIA GPU optimizations ───────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  echo ""
  echo "  Applying NVIDIA optimizations..."

  # Persistence mode — keeps GPU driver loaded; cuts cold-start latency on first request
  nvidia-smi -pm 1 2>/dev/null || true

  # Max clocks — locks GPU to highest stable frequency for consistent latency
  nvidia-smi --auto-boost-default=0 2>/dev/null || true
  nvidia-smi -ac "$(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits | head -1 | tr ',' ' ' | awk '{print $1","$2}')" 2>/dev/null || true

  # Show GPU info
  nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.sm \
    --format=csv,noheader | awk '{print "  GPU: " $0}'
fi

# ── 7. Start server ───────────────────────────────────────────────────────────
echo ""
echo "  Starting Bright Masker on http://$HOST:$PORT"
echo "  Open http://localhost:$PORT in your browser to test."
echo "  Press Ctrl+C to stop."
echo ""

if [[ "${UVICORN_RELOAD:-1}" == "0" ]]; then
  exec "$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT"
else
  exec "$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT" --reload
fi
