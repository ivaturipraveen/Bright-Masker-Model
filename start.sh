#!/usr/bin/env bash
# =============================================================================
# PII Model — Single entry point for GPU training (RunPod / any Linux GPU box)
#
# Usage:
#   chmod +x start.sh && ./start.sh                   # auto-detect GPU, 2000 samples/entity
#   ./start.sh --samples 3000                         # even more training data
#   ./start.sh --epochs 15                            # override epoch count
#   ./start.sh --device cpu                           # force CPU (dev/testing)
#   ./start.sh --skip-generate                        # retrain on existing data only
#
# What it does:
#   1. Creates / reuses a Python virtual environment (inherits system CUDA torch)
#   2. Installs lean training dependencies
#   3. Generates synthetic PII training data from entities_config.yaml
#      → any new entity added to the yaml is covered automatically (no code change)
#   4. Fine-tunes GLiNER on GPU (auto-selects A100 profile when VRAM >= 40 GB)
#   5. Saves the trained model to models/pii_gliner/
#   6. Updates .env with the model path
#
# RunPod recommended instance: A100 SXM 80 GB or H100 80 GB
# Estimated time: ~90 min on A100 at 2000 samples/entity (105 entities)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

# ── Defaults ──────────────────────────────────────────────────────────────────
SAMPLES=2000
DEVICE_ARG=""
EPOCHS_ARG=""
BASE_MODEL="urchade/gliner_large-v2.1"
OUT_DIR="$ROOT/models/pii_gliner"
DATA_FILE="$ROOT/train/data/pii_train.json"
SKIP_GENERATE=0

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --samples)       SAMPLES="$2";      shift 2 ;;
    --device)        DEVICE_ARG="$2";   shift 2 ;;
    --epochs)        EPOCHS_ARG="$2";   shift 2 ;;
    --out)           OUT_DIR="$2";      shift 2 ;;
    --base-model)    BASE_MODEL="$2";   shift 2 ;;
    --skip-generate) SKIP_GENERATE=1;   shift   ;;
    --help|-h)
      sed -n '2,30p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo ""
echo "============================================================"
echo "  GLiNER PII Training Pipeline"
echo "============================================================"
echo "  Root          : $ROOT"
echo "  Samples/entity: $SAMPLES"
echo "  Output model  : $OUT_DIR"
echo ""

# ── Detect Python ─────────────────────────────────────────────────────────────
if [ -d "$VENV" ]; then
  PYTHON="$VENV/bin/python"
  PIP="$VENV/bin/pip"
  echo "[1/5] Reusing existing virtual environment"
else
  # Find best available Python
  SYS_PYTHON=""
  for py in python3.11 python3.10 python3.9 python3; do
    if command -v "$py" &>/dev/null; then
      SYS_PYTHON="$py"
      break
    fi
  done
  if [ -z "$SYS_PYTHON" ]; then
    echo "ERROR: No python3 found on PATH" >&2; exit 1
  fi

  echo "[1/5] Creating virtual environment with $SYS_PYTHON …"
  # --system-site-packages inherits the RunPod pre-installed CUDA-enabled torch.
  # Without this, pip downloads a CPU-only torch and CUDA is invisible to training.
  "$SYS_PYTHON" -m venv --system-site-packages "$VENV"
  PYTHON="$VENV/bin/python"
  PIP="$VENV/bin/pip"
fi

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[2/5] Installing training dependencies…"
"$PIP" install -q --upgrade pip

# Install everything except torch/torchvision — those come from the system venv
# (already CUDA-enabled on RunPod). Installing torch again would overwrite with CPU.
"$PIP" install -q \
  "pyyaml>=6.0" \
  "faker>=24.0" \
  "gliner>=0.2" \
  "transformers>=4.40" \
  "accelerate>=0.26.0"

# Verify CUDA is visible
echo ""
"$PYTHON" -c "
import torch, gliner
cuda = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda else 'CPU (no CUDA)'
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda else 0
print(f'  gliner {gliner.__version__}  |  torch {torch.__version__}')
print(f'  Device : {device_name}' + (f'  ({vram:.0f} GB VRAM)' if cuda else ''))
if not cuda:
    print('  WARNING: CUDA not found — training will use CPU (slow)')
"

# ── Ensure output directories exist ───────────────────────────────────────────
mkdir -p "$ROOT/train/data"
mkdir -p "$OUT_DIR"

# ── Generate training data ────────────────────────────────────────────────────
if [ "$SKIP_GENERATE" -eq 0 ]; then
  echo ""
  echo "[3/5] Generating training data from entities_config.yaml…"
  echo "  Samples per entity: $SAMPLES"
  echo "  (New entities in entities_config.yaml are covered automatically)"
  "$PYTHON" "$ROOT/train/generate_data.py" \
    --samples "$SAMPLES" \
    --yaml    "$ROOT/entities_config.yaml" \
    --out     "$DATA_FILE"
else
  echo "[3/5] Skipping data generation (--skip-generate)"
fi

if [ ! -f "$DATA_FILE" ]; then
  echo "ERROR: Training data not found at $DATA_FILE" >&2
  exit 1
fi

EXAMPLE_COUNT=$("$PYTHON" -c "import json; d=json.load(open('$DATA_FILE')); print(len(d))")
echo "  Training examples: $EXAMPLE_COUNT"

# ── Fine-tune GLiNER ──────────────────────────────────────────────────────────
echo ""
echo "[4/5] Fine-tuning GLiNER…"

FINETUNE_ARGS=(
  "--data"       "$DATA_FILE"
  "--out"        "$OUT_DIR"
  "--base-model" "$BASE_MODEL"
)
[ -n "$DEVICE_ARG" ] && FINETUNE_ARGS+=("--device" "$DEVICE_ARG")
[ -n "$EPOCHS_ARG" ] && FINETUNE_ARGS+=("--epochs" "$EPOCHS_ARG")

"$PYTHON" "$ROOT/train/finetune.py" "${FINETUNE_ARGS[@]}"

# ── Update .env ───────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Updating .env…"

ENV_FILE="$ROOT/.env"
touch "$ENV_FILE"

if grep -q "^FINE_TUNED_MODEL_PATH=" "$ENV_FILE" 2>/dev/null; then
  sed -i "s|^FINE_TUNED_MODEL_PATH=.*|FINE_TUNED_MODEL_PATH=$OUT_DIR|" "$ENV_FILE"
else
  echo "FINE_TUNED_MODEL_PATH=$OUT_DIR" >> "$ENV_FILE"
fi

if grep -q "^GLINER_THRESHOLD=" "$ENV_FILE" 2>/dev/null; then
  sed -i "s|^GLINER_THRESHOLD=.*|GLINER_THRESHOLD=0.55|" "$ENV_FILE"
else
  echo "GLINER_THRESHOLD=0.55" >> "$ENV_FILE"
fi

echo ""
echo "============================================================"
echo "  Training complete"
echo "============================================================"
echo ""
echo "  Model saved : $OUT_DIR"
echo "  .env updated: FINE_TUNED_MODEL_PATH=$OUT_DIR"
echo ""
echo "  To download model from RunPod:"
echo "    runpodctl send $OUT_DIR"
echo ""
echo "  To serve locally after download:"
echo "    ./run.sh"
echo ""
echo "  To test:"
echo "    python main.py mask-text --text 'Patient John Smith DOB 07/15/1985 SSN 078-05-1120'"
echo ""
