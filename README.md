# PII Detection Model

GLiNER-based PII detection. Fine-tuned on 105 entity types across HIPAA, PCI-DSS, General, Law Enforcement, and Transportation. No LLM, no external API calls — fully local.

**Inference latency:** 20–50 ms on GPU · 200–400 ms on CPU (NER layer only)

---

## How It Works

```
Request → Pattern Layer (Presidio) ─┐
                                     ├─ Merge → Masking → Response
Request → NER Layer (GLiNER)       ─┘
```

Runs two layers in parallel, merges spans, then applies masking rules from `entities_config.yaml`.

---

## Quick Test (model already trained)

The fine-tuned model is at `models/pii_gliner/` — just run:

```bash
./run.sh
# then:
python main.py mask-text --text "Patient John Smith, DOB 07/15/1985, SSN 078-05-1120"
# → Patient [PERSON 1], DOB [DOB 1], SSN [SSN 1]
```

---

## Adding a New Entity

1. Add an entry to `entities_config.yaml`:
   ```yaml
   - id: my_new_entity
     display_name: My New Entity
     gliner_label: My New Entity
     enabled: true
     policy: [general]
     priority: 5
     masking:
       strategy: redact
       format: "[MY_NEW_ENTITY_{n}]"
   ```

2. Run `./start.sh` on RunPod — it auto-generates 1,000 training examples for the new entity and retrains. **No code changes required.** The new entity is automatically covered by generic fallback templates if no hand-crafted templates exist for it.

3. Transfer the updated model (only 4 files, ~1.7 GB) from RunPod and restart the server.

---

## Training on RunPod (from scratch or retrain)

### Recommended GPU
| GPU | VRAM | Est. Training Time |
|-----|------|--------------------|
| A100 SXM 80 GB | 80 GB | ~45 min |
| H100 80 GB | 80 GB | ~30 min |
| RTX 4090 | 24 GB | ~90 min |

**RunPod template:** `RunPod PyTorch 2.x` with Network Volume (50 GB minimum)

### Steps

```bash
# 1. Clone the repo on RunPod
git clone https://github.com/<your-username>/<repo>.git /workspace/pii-model
cd /workspace/pii-model

# 2. Run the full pipeline — one command does everything
./start.sh

# 3. Wait for training to complete (see logs for progress)
#    Output: models/pii_gliner/  (saved after each epoch)
```

`./start.sh` handles: venv creation → dependency install → data generation → fine-tuning → `.env` update.

### start.sh options
```bash
./start.sh                          # 2000 samples/entity, auto GPU  (~90 min on A100)
./start.sh --samples 3000           # higher quality, more data      (~135 min on A100)
./start.sh --epochs 15              # override epoch count
./start.sh --skip-generate          # retrain using existing data (skip generation)
./start.sh --device cpu             # force CPU (testing only)
```

---

## Getting the Trained Model from RunPod

After training completes, the model is at `/workspace/pii-model/models/pii_gliner/` on RunPod.

### Output files (what you need to download)

| File | Size | Purpose |
|------|------|---------|
| `pytorch_model.bin` | ~1.7 GB | Fine-tuned weights — **this is the trained model** |
| `gliner_config.json` | ~2 KB | Model architecture config |
| `tokenizer.json` | ~8 MB | Tokenizer vocabulary |
| `tokenizer_config.json` | ~1 KB | Tokenizer settings |

> **Note:** The full checkpoint directory on RunPod can be 5+ GB because it includes optimizer states (Adam momentum buffers). You only need the 4 files above for inference — they're ~1.7 GB total.

### Transfer with runpodctl

```bash
# On RunPod — send only the 4 inference files (~1.7 GB)
cd /workspace/pii-model/models/pii_gliner
runpodctl send pytorch_model.bin gliner_config.json tokenizer.json tokenizer_config.json

# On your Mac — install runpodctl if needed
brew install runpod/runpodctl/runpodctl

# Receive the files (use the code shown by runpodctl send)
mkdir -p models/pii_gliner
cd models/pii_gliner
runpodctl receive <code-from-send>
```

### Place files
Drop all 4 files into `models/pii_gliner/` in your local project root. Then set in `.env`:
```
FINE_TUNED_MODEL_PATH=models/pii_gliner
GLINER_THRESHOLD=0.55
```

---

## Running the Inference Server

### Local (Mac / Linux)
```bash
./run.sh
```

Open `http://localhost:8000` in your browser.

### Configuration (`.env`)

Copy `.env.example` to `.env` and set values for your environment. Key variables:

| Variable | Purpose |
|----------|---------|
| `FINE_TUNED_MODEL_PATH` | Path to fine-tuned GLiNER weights (empty = base model) |
| `RUNPOD_BASE_URL` | Remote RunPod inference URL for side-by-side comparison |
| `RUNPOD_PROXY_PATH` | Local proxy path (default `/proxy/runpod`) |
| `LOCAL_MODEL_*` / `REMOTE_MODEL_*` | UI labels for the comparison page |
| `PORT` / `HOST` | Server bind address (used by `./run.sh`) |

Leave `RUNPOD_BASE_URL` empty to run local-only masking (no remote comparison panel).

### Docker
```bash
# Build
docker build -t pii-model .

# Run with base GLiNER (no fine-tuning)
docker run -p 8000:8000 pii-model

# Run with fine-tuned model (mount your model directory)
docker run -p 8000:8000 \
  -v $(pwd)/models/pii_gliner:/app/models/pii_gliner \
  -e FINE_TUNED_MODEL_PATH=/app/models/pii_gliner \
  -e GLINER_THRESHOLD=0.55 \
  -e RUNPOD_BASE_URL=https://your-pod-id-8000.proxy.runpod.net \
  pii-model
```

---

## Testing

```bash
# CLI
python main.py mask-text --text "Patient John Smith, DOB 07/15/1985, SSN 078-05-1120"
# → Patient [PERSON 1], DOB [DOB 1], SSN [SSN 1]

# API (server must be running)
curl -X POST http://localhost:8000/mask \
  -H "Content-Type: application/json" \
  -d '{"text": "Call John at 555-867-5309 or john@acme.com"}'
```

---

## Project Structure

```
entities_config.yaml        ← add/edit entity types here (single source of truth)
start.sh                    ← GPU training entry point (RunPod)
run.sh                      ← inference server (local)
Dockerfile                  ← inference container
requirements.txt            ← inference dependencies
README.md

train/
  generate_data.py          ← synthetic data generator (auto-covers new entities)
  finetune.py               ← GLiNER fine-tuning (A100/GPU/MPS/CPU auto-detected)
  requirements-train.txt    ← lean training deps (no torch — inherited from RunPod system)

pipeline/
  ner_layer.py              ← GLiNER inference
  pattern_layer.py          ← Presidio regex patterns
  orchestrator.py           ← runs both layers in parallel
  masking_engine.py         ← applies masking from entities_config.yaml
  preprocessor.py
  span_merger.py

models/
  schemas.py
  pii_gliner/               ← fine-tuned model output (git-ignored, downloaded from RunPod)
    pytorch_model.bin       ← weights (~1.7 GB)
    gliner_config.json
    tokenizer.json
    tokenizer_config.json

strategies/
  masking_strategies.py     ← redact / substitute / hash / partial_redact / encrypt

static/                     ← web UI assets
utils/                      ← logging, text utilities
app.py                      ← FastAPI application
main.py                     ← CLI entry point
config.py                   ← configuration loader
```

**Git-ignored (never committed):**
- `train/data/` — generated training data, recreated by `./start.sh`
- `models/pii_gliner/` — trained model weights, downloaded from RunPod after training
- `.venv/`, `.env`, `__pycache__/`

---

## GPU Hardware Profiles (auto-selected by start.sh / finetune.py)

| GPU VRAM | Profile | Batch | Epochs | fp16 |
|----------|---------|-------|--------|------|
| ≥ 40 GB (A100 / H100) | A100 | 32 | 10 | yes |
| < 40 GB (RTX 4090 etc.) | GPU | 16 | 10 | yes |
| Apple MPS (≥ 14 GB free) | MPS | 4 × 4 | 10 | no |
| CPU | CPU | 4 × 4 | 5 | no |

---

## Training Data — How It's Generated

- **105 entities** — all sourced from `entities_config.yaml`
- **2,000 examples per entity** (default) — change with `--samples N`
- **12 templates per entity** on average — diverse sentence patterns, not memorized
- **Hard negatives** — ~268 sentences with no PII (teaches the model what NOT to flag)
- **Multi-entity documents** — 10,000 realistic documents mixing 8–9 entity types each
- **New entities** — auto-covered by generic fallback templates; no code change needed
- **Single unified format** — all examples: `{"tokenized_text": [...], "ner": [[start, end, "label"], ...]}`

Total training set at 2,000 samples: ~240,000+ examples.

### Adding a new entity — what happens to existing data?

Running `./start.sh` regenerates the **entire** dataset from scratch (all 105 entities). This ensures:
- Consistent format across all entities
- No stale examples from previous runs
- The new entity gets 1,000 samples alongside all existing ones

Use `--skip-generate` to retrain on existing data without regenerating.
