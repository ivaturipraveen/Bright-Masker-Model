# Bright Masker — PII Detection

GLiNER-based PII detection. Fine-tuned on 105 entity types across HIPAA, PCI-DSS, General, Law Enforcement, and Transportation. No LLM, no external API calls — fully local.

**Inference latency:** ~50ms on RTX 4090 (GPU) · ~700ms on Apple MPS · ~2–4s on CPU

---

## How It Works

```
Request → Pattern Layer (Presidio + Regex) ─┐
                                              ├─ Merge → Masking → Response
Request → NER Layer (GLiNER Fine-tuned)    ─┘
```

Both layers run in parallel. Results are merged and masked using rules from `entities_config.yaml`.

**GPU inference strategy:** On CUDA, all 105 labels run in a single forward pass (fp16 + TF32 + cuDNN benchmark). On CPU/MPS, 6 semantic groups run in parallel via thread pool.

---

## Entity Types — 105 Total

| Group | Count | Examples |
|-------|-------|---------|
| People & Contact | 21 | person_name, physician_name, phone, email, DOB, IP, device ID |
| Address | 5 | street_address, city, state, zipcode, geolocation |
| Government IDs & Medical | 24 | SSN, passport, driver's license, MRN, NPI, DEA, medication |
| Financial | 19 | credit card, bank account, routing, IBAN, card PIN, transaction ID |
| Legal & Criminal | 21 | case number, claim, arrest record, warrant, sex offender report |
| Education & Travel | 15 | student ID, university, flight number, booking reference, hotel |

All 105 entities defined in `entities_config.yaml` — the single source of truth.

---

## Running Locally

```bash
# 1. Copy and fill in .env
cp .env.example .env
# Edit .env — add AWS keys, ENCRYPTION_KEY

# 2. Start server (weights auto-downloaded from S3 if not present)
./run.sh
```

Open `http://localhost:8000`

### .env — Required Keys Only

```env
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=brightmasker
S3_MODEL_PREFIX=models/pii_gliner

FINE_TUNED_MODEL_PATH=models/pii_gliner
GLINER_THRESHOLD=0.35

ENCRYPTION_KEY=change-this-to-a-random-secret-key
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Leave empty to disable the remote comparison panel
RUNPOD_BASE_URL=
```

---

## Deploying on RunPod

### Recommended GPU

| GPU | VRAM | Inference | Cost/hr |
|-----|------|-----------|---------|
| **RTX 4090** | 24 GB | ~50ms | ~$0.70 ✅ |
| RTX A4000 | 16 GB | ~70ms | ~$0.35 |
| A100 40GB | 40 GB | ~30ms | ~$1.99 |

Model is 1.7 GB (850 MB in fp16) — any GPU with ≥ 8 GB VRAM works.

### RunPod Setup

1. Create pod with **RunPod PyTorch 2.x** template, RTX 4090, Network Volume attached
2. Set environment variables in the RunPod panel (same keys as `.env` above)
3. Set **Container Start Command**:

```bash
bash -c "if [ -d /workspace/pii-model ]; then cd /workspace/pii-model && git pull; else git clone https://github.com/ivaturipraveen/Bright-Masker-Model.git /workspace/pii-model && cd /workspace/pii-model; fi && bash /workspace/pii-model/run.sh"
```

`run.sh` will automatically:
- Install dependencies
- Download weights from `s3://brightmasker/models/pii_gliner/` if not present
- Set NVIDIA persistence mode + max GPU clocks
- Start the server on port 8000

---

## Model Weights (S3)

Weights are stored in S3 and auto-downloaded at startup. No manual transfer needed.

```
s3://brightmasker/models/pii_gliner/pytorch_model.bin     (~1.7 GB)
s3://brightmasker/models/pii_gliner/gliner_config.json
s3://brightmasker/models/pii_gliner/tokenizer.json
s3://brightmasker/models/pii_gliner/tokenizer_config.json
```

To push updated weights to S3 after retraining:

```python
import boto3
from pathlib import Path

s3 = boto3.client('s3')
for f in ['pytorch_model.bin', 'gliner_config.json', 'tokenizer.json', 'tokenizer_config.json']:
    s3.upload_file(f'models/pii_gliner/{f}', 'brightmasker', f'models/pii_gliner/{f}')
```

---

## Training on RunPod (A100)

### Recommended GPU for Training

| GPU | VRAM | Training Time |
|-----|------|--------------|
| A100 SXM 80 GB | 80 GB | ~90 min |
| H100 80 GB | 80 GB | ~60 min |

### Steps

```bash
# On RunPod A100
cd /workspace
git clone https://github.com/ivaturipraveen/Bright-Masker-Model.git pii-model
cd pii-model
./start.sh --samples 2000
```

Training runs: venv → deps → generate data → fine-tune → save to `models/pii_gliner/`

### start.sh Options

```bash
./start.sh                     # 2000 samples/entity, auto GPU (~90 min A100)
./start.sh --samples 3000      # more data, higher quality
./start.sh --epochs 15         # override epoch count
./start.sh --skip-generate     # retrain on existing data only
```

### After Training — Upload to S3

```bash
# On RunPod — copy best checkpoint (lowest eval_loss) to model dir
cp -r models/pii_gliner/checkpoint-XXXXX/* models/pii_gliner/
rm -rf models/pii_gliner/checkpoint-*

# Push to S3
pip install boto3
python3 -c "
import boto3
s3 = boto3.client('s3', aws_access_key_id='KEY', aws_secret_access_key='SECRET', region_name='us-east-1')
for f in ['pytorch_model.bin','gliner_config.json','tokenizer.json','tokenizer_config.json']:
    print(f'Uploading {f}...')
    s3.upload_file(f'models/pii_gliner/{f}', 'brightmasker', f'models/pii_gliner/{f}')
print('Done')
"
```

---

## Adding a New Entity

1. Add to `entities_config.yaml`:
```yaml
- id: my_new_entity
  display_name: My New Entity
  gliner_label: "my new entity natural description"
  enabled: true
  policy: [general]
  priority: 5
  masking:
    strategy: redact
    format: "[MY ENTITY {n}]"
```

2. Retrain on RunPod:
```bash
./start.sh --samples 2000
```

3. Upload new weights to S3 and restart inference pod.

---

## API

```bash
# Mask text
curl -X POST http://localhost:8000/mask \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Smith, DOB 07/15/1985, SSN 078-05-1120"}'

# Health check
curl http://localhost:8000/health
```

Response:
```json
{
  "masked_text": "Patient [PERSON 1], DOB [DATE 1], SSN [SSN 1]",
  "spans": [...],
  "stats": { "total_ms": 52, "ner_ms": 48, "pattern_ms": 4, "spans_total": 3 }
}
```

---

## Project Structure

```
entities_config.yaml     ← 105 entity definitions (single source of truth)
run.sh                   ← inference server (local + RunPod)
start.sh                 ← GPU training entry point (RunPod A100)
requirements.txt
.env.example             ← copy to .env, fill 7 keys

pipeline/
  ner_layer.py           ← GLiNER inference (CUDA: flat fp16 | CPU/MPS: parallel groups)
  pattern_layer.py       ← Presidio + regex (runs parallel to NER)
  orchestrator.py        ← asyncio.gather(pattern, ner) → merge → mask
  masking_engine.py      ← applies masking strategy from entities_config.yaml
  span_merger.py         ← deduplication, confidence-based conflict resolution

train/
  generate_data.py       ← synthetic training data generator (auto-covers new entities)
  finetune.py            ← GLiNER fine-tuning (A100/GPU/MPS/CPU auto-detected)

models/
  pii_gliner/            ← weights (gitignored, downloaded from S3 at startup)
    pytorch_model.bin    ← 1.7 GB fine-tuned weights
    gliner_config.json
    tokenizer.json
    tokenizer_config.json

strategies/
  masking_strategies.py  ← redact / substitute / hash / partial_redact / encrypt
static/                  ← web UI
app.py                   ← FastAPI application
config.py                ← configuration (only secrets read from env)
```

**Gitignored:** `.env` · `models/pii_gliner/` · `train/data/` · `.venv/`
