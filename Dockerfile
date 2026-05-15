# =============================================================================
# PII Model — Inference Container
#
# Runs the FastAPI inference server (GLiNER + Presidio pattern layer).
# Does NOT include training code — train on RunPod with ./start.sh, then
# copy models/pii_gliner/ here or mount it as a volume.
#
# Build:
#   docker build -t pii-model .
#
# Run with base GLiNER model (no fine-tuning):
#   docker run -p 8000:8000 pii-model
#
# Run with fine-tuned model (mount the trained model):
#   docker run -p 8000:8000 \
#     -v $(pwd)/models/pii_gliner:/app/models/pii_gliner \
#     -e FINE_TUNED_MODEL_PATH=/app/models/pii_gliner \
#     -e GLINER_THRESHOLD=0.55 \
#     pii-model
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# System deps for spaCy / tokenization
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python inference dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (pattern layer)
RUN python -m spacy download en_core_web_sm

# Pre-cache base GLiNER model so container starts instantly
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_large-v2.1'); print('GLiNER cached')"

# Copy application code
COPY . .

# Trained model directory — populated at runtime via volume mount or ./start.sh output
RUN mkdir -p models/pii_gliner

EXPOSE 8000

ENV PORT=8000
ENV HOST=0.0.0.0
ENV UVICORN_RELOAD=0

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
