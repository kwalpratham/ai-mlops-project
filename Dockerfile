# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY fixtures/ fixtures/

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --from=builder /build/src ./src
COPY --from=builder /build/scripts ./scripts
COPY --from=builder /build/fixtures ./fixtures

# Models are mounted at runtime or baked in during CI
# COPY models/ ./models/

# Environment variables (configurable at deploy time)
ENV MODEL_TYPE=baseline \
    MODEL_PATH=/app/models/baseline \
    PORT=8000 \
    LOG_LEVEL=INFO \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    FEEDBACK_DB_PATH=/app/data/feedback.db \
    PYTHONUNBUFFERED=1

EXPOSE ${PORT}

RUN mkdir -p /app/models /app/data && chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
