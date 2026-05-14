"""FastAPI inference server for text classification."""

import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src.config import (
    BASELINE_MODEL_DIR,
    FEEDBACK_DB_PATH,
    LOG_LEVEL,
    MODEL_PATH,
    MODEL_TYPE,
    TRANSFORMER_MODEL_DIR,
)
from src.serving.metrics import (
    ERROR_COUNT,
    MODEL_INFO,
    PREDICTION_LABELS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    get_prometheus_metrics,
    log_error,
    log_prediction,
)
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
)

logger = logging.getLogger(__name__)

# ── Global model reference ────────────────────────────────────────────────────
_model = None
_model_type: str = MODEL_TYPE


def _resolve_model_path() -> Path:
    """Determine which model directory to load."""
    if MODEL_PATH:
        return Path(MODEL_PATH)
    if _model_type == "transformer":
        return TRANSFORMER_MODEL_DIR
    return BASELINE_MODEL_DIR


def _load_model():
    """Load the configured model into memory."""
    global _model, _model_type
    model_path = _resolve_model_path()
    logger.info("Loading %s model from %s …", _model_type, model_path)

    if _model_type == "transformer":
        from src.models.transformer import TransformerClassifier
        _model = TransformerClassifier.load(model_path)
    else:
        from src.models.baseline import BaselineClassifier
        _model = BaselineClassifier.load(model_path)

    MODEL_INFO.info({
        "model_type": _model_type,
        "model_path": str(model_path),
    })
    logger.info("Model loaded successfully.")


def _init_feedback_db():
    """Create the feedback SQLite table if it doesn't exist."""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT NOT NULL,
            predicted   TEXT NOT NULL,
            corrected   TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        _load_model()
    except Exception as e:
        logger.error("Failed to load model: %s", e)
    _init_feedback_db()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Text Classification API",
    description="Real-time customer support ticket classifier",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_type=_model_type if _model else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type=_model_type,
        classes=_model.classes,
        parameters={k: str(v) for k, v in _model.get_params().items()},
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        result = _model.predict_with_confidence([req.text])[0]
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        log_error("/predict", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    # Metrics
    REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000)
    PREDICTION_LABELS.labels(label=result["predicted_label"]).inc()

    log_prediction(
        req.text,
        result["predicted_label"],
        max(result["confidence_scores"].values()),
        latency_ms,
    )

    return PredictResponse(
        predicted_label=result["predicted_label"],
        confidence_scores=result["confidence_scores"],
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_batch(req: BatchPredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        results = _model.predict_with_confidence(req.texts)
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict/batch", error_type=type(e).__name__).inc()
        log_error("/predict/batch", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    total_latency_ms = (time.perf_counter() - start) * 1000
    per_item_latency = total_latency_ms / len(req.texts)

    predictions = []
    for r, text in zip(results, req.texts):
        PREDICTION_LABELS.labels(label=r["predicted_label"]).inc()
        log_prediction(text, r["predicted_label"], max(r["confidence_scores"].values()), per_item_latency, "/predict/batch")
        predictions.append(PredictionResult(
            predicted_label=r["predicted_label"],
            confidence_scores=r["confidence_scores"],
            latency_ms=round(per_item_latency, 2),
        ))

    REQUEST_COUNT.labels(endpoint="/predict/batch", status="success").inc()
    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(total_latency_ms / 1000)

    return BatchPredictResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency_ms, 2),
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(
        content=get_prometheus_metrics().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ── Bonus C: Feedback endpoint ───────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """Accept a label correction and store it for future retraining."""
    try:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        cursor = conn.execute(
            "INSERT INTO feedback (text, predicted, corrected, created_at) VALUES (?, ?, ?, ?)",
            (req.text, req.predicted_label, req.correct_label, datetime.now(timezone.utc).isoformat()),
        )
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {e}")

    logger.info(
        "Feedback #%d stored: predicted=%s  correct=%s",
        feedback_id,
        req.predicted_label,
        req.correct_label,
    )

    return FeedbackResponse(
        status="accepted",
        message=(
            "Feedback recorded. It will be included in the next retraining batch. "
            "See docs/drift_detection.md for the full flywheel architecture."
        ),
        feedback_id=feedback_id,
    )
