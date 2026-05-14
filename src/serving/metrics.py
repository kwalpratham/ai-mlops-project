"""Prometheus metrics and structured JSON logging for the inference API."""

import json
import logging

from prometheus_client import Counter, Histogram, Info, generate_latest

logger = logging.getLogger("inference")

# ── Prometheus Metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_LABELS = Counter(
    "inference_prediction_labels_total",
    "Prediction label distribution",
    ["label"],
)

ERROR_COUNT = Counter(
    "inference_errors_total",
    "Total inference errors",
    ["endpoint", "error_type"],
)

MODEL_INFO = Info(
    "inference_model",
    "Currently loaded model information",
)


# ── Structured Logging ────────────────────────────────────────────────────────

def log_prediction(
    text_snippet: str,
    predicted_label: str,
    confidence: float,
    latency_ms: float,
    endpoint: str = "/predict",
) -> None:
    """Emit a structured JSON log line for each prediction."""
    logger.info(
        json.dumps({
            "event": "prediction",
            "endpoint": endpoint,
            "text_snippet": text_snippet[:80],
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
        })
    )


def log_error(
    endpoint: str,
    error_type: str,
    detail: str,
) -> None:
    """Emit a structured JSON log for errors."""
    logger.error(
        json.dumps({
            "event": "error",
            "endpoint": endpoint,
            "error_type": error_type,
            "detail": detail[:200],
        })
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def get_prometheus_metrics() -> bytes:
    """Return Prometheus metrics as bytes."""
    return generate_latest()
