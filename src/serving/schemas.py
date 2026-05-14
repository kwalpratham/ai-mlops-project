"""Pydantic request/response schemas for the inference API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Requests ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64, description="Batch of texts")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        cleaned = []
        for i, t in enumerate(v):
            if not t or not t.strip():
                raise ValueError(f"texts[{i}] must be non-empty")
            if len(t) > 5000:
                raise ValueError(f"texts[{i}] exceeds 5000 character limit")
            cleaned.append(t.strip())
        return cleaned


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    predicted_label: str
    correct_label: str

    @field_validator("correct_label", "predicted_label")
    @classmethod
    def valid_label(cls, v: str) -> str:
        allowed = {"World", "Sports", "Business", "Sci/Tech"}
        if v not in allowed:
            raise ValueError(f"label must be one of {allowed}")
        return v


# ── Responses ─────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    predicted_label: str
    confidence_scores: Dict[str, float]
    latency_ms: float


class PredictResponse(BaseModel):
    predicted_label: str
    confidence_scores: Dict[str, float]
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResult]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str = "1.0.0"
    classes: List[str]
    parameters: Dict[str, str] = {}


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: int


class ErrorResponse(BaseModel):
    detail: str
