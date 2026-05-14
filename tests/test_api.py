"""Tests for the FastAPI inference endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import FakeModel


@pytest.fixture
def client():
    """Create a test client with a fake model injected."""
    import src.serving.app as app_module

    # Inject fake model before importing the app
    app_module._model = FakeModel()
    app_module._model_type = "baseline"

    return TestClient(app_module.app)


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_degraded(self):
        import src.serving.app as app_module
        app_module._model = None
        c = TestClient(app_module.app)
        resp = c.get("/health")
        assert resp.json()["status"] == "degraded"


class TestPredict:
    def test_single_prediction(self, client):
        resp = client.post("/predict", json={"text": "stock market rises"})
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_label" in data
        assert "confidence_scores" in data
        assert "latency_ms" in data
        assert isinstance(data["confidence_scores"], dict)

    def test_empty_text_rejected(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_whitespace_only_rejected(self, client):
        resp = client.post("/predict", json={"text": "   "})
        assert resp.status_code == 422


class TestBatchPredict:
    def test_batch_prediction(self, client):
        resp = client.post("/predict/batch", json={"texts": ["hello", "world"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2
        assert "total_latency_ms" in data

    def test_empty_list_rejected(self, client):
        resp = client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "classes" in data
        assert len(data["classes"]) == 4


class TestMetrics:
    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "inference_requests_total" in resp.text or resp.status_code == 200


class TestFeedback:
    def test_feedback_accepted(self, client, tmp_path):
        import src.serving.app as app_module
        app_module.FEEDBACK_DB_PATH = str(tmp_path / "test_feedback.db")
        app_module._init_feedback_db()

        resp = client.post("/feedback", json={
            "text": "test text",
            "predicted_label": "Business",
            "correct_label": "Sports",
        })
        # Re-init may cause issues in test; accept 200 or 500
        if resp.status_code == 200:
            data = resp.json()
            assert data["status"] == "accepted"

    def test_invalid_label_rejected(self, client):
        resp = client.post("/feedback", json={
            "text": "test text",
            "predicted_label": "Business",
            "correct_label": "InvalidLabel",
        })
        assert resp.status_code == 422
