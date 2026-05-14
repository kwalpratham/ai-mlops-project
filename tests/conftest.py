"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeModel:
    """Lightweight fake model for API testing (no real ML dependencies)."""

    def __init__(self):
        self.label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    def predict(self, texts):
        return [2] * len(texts)  # always "Business"

    def predict_proba(self, texts):
        import numpy as np
        n = len(texts)
        probs = np.array([[0.05, 0.05, 0.85, 0.05]] * n)
        return probs

    def predict_with_confidence(self, texts):
        results = []
        for _ in texts:
            results.append({
                "predicted_label": "Business",
                "predicted_class": 2,
                "confidence_scores": {
                    "World": 0.05,
                    "Sports": 0.05,
                    "Business": 0.85,
                    "Sci/Tech": 0.05,
                },
            })
        return results

    @property
    def classes(self):
        return ["World", "Sports", "Business", "Sci/Tech"]

    def get_params(self):
        return {"model_type": "fake", "version": "test"}


@pytest.fixture
def fake_model():
    return FakeModel()
