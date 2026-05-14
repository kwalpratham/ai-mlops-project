"""Tests for model validation on fixture inputs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import FakeModel


FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "test_inputs.json"


class TestModelValidation:
    def test_fixtures_file_exists(self):
        assert FIXTURES_PATH.exists(), "fixtures/test_inputs.json not found"

    def test_fixtures_structure(self):
        fixtures = json.loads(FIXTURES_PATH.read_text())
        assert len(fixtures) == 10
        for item in fixtures:
            assert "text" in item
            assert "expected_label" in item
            assert item["expected_label"] in {"World", "Sports", "Business", "Sci/Tech"}

    def test_model_returns_valid_labels(self):
        model = FakeModel()
        fixtures = json.loads(FIXTURES_PATH.read_text())
        texts = [f["text"] for f in fixtures]
        preds = model.predict(texts)
        valid_labels = {0, 1, 2, 3}
        for p in preds:
            assert p in valid_labels

    def test_model_confidence_sums_to_one(self):
        model = FakeModel()
        probs = model.predict_proba(["test text"])
        total = sum(probs[0])
        assert abs(total - 1.0) < 0.01
