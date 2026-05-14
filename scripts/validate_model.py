#!/usr/bin/env python3
"""Model validation script for CI pipeline.

Loads the trained model, runs inference on fixture inputs,
and asserts quality thresholds are met.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import f1_score  # noqa: E402

from src.config import AG_NEWS_LABELS, BASELINE_MODEL_DIR, MIN_F1_THRESHOLD, MODEL_TYPE, TRANSFORMER_MODEL_DIR  # noqa: E402

FIXTURES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "test_inputs.json"
LABEL_TO_ID = {v: k for k, v in AG_NEWS_LABELS.items()}


def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else MODEL_TYPE

    # Load model
    if model_type == "transformer":
        from src.models.transformer import TransformerClassifier
        model = TransformerClassifier.load(TRANSFORMER_MODEL_DIR)
    else:
        from src.models.baseline import BaselineClassifier
        model = BaselineClassifier.load(BASELINE_MODEL_DIR)

    # Load fixtures
    fixtures = json.loads(FIXTURES_PATH.read_text())
    texts = [f["text"] for f in fixtures]
    expected = [f["expected_label"] for f in fixtures]

    # Predict
    preds = model.predict(texts)
    pred_labels = [AG_NEWS_LABELS[p] for p in preds]

    # Evaluate — compute macro-F1 (the assignment-mandated metric)
    correct = sum(1 for e, p in zip(expected, pred_labels) if e == p)
    accuracy = correct / len(expected)
    macro_f1 = f1_score(expected, pred_labels, average="macro")

    print(f"Model: {model_type}")
    print(f"Fixture accuracy: {accuracy:.2%} ({correct}/{len(expected)})")
    print(f"Fixture macro-F1: {macro_f1:.4f}")
    for i, (exp, pred) in enumerate(zip(expected, pred_labels)):
        status = "✓" if exp == pred else "✗"
        print(f"  {status} [{exp:10s}] predicted [{pred:10s}] — {texts[i][:60]}…")

    if macro_f1 < MIN_F1_THRESHOLD:
        print(f"\n✗ FAILED: macro-F1 {macro_f1:.4f} < threshold {MIN_F1_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"\n✓ PASSED: macro-F1 {macro_f1:.4f} >= threshold {MIN_F1_THRESHOLD}")
        sys.exit(0)


if __name__ == "__main__":
    main()
