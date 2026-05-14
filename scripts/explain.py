#!/usr/bin/env python3
"""Run LIME explainability on a subset of test samples."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import BASELINE_MODEL_DIR  # noqa: E402
from src.data.pipeline import load_and_prepare_data  # noqa: E402
from src.explainability.explain import explain_predictions, generate_failure_analysis  # noqa: E402
from src.models.baseline import BaselineClassifier  # noqa: E402


def main():
    _, _, test_df = load_and_prepare_data()
    model = BaselineClassifier.load(BASELINE_MODEL_DIR)

    # Explain 8 samples (2 per class)
    sample = test_df.groupby("label").head(2).head(8)
    explain_predictions(
        model=model,
        texts=sample["text"].tolist(),
        true_labels=sample["label"].tolist(),
        model_type="baseline",
        num_features=10,
        num_samples=300,
    )

    # Failure analysis on full test set
    generate_failure_analysis(
        model=model,
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist(),
        model_type="baseline",
    )
    print("\nExplainability complete!")


if __name__ == "__main__":
    main()
