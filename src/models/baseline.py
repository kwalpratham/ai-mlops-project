"""Baseline text classifier: TF-IDF + Logistic Regression."""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import AG_NEWS_LABELS, MAX_TFIDF_FEATURES, TFIDF_NGRAM_RANGE, RANDOM_SEED

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """TF-IDF + Logistic Regression classifier."""

    def __init__(self):
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_TFIDF_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ])
        self.label_map: Dict[int, str] = dict(AG_NEWS_LABELS)
        self._is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, texts: List[str], labels: List[int]) -> None:
        logger.info("Training baseline model on %d samples …", len(texts))
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        logger.info("Baseline training complete.")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[int]:
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def predict_with_confidence(self, texts: List[str]) -> List[dict]:
        probs = self.predict_proba(texts)
        preds = np.argmax(probs, axis=1)
        results = []
        for pred, prob_row in zip(preds, probs):
            results.append({
                "predicted_label": self.label_map[int(pred)],
                "predicted_class": int(pred),
                "confidence_scores": {
                    self.label_map[i]: round(float(p), 4)
                    for i, p in enumerate(prob_row)
                },
            })
        return results

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "label_map": self.label_map},
            directory / "model.joblib",
        )
        logger.info("Baseline model saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "BaselineClassifier":
        directory = Path(directory)
        data = joblib.load(directory / "model.joblib")
        obj = cls.__new__(cls)
        obj.pipeline = data["pipeline"]
        obj.label_map = data["label_map"]
        obj._is_fitted = True
        logger.info("Baseline model loaded from %s", directory)
        return obj

    # ── Helpers for serving ───────────────────────────────────────────────

    @property
    def classes(self) -> List[str]:
        return [self.label_map[i] for i in sorted(self.label_map)]

    def get_params(self) -> dict:
        return {
            "model_type": "baseline",
            "max_tfidf_features": MAX_TFIDF_FEATURES,
            "ngram_range": str(TFIDF_NGRAM_RANGE),
            "C": self.pipeline.named_steps["clf"].C,
            "solver": self.pipeline.named_steps["clf"].solver,
        }
