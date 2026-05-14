"""Bonus A: LLM-based text classifier using few-shot prompting.

Supports two backends:
  1. HuggingFace Inference API (free tier, no GPU needed)
  2. Local Ollama instance (self-hosted)

Usage:
    classifier = LLMClassifier(backend="huggingface")
    result = classifier.classify("NASA launches new Mars rover")
"""

import logging
import os
import time
from typing import List, Optional

import requests

from src.config import AG_NEWS_LABELS

logger = logging.getLogger(__name__)

LABEL_NAMES = [AG_NEWS_LABELS[i] for i in sorted(AG_NEWS_LABELS)]

FEW_SHOT_PROMPT = """Classify the following news text into exactly one category.
Categories: World, Sports, Business, Sci/Tech

Examples:
Text: "Stocks rally as inflation data comes in lower than expected."
Category: Business

Text: "Team USA wins gold medal in the 100m sprint at the Olympics."
Category: Sports

Text: "Researchers develop a new quantum computing chip at IBM."
Category: Sci/Tech

Text: "UN Security Council holds emergency meeting on Middle East crisis."
Category: World

Text: "{text}"
Category:"""


class LLMClassifier:
    """Few-shot LLM classifier with multiple backend support."""

    def __init__(
        self,
        backend: str = "huggingface",
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_token: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "mistral",
    ):
        self.backend = backend
        self.model_id = model_id
        self.api_token = api_token or os.getenv("HF_API_TOKEN", "")
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.label_map = dict(AG_NEWS_LABELS)
        self._valid_labels = set(LABEL_NAMES)

    def _parse_label(self, raw_output: str) -> str:
        """Extract a valid label from LLM output."""
        raw = raw_output.strip().split("\n")[0].strip().rstrip(".")
        # Direct match
        if raw in self._valid_labels:
            return raw
        # Fuzzy match
        raw_lower = raw.lower()
        for label in self._valid_labels:
            if label.lower() in raw_lower:
                return label
        return "World"  # fallback

    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model_id}",
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 10, "temperature": 0.1},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "").replace(prompt, "").strip()
        return str(data)

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance."""
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 10},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def classify(self, text: str) -> dict:
        """Classify a single text using the LLM."""
        prompt = FEW_SHOT_PROMPT.format(text=text[:500])
        start = time.perf_counter()

        try:
            if self.backend == "ollama":
                raw = self._call_ollama(prompt)
            else:
                raw = self._call_huggingface(prompt)
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return {
                "predicted_label": "World",
                "confidence_scores": {lbl: 0.25 for lbl in LABEL_NAMES},
                "latency_ms": (time.perf_counter() - start) * 1000,
                "error": str(e),
            }

        label = self._parse_label(raw)
        latency_ms = (time.perf_counter() - start) * 1000

        # LLMs don't give calibrated probabilities; we return a pseudo-distribution
        confidence = {lbl: 0.05 for lbl in LABEL_NAMES}
        confidence[label] = 0.85

        return {
            "predicted_label": label,
            "confidence_scores": confidence,
            "latency_ms": round(latency_ms, 2),
            "raw_output": raw[:100],
        }

    def classify_batch(self, texts: List[str]) -> List[dict]:
        """Classify multiple texts (sequential — LLM calls are slow)."""
        return [self.classify(t) for t in texts]


def compare_models(
    trained_model,
    llm_classifier: LLMClassifier,
    texts: List[str],
    true_labels: List[int],
) -> dict:
    """Compare a fine-tuned model vs. LLM on the same inputs.

    Returns a summary dict with accuracy and latency comparisons.
    """
    # Fine-tuned model
    start = time.perf_counter()
    ft_preds = trained_model.predict(texts)
    ft_latency = (time.perf_counter() - start) * 1000

    ft_correct = sum(1 for t, p in zip(true_labels, ft_preds) if t == p)
    ft_accuracy = ft_correct / len(true_labels)

    # LLM
    label_to_id = {v: k for k, v in AG_NEWS_LABELS.items()}
    start = time.perf_counter()
    llm_results = llm_classifier.classify_batch(texts)
    llm_latency = (time.perf_counter() - start) * 1000

    llm_preds = [label_to_id.get(r["predicted_label"], 0) for r in llm_results]
    llm_correct = sum(1 for t, p in zip(true_labels, llm_preds) if t == p)
    llm_accuracy = llm_correct / len(true_labels)

    comparison = {
        "fine_tuned": {
            "accuracy": round(ft_accuracy, 4),
            "total_latency_ms": round(ft_latency, 2),
            "avg_latency_ms": round(ft_latency / len(texts), 2),
        },
        "llm": {
            "accuracy": round(llm_accuracy, 4),
            "total_latency_ms": round(llm_latency, 2),
            "avg_latency_ms": round(llm_latency / len(texts), 2),
            "backend": llm_classifier.backend,
            "model": llm_classifier.model_id if llm_classifier.backend == "huggingface" else llm_classifier.ollama_model,
        },
        "recommendation": (
            "Use the fine-tuned model for production: it offers 10-100x lower latency, "
            "calibrated confidence scores, and deterministic behavior. The LLM approach "
            "is better suited for prototyping, zero-shot scenarios with novel categories, "
            "or as a fallback when no labeled data is available for fine-tuning. In a "
            "production setting, the LLM can serve as a teacher model to generate "
            "training labels for new categories, which are then distilled into a "
            "lightweight fine-tuned classifier."
        ),
    }

    print("\n── Model Comparison ──")
    print(f"  Fine-tuned:  accuracy={ft_accuracy:.2%}  latency={ft_latency:.0f}ms total")
    print(f"  LLM:         accuracy={llm_accuracy:.2%}  latency={llm_latency:.0f}ms total")
    print(f"  → {comparison['recommendation'][:120]}…")

    return comparison
