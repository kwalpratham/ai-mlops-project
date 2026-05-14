"""Transformer text classifier: fine-tuned DistilBERT."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import (
    AG_NEWS_LABELS,
    NUM_CLASSES,
    RANDOM_SEED,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_LEARNING_RATE,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_MODEL_NAME,
    TRANSFORMER_WARMUP_RATIO,
    TRANSFORMER_WEIGHT_DECAY,
)

logger = logging.getLogger(__name__)


# ── Dataset wrapper for HuggingFace Trainer ───────────────────────────────────

class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerClassifier:
    """DistilBERT-based text classifier."""

    def __init__(self, model_name: str = TRANSFORMER_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_CLASSES,
        )
        self.label_map: Dict[int, str] = dict(AG_NEWS_LABELS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_fitted = False

    # ── Tokenization ──────────────────────────────────────────────────────

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=TRANSFORMER_MAX_LENGTH,
            return_tensors="pt",
        )

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        output_dir: str | Path = "./transformer_training",
    ) -> dict:
        logger.info(
            "Fine-tuning %s on %d samples …", self.model_name, len(train_texts)
        )

        train_enc = self.tokenizer(
            train_texts,
            padding=True,
            truncation=True,
            max_length=TRANSFORMER_MAX_LENGTH,
        )
        val_enc = self.tokenizer(
            val_texts,
            padding=True,
            truncation=True,
            max_length=TRANSFORMER_MAX_LENGTH,
        )

        train_dataset = _TextDataset(train_enc, train_labels)
        val_dataset = _TextDataset(val_enc, val_labels)

        # Compute warmup_steps from ratio (warmup_ratio deprecated in transformers >=5.2)
        steps_per_epoch = len(train_dataset) // TRANSFORMER_BATCH_SIZE
        total_steps = steps_per_epoch * TRANSFORMER_EPOCHS
        warmup_steps = int(total_steps * TRANSFORMER_WARMUP_RATIO)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=TRANSFORMER_EPOCHS,
            per_device_train_batch_size=TRANSFORMER_BATCH_SIZE,
            per_device_eval_batch_size=TRANSFORMER_BATCH_SIZE * 2,
            learning_rate=TRANSFORMER_LEARNING_RATE,
            warmup_steps=warmup_steps,
            weight_decay=TRANSFORMER_WEIGHT_DECAY,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=100,
            seed=RANDOM_SEED,
            fp16=torch.cuda.is_available(),
            report_to="none",  # we log to MLflow ourselves
        )

        from sklearn.metrics import accuracy_score, f1_score

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "macro_f1": f1_score(labels, preds, average="macro"),
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.model = trainer.model
        self._is_fitted = True
        eval_results = trainer.evaluate()
        logger.info("Transformer training complete — eval loss: %.4f", eval_results["eval_loss"])
        return eval_results

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[int]:
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        encodings = self._tokenize(texts)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
        logits = outputs.logits.cpu().numpy()
        # softmax
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

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
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        logger.info("Transformer model saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> "TransformerClassifier":
        directory = Path(directory)
        obj = cls.__new__(cls)
        obj.model_name = TRANSFORMER_MODEL_NAME
        obj.tokenizer = AutoTokenizer.from_pretrained(directory)
        obj.model = AutoModelForSequenceClassification.from_pretrained(directory)
        obj.label_map = dict(AG_NEWS_LABELS)
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.model.to(obj.device)
        obj._is_fitted = True
        logger.info("Transformer model loaded from %s", directory)
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────

    @property
    def classes(self) -> List[str]:
        return [self.label_map[i] for i in sorted(self.label_map)]

    def get_params(self) -> dict:
        return {
            "model_type": "transformer",
            "base_model": self.model_name,
            "epochs": TRANSFORMER_EPOCHS,
            "batch_size": TRANSFORMER_BATCH_SIZE,
            "learning_rate": TRANSFORMER_LEARNING_RATE,
            "max_length": TRANSFORMER_MAX_LENGTH,
            "warmup_ratio": TRANSFORMER_WARMUP_RATIO,
            "weight_decay": TRANSFORMER_WEIGHT_DECAY,
        }
