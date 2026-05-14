"""Training orchestration with MLflow experiment tracking."""

import json
import logging
from pathlib import Path
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import (
    AG_NEWS_LABELS,
    ARTIFACTS_DIR,
    BASELINE_MODEL_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TRANSFORMER_MAX_TRAIN_SAMPLES,
    TRANSFORMER_MODEL_DIR,
)
from src.data.pipeline import load_and_prepare_data, run_eda
from src.models.baseline import BaselineClassifier
from src.models.transformer import TransformerClassifier

logger = logging.getLogger(__name__)

LABEL_NAMES = [AG_NEWS_LABELS[i] for i in sorted(AG_NEWS_LABELS)]


def _evaluate_model(
    y_true: list, y_pred: list, label_names: list
) -> dict:
    """Compute all evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(label_names)))
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names
        ),
    }


def _log_to_mlflow(
    params: dict,
    metrics: dict,
    model_type: str,
    model_dir: Path,
) -> str:
    """Log a training run to MLflow. Returns the run ID."""
    with mlflow.start_run(run_name=f"{model_type}-run") as run:
        # Parameters
        mlflow.log_params(params)

        # Scalar metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        for cls_name, cls_metrics in metrics["per_class"].items():
            safe = cls_name.replace("/", "_")
            mlflow.log_metric(f"precision_{safe}", cls_metrics["precision"])
            mlflow.log_metric(f"recall_{safe}", cls_metrics["recall"])
            mlflow.log_metric(f"f1_{safe}", cls_metrics["f1"])

        # Artifacts
        artifacts_path = ARTIFACTS_DIR / model_type
        artifacts_path.mkdir(parents=True, exist_ok=True)

        # Save confusion matrix
        cm_path = artifacts_path / "confusion_matrix.json"
        cm_path.write_text(json.dumps(metrics["confusion_matrix"]))
        mlflow.log_artifact(str(cm_path))

        # Save classification report
        report_path = artifacts_path / "classification_report.txt"
        report_path.write_text(metrics["classification_report"])
        mlflow.log_artifact(str(report_path))

        # Save per-class metrics
        per_class_path = artifacts_path / "per_class_metrics.json"
        per_class_path.write_text(json.dumps(metrics["per_class"], indent=2))
        mlflow.log_artifact(str(per_class_path))

        # Log model directory as artifacts + register as an MLflow model
        if model_dir.exists():
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            # Log a pyfunc model entry so mlflow.register_model() can find it
            mlflow.pyfunc.log_model(
                artifact_path="registered_model",
                python_model=mlflow.pyfunc.PythonModel(),
                artifacts={"model_dir": str(model_dir)},
            )

        run_id = run.info.run_id
        logger.info("MLflow run %s logged for %s", run_id, model_type)
        return run_id


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[dict, str]:
    """Train baseline model, evaluate, and log to MLflow."""
    logger.info("── Training Baseline (TF-IDF + LogReg) ──")

    model = BaselineClassifier()
    model.train(train_df["text"].tolist(), train_df["label"].tolist())
    model.save(BASELINE_MODEL_DIR)

    # Evaluate on test set
    y_pred = model.predict(test_df["text"].tolist())
    metrics = _evaluate_model(test_df["label"].tolist(), y_pred, LABEL_NAMES)

    print(f"\n  Baseline Test Accuracy:  {metrics['accuracy']}")
    print(f"  Baseline Test Macro-F1: {metrics['macro_f1']}")
    print(f"\n{metrics['classification_report']}")

    # Log to MLflow
    run_id = _log_to_mlflow(model.get_params(), metrics, "baseline", BASELINE_MODEL_DIR)

    return metrics, run_id


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[dict, str]:
    """Train transformer model, evaluate, and log to MLflow."""
    logger.info("── Training Transformer (DistilBERT) ──")

    # Optionally subsample for CPU-only training
    if TRANSFORMER_MAX_TRAIN_SAMPLES > 0 and len(train_df) > TRANSFORMER_MAX_TRAIN_SAMPLES:
        logger.info(f"Subsampling training data: {len(train_df)} -> {TRANSFORMER_MAX_TRAIN_SAMPLES}")
        train_df = train_df.sample(n=TRANSFORMER_MAX_TRAIN_SAMPLES, random_state=42)
        val_df = val_df.sample(n=min(len(val_df), TRANSFORMER_MAX_TRAIN_SAMPLES // 5), random_state=42)

    model = TransformerClassifier()
    model.train(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        val_texts=val_df["text"].tolist(),
        val_labels=val_df["label"].tolist(),
        output_dir=str(TRANSFORMER_MODEL_DIR / "checkpoints"),
    )
    model.save(TRANSFORMER_MODEL_DIR)

    # Evaluate on test set
    y_pred = model.predict(test_df["text"].tolist())
    metrics = _evaluate_model(test_df["label"].tolist(), y_pred, LABEL_NAMES)

    print(f"\n  Transformer Test Accuracy:  {metrics['accuracy']}")
    print(f"  Transformer Test Macro-F1: {metrics['macro_f1']}")
    print(f"\n{metrics['classification_report']}")

    # Log to MLflow
    run_id = _log_to_mlflow(model.get_params(), metrics, "transformer", TRANSFORMER_MODEL_DIR)

    return metrics, run_id


def register_champion(run_id: str, model_name: str = "text-classifier") -> None:
    """Register the best model in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/registered_model"
    result = mlflow.register_model(model_uri, model_name)
    logger.info(
        "Registered model '%s' version %s as champion.",
        result.name,
        result.version,
    )


def run_full_training(model_type: str = "all") -> dict:
    """End-to-end training pipeline.

    Args:
        model_type: 'baseline', 'transformer', or 'all'.
    """
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load data
    train_df, val_df, test_df = load_and_prepare_data()
    run_eda(train_df, val_df, test_df)

    results = {}

    if model_type in ("baseline", "all"):
        metrics, run_id = train_baseline(train_df, val_df, test_df)
        results["baseline"] = {"metrics": metrics, "run_id": run_id}

    if model_type in ("transformer", "all"):
        metrics, run_id = train_transformer(train_df, val_df, test_df)
        results["transformer"] = {"metrics": metrics, "run_id": run_id}

    # Decide champion
    if model_type == "all" and len(results) == 2:
        b_f1 = results["baseline"]["metrics"]["macro_f1"]
        t_f1 = results["transformer"]["metrics"]["macro_f1"]
        champion = "transformer" if t_f1 >= b_f1 else "baseline"
        print(f"\n{'=' * 48}")
        print(f"  CHAMPION MODEL: {champion} (F1={results[champion]['metrics']['macro_f1']})")
        print(f"{'=' * 48}\n")
        results["champion"] = champion
        try:
            register_champion(results[champion]["run_id"])
        except Exception as e:
            logger.warning("Model registry failed (MLflow server may be unavailable): %s", e)
    elif len(results) == 1:
        only = list(results.keys())[0]
        results["champion"] = only
        try:
            register_champion(results[only]["run_id"])
        except Exception as e:
            logger.warning("Model registry failed (MLflow server may be unavailable): %s", e)

    return results
