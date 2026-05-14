import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ── Data ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
DATASET_NAME = "ag_news"
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
NUM_CLASSES = len(AG_NEWS_LABELS)
VALIDATION_SPLIT = 0.1  # fraction of train set used for validation

# ── Baseline model ────────────────────────────────────────────────────────────
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
MAX_TFIDF_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)

# ── Transformer model ────────────────────────────────────────────────────────
TRANSFORMER_MODEL_DIR = MODELS_DIR / "transformer"
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_EPOCHS = int(os.getenv("TRANSFORMER_EPOCHS", "3"))
TRANSFORMER_BATCH_SIZE = int(os.getenv("TRANSFORMER_BATCH_SIZE", "16"))
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_WARMUP_RATIO = 0.1
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_MAX_TRAIN_SAMPLES = int(os.getenv("TRANSFORMER_MAX_TRAIN_SAMPLES", "0"))  # 0=use all

# ── Serving ───────────────────────────────────────────────────────────────────
MODEL_TYPE = os.getenv("MODEL_TYPE", "baseline")
MODEL_PATH = os.getenv("MODEL_PATH", "")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "text-classification"

# ── Feedback (Bonus C) ───────────────────────────────────────────────────────
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", str(PROJECT_ROOT / "feedback.db"))

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_F1_THRESHOLD = 0.85
