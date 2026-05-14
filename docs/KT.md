# Knowledge Transfer — Text Classification Service

> A plain-English walkthrough of the entire project for new developers.

---

## 1. What Does This Project Do?

We built a **text classification API** that takes a piece of news text and tells you which category it belongs to:

| Label ID | Category |
|----------|----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

Think of it as auto-routing: a customer sends a support ticket → the API instantly labels it so it goes to the right team.

We trained **two models**, wrapped them in a **REST API**, and added **Docker**, **CI/CD**, **monitoring**, and **explainability** around them.

---

## 2. Project Map

```
ai-mlops-project/
│
├── src/                          ← All source code lives here
│   ├── config.py                 ← Every setting in one place
│   ├── data/
│   │   └── pipeline.py           ← Load + clean + split the dataset
│   ├── models/
│   │   ├── baseline.py           ← Simple model (TF-IDF + Logistic Regression)
│   │   ├── transformer.py        ← Smart model (DistilBERT)
│   │   └── llm_classifier.py     ← Bonus: LLM few-shot classifier
│   ├── training/
│   │   └── train.py              ← Trains models + logs to MLflow
│   ├── explainability/
│   │   └── explain.py            ← LIME explanations + failure analysis
│   └── serving/
│       ├── app.py                ← The FastAPI server (all endpoints)
│       ├── schemas.py            ← Request/response shapes (Pydantic)
│       └── metrics.py            ← Prometheus counters + JSON logging
│
├── scripts/                      ← CLI entry-points you run directly
│   ├── train.py                  ← `python scripts/train.py --model baseline`
│   ├── validate_model.py         ← CI gate: checks F1 on 10 fixtures
│   └── explain.py                ← Runs LIME explainability
│
├── tests/                        ← pytest tests (20 tests)
├── fixtures/test_inputs.json     ← 10 curated inputs for CI validation
├── docs/
│   ├── drift_detection.md        ← How we'd handle data drift
│   └── KT.md                     ← This file
│
├── Dockerfile                    ← Multi-stage Docker image for the API
├── Dockerfile.mlflow             ← MLflow tracking server image
├── docker-compose.yml            ← One command to start everything
├── .github/workflows/ci.yml      ← GitHub Actions CI/CD
├── terraform/                    ← AWS infrastructure-as-code (ECS Fargate)
├── requirements.txt              ← Production dependencies
└── requirements-dev.txt          ← + test/lint tools (pytest, ruff)
```

---

## 3. How the Data Flows (End to End)

```
                     TRAINING (offline, run once)
                     ============================

 ┌───────────┐      ┌──────────┐      ┌─────────────┐      ┌─────────┐
 │  AG News  │─────▶│  Clean   │─────▶│  Train Two  │─────▶│  Save   │
 │  Dataset  │      │  & Split │      │  Models     │      │  Models │
 │(HuggingFace)     │          │      │             │      │  + Log  │
 └───────────┘      └──────────┘      └─────────────┘      └────┬────┘
                                                                │
                    120k train (→ 108k/12k)                     │
                    7.6k test                                   ▼
                                                          ┌──────────┐
                                                          │  MLflow  │
                                                          │  (params,│
                                                          │  metrics,│
                                                          │  artifacts)
                                                          └──────────┘

                     SERVING (online, always running)
                     ================================

              ┌──────────────────────────────────┐
  User ──────▶│  FastAPI Server (port 8000)      │
  (HTTP)      │                                  │
              │  POST /predict     → one text    │──▶ Model.predict()
              │  POST /predict/batch → many texts│──▶ Model.predict()
              │  GET  /health      → is it alive?│
              │  GET  /model/info  → what model? │
              │  GET  /metrics     → Prometheus  │
              │  POST /feedback    → corrections │──▶ SQLite
              └──────────────────────────────────┘
```

---

## 4. The Two Models

### Model A: Baseline (TF-IDF + Logistic Regression)

```
"Stock market rallies"  →  TF-IDF vectorizer  →  Logistic Regression  →  "Business"
                           (50k features,          (multinomial,
                            unigrams+bigrams)       sklearn)
```

- **Train time**: ~30 seconds
- **Inference**: <5ms per request
- **Accuracy**: ~92%
- **When to use**: When you need speed and simplicity

### Model B: Transformer (DistilBERT)

```
"Stock market rallies"  →  DistilBERT tokenizer  →  DistilBERT encoder  →  Classification head  →  "Business"
                           (subword tokens,           (6 transformer         (2 linear layers)
                            max 128 tokens)            layers, 66M params)
```

- **Train time**: ~45 min (GPU) / hours (CPU)
- **Inference**: ~50ms per request (CPU)
- **Accuracy**: ~94% (full data) / ~89% (5k samples)
- **When to use**: When accuracy matters more than latency

---

## 5. How to Run Things

### Train a model

```bash
cd /opt/test/ai-mlops-project
source .venv/bin/activate

# Train just the baseline (fast, ~30 seconds)
MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db python scripts/train.py --model baseline

# Train just the transformer (slow on CPU — use subsampling)
TRANSFORMER_MAX_TRAIN_SAMPLES=5000 TRANSFORMER_EPOCHS=1 \
  MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db python scripts/train.py --model transformer

# Train both
MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db python scripts/train.py --model all
```

### Start the API server

```bash
MODEL_TYPE=baseline uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Single prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launches new Mars rover"}' | python -m json.tool

# Batch prediction
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Stock market rises", "Lakers win game"]}' | python -m json.tool

# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics
```

### Run tests

```bash
pytest tests/ -v          # 20 tests, should all pass
ruff check src/ tests/    # Lint check, should show 0 errors
```

### Validate model for CI

```bash
python scripts/validate_model.py baseline      # Should print "PASSED: macro-F1 >= 0.85"
python scripts/validate_model.py transformer    # Same check for transformer
```

### Run explainability

```bash
python scripts/explain.py    # Generates LIME HTMLs + failure analysis
```

### Docker (full stack)

```bash
docker compose up --build    # Starts API on :8000 + MLflow on :5000
```

---

## 6. Configuration

**Everything is controlled via environment variables.** No hardcoded values.

| Variable | Default | What it does |
|----------|---------|-------------|
| `MODEL_TYPE` | `baseline` | Which model the API loads (`baseline` or `transformer`) |
| `MODEL_PATH` | `""` | Override model directory path |
| `PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Where MLflow logs go |
| `TRANSFORMER_EPOCHS` | `3` | How many training epochs |
| `TRANSFORMER_BATCH_SIZE` | `16` | Training batch size |
| `TRANSFORMER_MAX_TRAIN_SAMPLES` | `0` (=all) | Subsample training data (for CPU) |
| `FEEDBACK_DB_PATH` | `feedback.db` | Where feedback corrections are stored |

---

## 7. API Endpoints at a Glance

| Method | Path | Input | Output | Purpose |
|--------|------|-------|--------|---------|
| POST | `/predict` | `{"text": "..."}` | Label + confidence scores + latency | Classify one text |
| POST | `/predict/batch` | `{"texts": ["...", "..."]}` | Array of predictions + total latency | Classify many texts |
| GET | `/health` | — | `{"status": "healthy", "model_loaded": true}` | Liveness check |
| GET | `/model/info` | — | Model type, version, classes, parameters | Model metadata |
| GET | `/metrics` | — | Prometheus text format | Monitoring metrics |
| POST | `/feedback` | `{"text": "...", "predicted_label": "...", "correct_label": "..."}` | `{"status": "accepted", "feedback_id": 1}` | Submit corrections |

### Example response from `/predict`:

```json
{
  "predicted_label": "Sci/Tech",
  "confidence_scores": {
    "World": 0.02,
    "Sports": 0.01,
    "Business": 0.05,
    "Sci/Tech": 0.92
  },
  "latency_ms": 3.42
}
```

---

## 8. How the CI/CD Pipeline Works

```
  Developer pushes code or opens PR
              │
              ▼
  ┌─────────────────────┐
  │  1. LINT (ruff)     │  Catches style issues instantly
  └──────────┬──────────┘
             │ passes
             ▼
  ┌─────────────────────┐
  │  2. TEST (pytest)   │  Runs 20 unit + integration tests
  └──────────┬──────────┘
             │ passes
             ▼
  ┌─────────────────────┐
  │  3. MODEL VALIDATE  │  Loads model → runs 10 fixture inputs
  │     (F1 > 0.85)     │  → asserts macro-F1 above threshold
  └──────────┬──────────┘
             │ passes
             ▼
  ┌─────────────────────┐
  │  4. DOCKER BUILD    │  Builds multi-stage image
  │     (push disabled) │  Push is stubbed for the assignment
  └─────────────────────┘
```

**Key file**: `.github/workflows/ci.yml`

---

## 9. How MLflow Tracking Works

Every training run logs to MLflow:

```
MLflow Experiment: "text-classification"
│
├── Run: "baseline-run"
│   ├── Parameters: max_tfidf_features=50000, ngram_range=(1,2), ...
│   ├── Metrics: accuracy=0.9174, macro_f1=0.9172, precision_World=0.92, ...
│   └── Artifacts: confusion_matrix.json, classification_report.txt, model/
│
└── Run: "transformer-run"
    ├── Parameters: model_name=distilbert-base-uncased, epochs=1, ...
    ├── Metrics: accuracy=0.8937, macro_f1=0.8933, ...
    └── Artifacts: confusion_matrix.json, model/, registered_model/
```

The winning model is registered in the **Model Registry** as `text-classifier` version N.

**Storage**: We use `sqlite:///mlflow.db` locally. In production, you'd use a PostgreSQL backend + S3 for artifacts.

---

## 10. Monitoring & Observability

### Prometheus Metrics (GET /metrics)

| Metric | Type | Labels | What it measures |
|--------|------|--------|-----------------|
| `inference_requests_total` | Counter | endpoint, status | Total requests per endpoint |
| `inference_request_latency_seconds` | Histogram | endpoint | Request latency distribution |
| `inference_prediction_labels_total` | Counter | label | Which labels are predicted (drift signal) |
| `inference_errors_total` | Counter | endpoint, error_type | Error counts |

### Structured JSON Logs

Every prediction is logged as structured JSON:

```json
{"event": "prediction", "text_preview": "Stock market...", "label": "Business", "confidence": 0.97, "latency_ms": 3.4}
```

### How to detect drift (see `docs/drift_detection.md`)

- Watch `inference_prediction_labels_total` — if one label spikes, something changed.
- Watch prediction confidence — if it drops, the model is unsure about new data.
- Use the `/feedback` endpoint to collect corrections → track correction rate over time.

---

## 11. Explainability (LIME)

LIME answers: **"Why did the model predict this label?"**

It works by:
1. Taking one text, e.g. `"NASA launches Mars rover"`
2. Randomly removing words to create ~300 perturbed versions
3. Running the model on all perturbed versions
4. Fitting a simple model to see which words mattered most

Output: `artifacts/baseline/explanations/explanation_0.html` — an interactive HTML showing word importance.

**Failure analysis** (`artifacts/baseline/explanations/failure_analysis.txt`):
- Shows the most common confusion pairs (e.g., Business ↔ Sci/Tech)
- Explains *why* the model fails on those cases

---

## 12. Docker Setup

### Two containers:

```yaml
# docker-compose.yml
services:
  mlflow:           # MLflow tracking UI on port 5000
    build: Dockerfile.mlflow
    volumes: mlflow data persisted

  api:              # FastAPI inference server on port 8000
    build: Dockerfile
    depends_on: mlflow (healthy)
    volumes: models/ mounted in
    environment: MODEL_TYPE, PORT, LOG_LEVEL, etc.
```

### Dockerfile (multi-stage):

```
Stage 1 "builder":  python:3.11-slim → pip install → copy code
Stage 2 "runtime":  python:3.11-slim → copy installed packages → run as non-root user
```

**Why multi-stage?** The builder stage has pip, compilers, etc. The runtime stage only has what's needed to run — smaller, safer image.

---

## 13. Terraform (AWS Infrastructure)

Located in `terraform/`. This is a **design skeleton** — it shows *how* you'd deploy this for real.

```
┌──────────────────────────────────────────────┐
│  AWS                                          │
│                                               │
│  ECR Repository ──▶ Docker image storage      │
│  S3 Bucket ──────▶ Model artifact storage     │
│  ECS Fargate ────▶ Runs the API container     │
│  CloudWatch ─────▶ Logs + CPU/memory alarms   │
│  Auto-scaling ───▶ Scale on CPU > 70%         │
│  IAM Roles ──────▶ Least-privilege access     │
└──────────────────────────────────────────────┘
```

You don't need to run `terraform apply` — it's there to show production thinking.

---

## 14. Bonus Features

### A. LLM Classifier (`src/models/llm_classifier.py`)

Uses a few-shot prompt to classify text via an LLM (HuggingFace API or local Ollama). Includes a `compare_models()` function that benchmarks LLM vs fine-tuned model on accuracy and latency.

**When to use LLM**: Prototyping, zero-shot on new categories, no labeled data.
**When to use fine-tuned**: Production with consistent latency and calibrated scores.

### B. Feedback Loop (`POST /feedback`)

Users submit corrections → stored in SQLite → could trigger retraining when enough corrections accumulate. The endpoint works today; automated retraining is designed but not wired up.

---

## 15. Common Tasks Cheat Sheet

| I want to... | Command |
|--------------|---------|
| Train baseline model | `python scripts/train.py --model baseline` |
| Train transformer (CPU-friendly) | `TRANSFORMER_MAX_TRAIN_SAMPLES=5000 TRANSFORMER_EPOCHS=1 python scripts/train.py --model transformer` |
| Start API server | `MODEL_TYPE=baseline uvicorn src.serving.app:app --port 8000` |
| Run all tests | `pytest tests/ -v` |
| Lint code | `ruff check src/ tests/ scripts/` |
| Validate model for CI | `python scripts/validate_model.py baseline` |
| Generate LIME explanations | `python scripts/explain.py` |
| Start full stack (Docker) | `docker compose up --build` |
| Check API health | `curl http://localhost:8000/health` |

---

## 16. Key Design Decisions (and Why)

| Decision | Why |
|----------|-----|
| **AG News** over DBpedia | 4 classes vs 14 — faster to iterate, simpler to evaluate |
| **FastAPI** over Flask | Built-in validation, async, auto-generated docs at `/docs` |
| **MLflow** over W&B | Self-hosted, no account needed, Docker-friendly |
| **LIME** over SHAP | Faster for text, produces intuitive word-level explanations |
| **TF-IDF baseline** | Surprisingly strong on news text, <5ms latency, great sanity check |
| **DistilBERT** over BERT | 40% smaller, 97% of BERT's accuracy, practical for CPU |
| **SQLite for MLflow** | Zero setup for local dev; swap to PostgreSQL for production |
| **Prometheus** for metrics | Industry standard, scrape-based (no push infrastructure) |
