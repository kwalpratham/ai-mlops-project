# Text Classification Service — AI & MLOps Assignment

A production-grade, end-to-end text classification system that auto-routes customer support tickets into **World**, **Sports**, **Business**, and **Sci/Tech** categories using the AG News dataset as a stand-in for internal data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                           │
│                                                                    │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ AG News  │───▶│  Data        │───▶│  Model Training          │  │
│  │ Dataset  │    │  Pipeline    │    │  ┌────────┐ ┌──────────┐ │  │
│  └──────────┘    │  • clean     │    │  │TF-IDF +│ │DistilBERT│ │  │
│                  │  • split     │    │  │LogReg  │ │Fine-tuned│ │  │
│                  │  • EDA       │    │  └───┬────┘ └────┬─────┘ │  │
│                  └──────────────┘    │      │           │       │  │
│                                     │      └─────┬─────┘       │  │
│                                     └────────────│─────────────┘  │
│                                                  │                │
│                                     ┌────────────▼─────────────┐  │
│                                     │  MLflow Tracking Server  │  │
│                                     │  • experiments           │  │
│                                     │  • model registry        │  │
│                                     │  • artifacts             │  │
│                                     └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER                               │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  FastAPI Inference Server                    │  │
│  │                                                              │  │
│  │  POST /predict          → single text classification         │  │
│  │  POST /predict/batch    → batch classification               │  │
│  │  GET  /health           → liveness + readiness               │  │
│  │  GET  /model/info       → model metadata                     │  │
│  │  GET  /metrics          → Prometheus metrics                 │  │
│  │  POST /feedback         → correction feedback (Bonus C)      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                          │                │                        │
│              ┌───────────▼──┐    ┌───────▼────────┐               │
│              │  Prometheus  │    │  Structured     │               │
│              │  Metrics     │    │  JSON Logs      │               │
│              │  • latency   │    │  • predictions  │               │
│              │  • counts    │    │  • errors       │               │
│              │  • errors    │    │  • latency      │               │
│              └──────────────┘    └────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE                              │
│                                                                    │
│  Docker Compose             GitHub Actions         Terraform       │
│  ┌──────────────┐          ┌──────────────┐       ┌────────────┐  │
│  │ api (FastAPI)│          │ lint → test  │       │ ECS Fargate│  │
│  │ mlflow       │          │ → validate  │       │ S3 Artifacts│  │
│  └──────────────┘          │ → docker    │       │ CloudWatch  │  │
│                            └──────────────┘       └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)

### Local Development

```bash
# Clone and set up
git clone <repo-url> && cd ai-mlops-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Train models (baseline only — fast)
python scripts/train.py --model baseline

# Train both models
python scripts/train.py --model all

# Run tests
pytest tests/ -v

# Start the API server
MODEL_TYPE=baseline uvicorn src.serving.app:app --reload --port 8000

# Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Stock market rallies after Fed announces rate cut"}'
```

### Docker Deployment

```bash
# Train models first (or mount pre-trained models)
python scripts/train.py --model baseline

# Start everything with one command
docker compose up --build

# API available at http://localhost:8000
# MLflow UI at http://localhost:5000
```

---

## Project Structure

```
ai-mlops-project/
├── src/
│   ├── config.py                  # Central configuration (env vars + defaults)
│   ├── data/
│   │   └── pipeline.py            # Data loading, cleaning, EDA
│   ├── models/
│   │   ├── baseline.py            # TF-IDF + Logistic Regression
│   │   ├── transformer.py         # DistilBERT fine-tuning
│   │   └── llm_classifier.py      # Bonus A: LLM few-shot classifier
│   ├── training/
│   │   └── train.py               # Training orchestration + MLflow
│   ├── explainability/
│   │   └── explain.py             # LIME explanations + failure analysis
│   └── serving/
│       ├── app.py                 # FastAPI application
│       ├── schemas.py             # Pydantic request/response models
│       └── metrics.py             # Prometheus + structured logging
├── tests/                         # Unit + integration tests
├── scripts/
│   ├── train.py                   # CLI training entry-point
│   └── validate_model.py          # CI model-validation gate
├── fixtures/
│   └── test_inputs.json           # 10 fixture inputs for CI
├── terraform/                     # Bonus B: AWS IaC skeleton
├── docs/
│   └── drift_detection.md         # Monitoring & drift write-up
├── Dockerfile                     # Multi-stage API image
├── Dockerfile.mlflow              # MLflow tracking server
├── docker-compose.yml             # Full stack: API + MLflow
├── .github/workflows/ci.yml       # CI/CD pipeline
├── requirements.txt
└── requirements-dev.txt
```

---

## Design Decisions & Trade-offs

### Model Selection

| Model | Accuracy | Macro-F1 | Inference Latency | Training Time |
|-------|----------|----------|-------------------|---------------|
| TF-IDF + LogReg | ~92% | ~0.92 | <5ms | ~30s |
| DistilBERT | ~94% | ~0.94 | ~50ms (CPU) | ~45min (GPU) |

**Why these two?**
- **TF-IDF + LogReg** serves as a strong, interpretable baseline. It's fast to train, fast to serve, and surprisingly competitive on news classification. It's the right choice when latency matters more than 2% accuracy.
- **DistilBERT** captures semantic nuance that bag-of-words misses. It's 40% smaller than BERT with ~97% of its accuracy. On AG News, it consistently hits >94% accuracy.

**Champion selection**: The transformer wins on accuracy/F1 but the baseline is preferred when latency constraints are tight (<10ms SLA). The training pipeline logs both to MLflow and auto-selects the winner by macro-F1.

### Technology Choices

- **FastAPI** over Flask: native async, automatic OpenAPI docs, Pydantic validation, built-in dependency injection. Production-ready out of the box.
- **MLflow** over W&B: fully self-hosted, no account needed, docker-compose friendly, open-source. Easier for evaluators to run locally.
- **LIME** over SHAP: faster execution for text, produces intuitive word-level explanations, works identically with both sklearn and transformer models.
- **Prometheus** for metrics: industry standard, scrape-based (no push infrastructure needed), integrates trivially with Grafana.
- **AG News** over DBpedia: 4 classes vs 14 — faster iteration, simpler to evaluate, sufficient complexity to demonstrate the full pipeline.

### Containerization

- Multi-stage Docker build reduces image size (no build tools in runtime).
- Models are volume-mounted rather than baked into the image — allows model updates without rebuilds.
- Environment variables for all configuration — no hardcoded paths.
- Health checks ensure container orchestrators can detect failures.

---

## API Reference

### `POST /predict`
```json
// Request
{"text": "NASA launches new Mars rover with AI navigation"}

// Response
{
  "predicted_label": "Sci/Tech",
  "confidence_scores": {"World": 0.02, "Sports": 0.01, "Business": 0.05, "Sci/Tech": 0.92},
  "latency_ms": 3.42
}
```

### `POST /predict/batch`
```json
// Request
{"texts": ["Stock market rises", "Lakers win championship"]}

// Response
{
  "predictions": [
    {"predicted_label": "Business", "confidence_scores": {...}, "latency_ms": 1.5},
    {"predicted_label": "Sports", "confidence_scores": {...}, "latency_ms": 1.5}
  ],
  "total_latency_ms": 3.1
}
```

### `GET /health`
```json
{"status": "healthy", "model_loaded": true, "model_type": "baseline", "timestamp": "2026-05-13T..."}
```

### `GET /model/info`
```json
{"model_type": "baseline", "model_version": "1.0.0", "classes": ["World","Sports","Business","Sci/Tech"], "parameters": {...}}
```

### `GET /metrics`
Prometheus exposition format — includes `inference_requests_total`, `inference_request_latency_seconds`, `inference_prediction_labels_total`, `inference_errors_total`.

### `POST /feedback` (Bonus C)
```json
// Request
{"text": "...", "predicted_label": "Business", "correct_label": "Sci/Tech"}

// Response
{"status": "accepted", "message": "Feedback recorded...", "feedback_id": 42}
```

---

## CI/CD Pipeline

```
 Pull Request / Push to main
          │
          ▼
    ┌───────────┐
    │   Lint     │  ruff check src/ tests/ scripts/
    │  (ruff)    │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │   Test     │  pytest tests/ -v
    │ (pytest)   │
    └─────┬─────┘
          │
          ├─────────────────────────────┐
          ▼                             ▼
    ┌───────────────┐          ┌───────────────┐
    │ Model         │          │ Docker Build   │
    │ Validation    │          │ + Push (GHCR)  │
    │ (10 fixtures) │          └───────────────┘
    │ F1 > 0.85     │
    └───────────────┘
```

---

## Bonus Challenges

### Bonus A: LLM Integration (+8 pts)

`src/models/llm_classifier.py` implements a few-shot classifier using either HuggingFace Inference API or a local Ollama instance. The `compare_models()` function runs both the fine-tuned model and the LLM on the same inputs and reports accuracy and latency side by side.

**When to use each approach:**
- **Fine-tuned model**: Production workloads requiring low latency (<10ms), consistent behavior, and calibrated confidence scores.
- **LLM (few-shot)**: Rapid prototyping, zero-shot classification on new categories without labeled data, or as a teacher model to bootstrap training labels.

### Bonus B: Terraform IaC (+6 pts)

`terraform/` contains a complete AWS ECS Fargate deployment skeleton:
- ECR repository for container images
- S3 bucket (versioned, encrypted) for model artifacts
- ECS cluster + Fargate service with health checks
- CloudWatch log group + CPU/memory alarms
- Auto-scaling policy (target: 70% CPU)
- IAM roles with least-privilege S3 access

### Bonus C: Feedback Loop (+6 pts)

`POST /feedback` accepts label corrections and stores them in SQLite. The feedback flywheel:
1. **Collect**: Operators correct predictions via `/feedback`
2. **Accumulate**: Corrections stored with timestamps in SQLite
3. **Trigger**: When correction count exceeds threshold, trigger retraining
4. **Retrain**: Augment training data with feedback-corrected samples
5. **Validate**: Run CI model gate (F1 > 0.85 on fixtures)
6. **Deploy**: Canary deployment → monitor → full rollout

See `docs/drift_detection.md` for the full flywheel architecture.

---

## Known Limitations

1. **Transformer inference on CPU is slow** (~50ms per request). For production, ONNX Runtime or TorchScript optimization would cut this to ~10ms.
2. **No GPU support in Docker** — the Dockerfile targets CPU-only for portability.
3. **Feedback retraining is a stub** — the endpoint stores data but doesn't trigger actual retraining automatically.
4. **LLM comparison** requires an API token or local Ollama — may not work out of the box.
5. **Single-worker serving** — for production, use multiple Uvicorn workers behind an NGINX reverse proxy.

## What I'd Do With More Time

1. **ONNX export** — Convert the DistilBERT model to ONNX for 3-5x faster CPU inference.
2. **A/B testing framework** — Serve multiple model versions simultaneously and route traffic based on experiment config.
3. **Full Grafana dashboard** — Prometheus → Grafana with panels for latency percentiles, error rates, and label distribution over time.
4. **Automated retraining pipeline** — Airflow DAG that triggers on feedback volume thresholds, retrains, validates, and deploys.
5. **Model distillation** — Use the transformer as a teacher to train a lighter student model for the lowest possible latency.
6. **Load testing** — Locust or k6 benchmarks to establish throughput limits and inform auto-scaling policies.
7. **Data versioning** — DVC for tracking dataset versions alongside model versions.
8. **Kubernetes deployment** — Helm chart for EKS with HPA, pod disruption budgets, and rolling updates.

---

## License

This project is submitted as part of a technical assessment and is not licensed for redistribution.
