## How Would You Detect and Respond to Data Drift in Production?

### Detection Strategy

In production, drift manifests in two ways: **feature drift** (the statistical distribution of incoming text changes) and **concept drift** (the relationship between inputs and correct labels shifts). Both require distinct monitoring approaches.

**Feature drift** is tracked by comparing the distribution of incoming request features against a reference window from training data. For text, useful proxies include: average text length, vocabulary overlap with the training corpus, TF-IDF cosine similarity to training centroids, and the proportion of out-of-vocabulary tokens. We compute these metrics over rolling 1-hour and 24-hour windows and apply the Population Stability Index (PSI) to flag significant distributional shifts. A PSI > 0.2 triggers an alert.

**Concept drift** is harder to detect without ground-truth labels. Our primary signal comes from the **/feedback** endpoint: when operators or downstream systems correct predictions, we track the correction rate over time. A sustained increase in correction rate (e.g., > 15% over 24 hours) indicates the model's learned patterns no longer match reality. Additionally, we monitor **prediction confidence entropy** — if the model becomes increasingly uncertain (average top-class confidence drops below 0.6), this suggests the input distribution has shifted away from learned decision boundaries.

### Tooling

We would integrate **Evidently AI** as a drift monitoring library. On each batch of predictions, Evidently computes drift reports that can be exported to our existing Prometheus + Grafana stack. Statistical tests (Kolmogorov-Smirnov for numerical features, Jensen-Shannon divergence for categorical/text features) run on a scheduled basis (every 6 hours) against the training reference dataset.

### Response Plan

1. **Alert** — Slack/PagerDuty notification when PSI > 0.2 or correction rate spikes.
2. **Diagnose** — Generate an Evidently drift report; inspect which features shifted.
3. **Mitigate** — If drift is confirmed, trigger a lightweight retraining job using the original training data augmented with recent feedback-corrected samples from the `/feedback` store.
4. **Validate** — Run the retrained model through the CI model-validation gate (F1 > 0.85 on fixtures + hold-out set).
5. **Deploy** — Canary deployment at 10% traffic, monitor for 1 hour, then full rollout.
6. **Update reference** — Refresh the drift detection reference window to include the new data distribution.

This creates a **feedback flywheel**: production corrections flow into retraining, which improves the model, which reduces future corrections.
