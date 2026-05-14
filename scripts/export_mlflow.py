"""Export MLflow run comparison data."""
import os
import json
import mlflow

os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.getcwd()}/mlflow.db"
)

runs = mlflow.search_runs(experiment_names=["text-classification"])
print("=== MLflow Experiment Runs ===")
print(f"Total runs: {len(runs)}")
print()

for _, row in runs.iterrows():
    model_type = row.get("params.model_type", "unknown")
    accuracy = row.get("metrics.accuracy", 0)
    macro_f1 = row.get("metrics.macro_f1", 0)
    run_id = row["run_id"][:12]
    print(f"  {model_type:15s} accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}  run_id={run_id}")

# Export comparison as JSON
comparison = []
for _, row in runs.iterrows():
    entry = {
        "run_id": row["run_id"],
        "model_type": row.get("params.model_type", "unknown"),
        "accuracy": round(row.get("metrics.accuracy", 0), 4),
        "macro_f1": round(row.get("metrics.macro_f1", 0), 4),
    }
    # Add per-class metrics if available
    for cls in ["World", "Sports", "Business", "Sci_Tech"]:
        for metric in ["precision", "recall", "f1"]:
            key = f"metrics.{metric}_{cls}"
            if key in row and row[key] is not None:
                entry[f"{metric}_{cls}"] = round(row[key], 4)
    comparison.append(entry)

out_path = "artifacts/mlflow_run_comparison.json"
os.makedirs("artifacts", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nExported to {out_path}")

# Print champion justification
if len(comparison) >= 2:
    best = max(comparison, key=lambda x: x["macro_f1"])
    print(f"\n  Champion: {best['model_type']} (macro_f1={best['macro_f1']})")
