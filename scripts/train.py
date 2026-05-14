#!/usr/bin/env python3
"""CLI entry-point for model training."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import run_full_training  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train text classification models")
    parser.add_argument(
        "--model",
        choices=["baseline", "transformer", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    results = run_full_training(model_type=args.model)

    print("\n── Summary ──")
    for name, data in results.items():
        if name == "champion":
            continue
        m = data["metrics"]
        print(f"  {name:12s}  accuracy={m['accuracy']}  macro_f1={m['macro_f1']}  run_id={data['run_id'][:8]}…")
    if "champion" in results:
        print(f"  → Champion: {results['champion']}")


if __name__ == "__main__":
    main()
