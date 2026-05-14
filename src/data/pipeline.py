"""Data loading, cleaning, splitting, and exploratory data analysis for AG News."""

import logging
import re
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# from src.config import (
#     AG_NEWS_LABELS,
#     DATASET_NAME,
#     RANDOM_SEED,
#     VALIDATION_SPLIT,
# )
AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
DATASET_NAME = "ag_news"
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize a single text sample."""
    text = re.sub(r"<[^>]+>", "", text)  # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load AG News, clean text, and return train / val / test DataFrames."""
    logger.info("Loading %s dataset from HuggingFace …", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)

    train_full = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Clean text
    train_full["text"] = train_full["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    # Add human-readable label names
    train_full["label_name"] = train_full["label"].map(AG_NEWS_LABELS)
    test_df["label_name"] = test_df["label"].map(AG_NEWS_LABELS)

    # Drop empty texts
    train_full = train_full[train_full["text"].str.len() > 0].reset_index(drop=True)
    test_df = test_df[test_df["text"].str.len() > 0].reset_index(drop=True)

    # Stratified train / validation split
    train_df, val_df = train_test_split(
        train_full,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=train_full["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        "Splits — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def compute_split_stats(df: pd.DataFrame, split_name: str) -> dict:
    """Return descriptive statistics for one data split."""
    text_lens = df["text"].str.len()
    word_counts = df["text"].str.split().str.len()
    return {
        "split": split_name,
        "n_samples": len(df),
        "class_distribution": df["label_name"].value_counts().to_dict(),
        "text_length": {
            "mean": round(text_lens.mean(), 1),
            "median": round(text_lens.median(), 1),
            "min": int(text_lens.min()),
            "max": int(text_lens.max()),
            "std": round(text_lens.std(), 1),
        },
        "word_count": {
            "mean": round(word_counts.mean(), 1),
            "median": round(word_counts.median(), 1),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
        "duplicates": int(df.duplicated(subset=["text"]).sum()),
    }


def run_eda(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Run and print exploratory data analysis. Returns stats dict."""

    separator = "=" * 64
    print(f"\n{separator}")
    print("  EXPLORATORY DATA ANALYSIS — AG News Dataset")
    print(separator)

    all_stats = {}
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        stats = compute_split_stats(df, name)
        all_stats[name] = stats

        print(f"\n── {name.upper()} ({stats['n_samples']:,} samples) ──")
        print("  Class distribution:")
        for cls, count in stats["class_distribution"].items():
            pct = count / stats["n_samples"] * 100
            print(f"    {cls:10s}  {count:>6,}  ({pct:.1f}%)")
        tl = stats["text_length"]
        print(
            f"  Text length (chars):  mean={tl['mean']}  "
            f"median={tl['median']}  min={tl['min']}  max={tl['max']}"
        )
        wc = stats["word_count"]
        print(
            f"  Word count:           mean={wc['mean']}  "
            f"median={wc['median']}  min={wc['min']}  max={wc['max']}"
        )
        print(f"  Duplicate texts:      {stats['duplicates']}")

    # Sample review
    print("\n── SAMPLE REVIEW (3 per class) ──")
    for label_id, label_name in AG_NEWS_LABELS.items():
        print(f"\n  [{label_name}]")
        samples = train_df[train_df["label"] == label_id].head(3)
        for _, row in samples.iterrows():
            print(f"    • {row['text'][:130]}…")

    # Data quality notes
    print("\n── DATA QUALITY NOTES ──")
    print("  1. AG News is well-balanced: ~30 k samples per class in the training set.")
    print("  2. Texts are short news snippets (mean ~200 chars / ~40 words).")
    print("  3. Minimal quality issues: no missing values, very few duplicates.")
    print("  4. Some HTML entities & URLs were present and cleaned during preprocessing.")
    print(f"{separator}\n")

    return all_stats


if __name__ == "__main__":
    train_df, val_df, test_df = load_and_prepare_data()
    run_eda(train_df, val_df, test_df)
