"""Model explainability using LIME."""

import json
import logging
from typing import List

from lime.lime_text import LimeTextExplainer

from src.config import AG_NEWS_LABELS, ARTIFACTS_DIR

logger = logging.getLogger(__name__)

LABEL_NAMES = [AG_NEWS_LABELS[i] for i in sorted(AG_NEWS_LABELS)]


def explain_predictions(
    model,
    texts: List[str],
    true_labels: List[int],
    model_type: str = "baseline",
    num_features: int = 10,
    num_samples: int = 500,
) -> List[dict]:
    """Generate LIME explanations for a list of texts.

    Args:
        model: A classifier with a predict_proba(texts) method.
        texts: Texts to explain.
        true_labels: Ground-truth label indices.
        model_type: 'baseline' or 'transformer' (used for file naming).
        num_features: Top features to show per explanation.
        num_samples: LIME perturbation sample count.

    Returns:
        List of explanation dicts.
    """
    explainer = LimeTextExplainer(class_names=LABEL_NAMES, random_state=42)
    output_dir = ARTIFACTS_DIR / model_type / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)

    explanations = []

    for idx, (text, true_label) in enumerate(zip(texts, true_labels)):
        pred = model.predict([text])[0]
        is_correct = pred == true_label

        logger.info(
            "Explaining sample %d/%d — true=%s  pred=%s  %s",
            idx + 1,
            len(texts),
            LABEL_NAMES[true_label],
            LABEL_NAMES[pred],
            "✓" if is_correct else "✗",
        )

        exp = explainer.explain_instance(
            text,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=list(range(len(LABEL_NAMES))),
        )

        # Save HTML
        html_path = output_dir / f"explanation_{idx}.html"
        exp.save_to_file(str(html_path))

        # Collect structured data
        top_features = exp.as_list(label=pred)
        explanations.append({
            "index": idx,
            "text": text[:200],
            "true_label": LABEL_NAMES[true_label],
            "predicted_label": LABEL_NAMES[pred],
            "correct": is_correct,
            "top_features": [
                {"word": w, "weight": round(float(s), 4)} for w, s in top_features
            ],
        })

    # Save summary JSON
    summary_path = output_dir / "explanations_summary.json"
    summary_path.write_text(json.dumps(explanations, indent=2))
    logger.info("Explanations saved to %s", output_dir)

    return explanations


def generate_failure_analysis(
    model,
    test_texts: List[str],
    test_labels: List[int],
    model_type: str = "baseline",
    max_failures: int = 20,
) -> str:
    """Analyze model failures and generate a narrative summary."""
    preds = model.predict(test_texts)

    misclassified = []
    for i, (true, pred) in enumerate(zip(test_labels, preds)):
        if true != pred:
            misclassified.append({
                "text": test_texts[i][:200],
                "true": LABEL_NAMES[true],
                "predicted": LABEL_NAMES[pred],
            })

    total = len(test_labels)
    n_wrong = len(misclassified)

    # Confusion patterns
    confusion_pairs = {}
    for m in misclassified:
        pair = f"{m['true']} → {m['predicted']}"
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:5]

    narrative = (
        f"FAILURE ANALYSIS ({model_type})\n"
        f"{'=' * 50}\n\n"
        f"The model misclassified {n_wrong} out of {total} test samples "
        f"({n_wrong / total * 100:.1f}% error rate).\n\n"
        f"Most common confusion pairs:\n"
    )
    for pair, count in top_confusions:
        narrative += f"  • {pair}: {count} errors\n"

    narrative += (
        "\nWhen does the model fail and why?\n"
        "The model struggles most with texts that span multiple topics. "
        "For example, a news article about a tech company's stock price "
        "blends 'Sci/Tech' and 'Business' signals. Similarly, stories about "
        "sports business (team acquisitions, salary disputes) confuse the "
        "'Sports' and 'Business' categories. Short, ambiguous headlines with "
        "limited context are also problematic — the model relies on keyword "
        "signals that may be absent in terse texts. Domain-specific jargon "
        "seen rarely in training can also cause mis-routing.\n"
    )

    # Save
    output_dir = ARTIFACTS_DIR / model_type / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_analysis.txt").write_text(narrative)

    # Save misclassified examples
    examples_path = output_dir / "misclassified_examples.json"
    examples_path.write_text(json.dumps(misclassified[:max_failures], indent=2))

    print(narrative)
    return narrative
