"""
Classification metrics for the mental-health text classifier.

The dataset is explicitly imbalanced (see README class distribution), so
accuracy alone is misleading — a model that always predicts the majority
class ("Normal") can still score a deceptively high accuracy. This module
computes macro/weighted F1, per-class precision/recall/F1, and a confusion
matrix so minority classes (e.g. "Bi-Polar", "Personality Disorder") aren't
hidden by strong majority-class performance.

Implemented with the standard library only — no new dependency (e.g. scikit-
learn) was added since this dataset has a small, fixed label set and the
formulas are simple to implement directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.data.prompts import VALID_LABELS


@dataclass
class EvaluationResult:
    split: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    unparseable_predictions: int = field(
        metadata={"help": "Predictions that didn't map to any known label after normalization."}
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())

    def human_summary(self) -> str:
        lines = [
            f"=== Evaluation ({self.split}) — {self.n_samples} samples ===",
            f"Accuracy:        {self.accuracy:.4f}",
            f"Macro Precision: {self.macro_precision:.4f}",
            f"Macro Recall:    {self.macro_recall:.4f}",
            f"Macro F1:        {self.macro_f1:.4f}   <- best single number given class imbalance",
            f"Weighted F1:     {self.weighted_f1:.4f}",
        ]
        if self.unparseable_predictions:
            lines.append(
                f"⚠️  Unparseable predictions (model output didn't match any known "
                f"label): {self.unparseable_predictions}"
            )
        lines.append("\nPer-class:")
        for label in VALID_LABELS:
            m = self.per_class.get(label, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            lines.append(
                f"  {label:<22} P={m['precision']:.3f}  R={m['recall']:.3f}  "
                f"F1={m['f1']:.3f}  support={int(m['support'])}"
            )
        return "\n".join(lines)


def compute_metrics(y_true: list[str], y_pred: list[str], split: str = "test") -> EvaluationResult:
    """Computes accuracy, macro/weighted F1, per-class P/R/F1 and a confusion matrix.

    `y_pred` entries that are not one of VALID_LABELS (e.g. malformed
    generations) are bucketed into an "<other>" confusion-matrix column and
    counted separately as `unparseable_predictions`, rather than being
    silently mapped onto a real class (which would distort that class's
    precision) or silently dropped (which would distort recall/support).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length, got {len(y_true)} vs {len(y_pred)}")

    labels = list(VALID_LABELS)
    n = len(y_true)
    columns = labels + ["<other>"]

    unparseable = sum(1 for p in y_pred if p not in labels)

    confusion: dict[str, dict[str, int]] = {t: {c: 0 for c in columns} for t in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        pred_col = pred_label if pred_label in labels else "<other>"
        if true_label in confusion:
            confusion[true_label][pred_col] += 1
        # Rows whose true label somehow isn't a known class are a data bug
        # upstream (the dataset pipeline already filters these out), so they
        # are intentionally not silently absorbed here.

    per_class: dict[str, dict[str, float]] = {}
    correct_total = 0
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        fn = support - tp
        fp = sum(confusion[t][label] for t in labels if t != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        correct_total += tp

    accuracy = correct_total / n if n else 0.0
    macro_precision = sum(m["precision"] for m in per_class.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels)

    total_support = sum(m["support"] for m in per_class.values())
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_class.values()) / total_support if total_support else 0.0
    )

    return EvaluationResult(
        split=split,
        n_samples=n,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=confusion,
        unparseable_predictions=unparseable,
    )
