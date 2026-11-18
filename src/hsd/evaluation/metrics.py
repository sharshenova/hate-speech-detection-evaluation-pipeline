"""Classification metrics for hate-speech experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    *,
    labels: tuple[int, ...] = (0, 1),
) -> dict[str, Any]:
    """Compute standard binary/multiclass metrics with macro-F1 as primary."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="macro",
        labels=list(labels),
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="weighted",
        labels=list(labels),
        zero_division=0,
    )

    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=list(labels),
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": {
            str(label): {
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=list(labels),
            target_names=[str(label) for label in labels],
            zero_division=0,
        ),
    }


def confusion_counts(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> tuple[int, int, int, int]:
    """Return TP, FP, FN, TN for the positive class (hate=1)."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    return tp, fp, fn, tn


def tune_binary_threshold(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    *,
    metric: str = "macro_f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """Search a probability threshold on dev data."""
    if metric != "macro_f1":
        msg = f"Unsupported threshold metric: {metric}"
        raise ValueError(msg)

    y_true_arr = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)

    best_threshold = 0.5
    best_value = -1.0
    threshold = min_threshold
    while threshold <= max_threshold + 1e-9:
        preds = (scores >= threshold).astype(int)
        value = float(f1_score(y_true_arr, preds, average="macro", zero_division=0))
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
        threshold += step

    return best_threshold, best_value


def apply_threshold(y_score: np.ndarray | list[float], threshold: float) -> np.ndarray:
    """Convert positive-class probabilities to binary predictions."""
    scores = np.asarray(y_score, dtype=float)
    return (scores >= threshold).astype(int)
