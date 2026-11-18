"""Tests for classification metrics."""

from __future__ import annotations

import numpy as np

from hsd.evaluation.metrics import apply_threshold, classification_metrics, tune_binary_threshold


def test_classification_metrics_macro_f1() -> None:
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
    assert metrics["accuracy"] == 0.75
    assert "f1_macro" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert set(metrics["per_class"]) == {"0", "1"}


def test_tune_binary_threshold_prefers_better_cutoff() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.55, 0.9])
    threshold, best_f1 = tune_binary_threshold(y_true, scores, step=0.05)
    preds = apply_threshold(scores, threshold)
    assert best_f1 >= 0.99
    assert preds.tolist() == [0, 0, 1, 1]
