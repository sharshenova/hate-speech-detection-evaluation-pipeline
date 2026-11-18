"""Tests for FPED / FNED metrics on HateCheck identity slices."""

from __future__ import annotations

import pandas as pd
import pytest

from hsd.evaluation.bias import (
    compute_fned,
    compute_fped,
    compute_hatecheck_bias_metrics,
    false_negative_rate,
    false_positive_rate,
)


def test_false_positive_and_negative_rates() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 0, 1]
    assert false_positive_rate(y_true, y_pred) == pytest.approx(1 / 3)
    assert false_negative_rate(y_true, y_pred) == pytest.approx(0.5)


def test_compute_fped_simple() -> None:
    y_true = [0, 0, 0, 0]
    y_pred = [1, 0, 1, 0]
    terms = ["a", "a", "b", "b"]
    result = compute_fped(y_true, y_pred, terms)
    assert result["fpr_overall"] == pytest.approx(0.5)
    assert result["fped"] == pytest.approx(0.0)
    assert result["n_terms"] == 2


def test_compute_fped_unequal() -> None:
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 0, 0]
    terms = ["a", "a", "b", "b"]
    result = compute_fped(y_true, y_pred, terms)
    assert result["fped"] == pytest.approx(1.0)
    per_term = {row["identity_term"]: row for row in result["per_term"]}
    assert per_term["a"]["signed_gap_vs_overall"] == pytest.approx(0.5)
    assert per_term["b"]["signed_gap_vs_overall"] == pytest.approx(-0.5)


def test_compute_fned_unequal() -> None:
    y_true = [1, 1, 1, 1]
    y_pred = [0, 0, 1, 1]
    terms = ["a", "a", "b", "b"]
    result = compute_fned(y_true, y_pred, terms)
    assert result["fned"] == pytest.approx(1.0)
    assert result["fnr_overall"] == pytest.approx(0.5)


def test_compute_fped_rejects_positives() -> None:
    with pytest.raises(ValueError, match="hate=0"):
        compute_fped([0, 1], [0, 0], ["a", "a"])


def test_compute_hatecheck_bias_metrics_drops_missing_target() -> None:
    frame = pd.DataFrame(
        [
            {"hate": 0, "y_pred": 1, "target_raw": "women"},
            {"hate": 0, "y_pred": 0, "target_raw": "women"},
            {"hate": 0, "y_pred": 0, "target_raw": "Muslims"},
            {"hate": 0, "y_pred": 0, "target_raw": "Muslims"},
            {"hate": 1, "y_pred": 0, "target_raw": "women"},
            {"hate": 1, "y_pred": 1, "target_raw": "women"},
            {"hate": 1, "y_pred": 1, "target_raw": "Muslims"},
            {"hate": 1, "y_pred": 1, "target_raw": "Muslims"},
            {"hate": 0, "y_pred": 1, "target_raw": None},
            {"hate": 0, "y_pred": 1, "target_raw": "(missing)"},
        ]
    )
    result = compute_hatecheck_bias_metrics(frame)
    assert result["n_missing_target_excluded"] == 2
    assert result["n_fped_rows"] == 4
    assert result["n_fned_rows"] == 4
    assert result["n_identity_terms_fped"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["fped"] == pytest.approx(0.5)
    assert result["fned"] == pytest.approx(0.5)
