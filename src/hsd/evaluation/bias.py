"""Identity-wise FPR/FNR and FPED/FNED (Dixon-style, calculated on HateCheck)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def false_positive_rate(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> float:
    """FPR = FP / (FP + TN); 0.0 if there are no negatives."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    negatives = y_true_arr == 0
    n_neg = int(negatives.sum())
    if n_neg == 0:
        return 0.0
    fp = int(((y_pred_arr == 1) & negatives).sum())
    return fp / n_neg


def false_negative_rate(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> float:
    """FNR = FN / (FN + TP); 0.0 if there are no positives."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    positives = y_true_arr == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return 0.0
    fn = int(((y_pred_arr == 0) & positives).sum())
    return fn / n_pos


def _per_term_rate_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identity_term: np.ndarray,
    *,
    rate_fn: Any,
    rate_name: str,
) -> tuple[float, pd.DataFrame, float]:
    """Compare each identity rate with the rate over the full class pool.

    Signed gaps show direction; the equality difference sums their magnitudes.
    """
    overall = float(rate_fn(y_true, y_pred))
    terms = sorted({str(t) for t in identity_term})
    rows: list[dict[str, Any]] = []
    equality_diff = 0.0
    for term in terms:
        mask = identity_term == term
        rate = float(rate_fn(y_true[mask], y_pred[mask]))
        signed_gap = rate - overall
        abs_gap = abs(signed_gap)
        equality_diff += abs_gap
        rows.append(
            {
                "identity_term": term,
                "n": int(mask.sum()),
                rate_name: rate,
                "signed_gap_vs_overall": signed_gap,
                "abs_gap_vs_overall": abs_gap,
            }
        )
    table = pd.DataFrame(rows).sort_values("identity_term").reset_index(drop=True)
    return overall, table, float(equality_diff)


def compute_fped(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    identity_term: np.ndarray | list[str],
) -> dict[str, Any]:
    """False Positive Equality Difference on a non-hate (FPED) pool.

    The pool contains only gold non-hate rows, where positive predictions are errors.
    FPED = sum_t |FPR_overall - FPR_t|.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    terms = np.asarray(identity_term)
    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) != len(terms):
        msg = "y_true, y_pred, and identity_term must have the same length"
        raise ValueError(msg)
    if not np.all(y_true_arr == 0):
        msg = "FPED pool must contain only hate=0 / non-toxic labels"
        raise ValueError(msg)

    overall, table, fped = _per_term_rate_table(
        y_true_arr,
        y_pred_arr,
        terms,
        rate_fn=false_positive_rate,
        rate_name="fpr",
    )
    return {
        "fped": fped,
        "fpr_overall": overall,
        "n_terms": int(table.shape[0]),
        "n_rows": int(len(y_true_arr)),
        "per_term": table.to_dict(orient="records"),
    }


def compute_fned(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    identity_term: np.ndarray | list[str],
) -> dict[str, Any]:
    """False Negative Equality Difference on a hate (FNED) pool.

    The pool contains only gold hate rows, where negative predictions are errors.
    FNED = sum_t |FNR_overall - FNR_t|.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    terms = np.asarray(identity_term)
    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) != len(terms):
        msg = "y_true, y_pred, and identity_term must have the same length"
        raise ValueError(msg)
    if not np.all(y_true_arr == 1):
        msg = "FNED pool must contain only hate=1 / toxic labels"
        raise ValueError(msg)

    overall, table, fned = _per_term_rate_table(
        y_true_arr,
        y_pred_arr,
        terms,
        rate_fn=false_negative_rate,
        rate_name="fnr",
    )
    return {
        "fned": fned,
        "fnr_overall": overall,
        "n_terms": int(table.shape[0]),
        "n_rows": int(len(y_true_arr)),
        "per_term": table.to_dict(orient="records"),
    }


HATECHECK_MISSING_TARGET = "(missing)"


def compute_hatecheck_bias_metrics(
    frame: pd.DataFrame,
    *,
    y_true_col: str = "hate",
    y_pred_col: str = "y_pred",
    identity_col: str = "target_raw",
    missing_label: str = HATECHECK_MISSING_TARGET,
) -> dict[str, Any]:
    """FPED/FNED on HateCheck rows that name a protected group.

    Empty, null, and explicit missing ``target_raw`` values are excluded because
    they do not identify a group. FPED uses gold non-hate rows and FNED uses gold hate rows.
    """
    required = (y_true_col, y_pred_col, identity_col)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        msg = f"HateCheck bias metrics require columns: {missing}"
        raise ValueError(msg)

    terms = frame[identity_col]
    keep = (
        terms.notna() & terms.astype(str).str.strip().ne("") & terms.astype(str).ne(missing_label)
    )
    subset = frame.loc[keep]
    n_excluded = int((~keep).sum())
    if subset.empty:
        msg = "No HateCheck rows with a non-missing target_raw"
        raise ValueError(msg)

    y_true = subset[y_true_col].astype(int)
    fped_pool = subset.loc[y_true == 0]
    fned_pool = subset.loc[y_true == 1]
    if fped_pool.empty:
        msg = "No non-hate rows with target_raw for FPED"
        raise ValueError(msg)
    if fned_pool.empty:
        msg = "No hate rows with target_raw for FNED"
        raise ValueError(msg)

    fped = compute_fped(
        fped_pool[y_true_col],
        fped_pool[y_pred_col],
        fped_pool[identity_col].astype(str),
    )
    fned = compute_fned(
        fned_pool[y_true_col],
        fned_pool[y_pred_col],
        fned_pool[identity_col].astype(str),
    )
    overall_fp = int((fped_pool[y_pred_col].astype(int) == 1).sum())
    overall_fn = int((fned_pool[y_pred_col].astype(int) == 0).sum())
    return {
        "fped": fped["fped"],
        "fned": fned["fned"],
        "fpr_overall": fped["fpr_overall"],
        "fnr_overall": fned["fnr_overall"],
        "fp": overall_fp,
        "fn": overall_fn,
        "n_identity_terms_fped": fped["n_terms"],
        "n_identity_terms_fned": fned["n_terms"],
        "n_fped_rows": fped["n_rows"],
        "n_fned_rows": fned["n_rows"],
        "n_missing_target_excluded": n_excluded,
        "fped_per_term": fped["per_term"],
        "fned_per_term": fned["per_term"],
    }
