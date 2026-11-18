"""Layer 2: McNemar + paired bootstrap Delta macro-F1 on aligned test predictions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TABLES_DIR, load_pred_tsv, results_dir, write_tsv
from hsd.data.canonical import load_canonical_dataset
from hsd.data.paths import REPO_ROOT

ALPHA = 0.05
BOOTSTRAP_B = 10_000
SEED = 42

PRED_PATHS: dict[str, Path] = {
    "deberta_ghc": results_dir("exp-005a-deberta-large-lr2e-5/ghc") / "predictions_test.tsv",
    "deberta_dghd": results_dir("exp-005b-deberta-large-lr3e-5/dghd") / "predictions_test.tsv",
    "deberta_ktc": results_dir("exp-005c-deberta-large-lr1e-5/ktc") / "predictions_test.tsv",
    "gemma_ghc": REPO_ROOT / "results/exp-012a-gemma4-ghc-vo-note-test-nothink/ghc/predictions.tsv",
    "gemma_dghd": REPO_ROOT / "results/exp-012b-gemma4-dghd-vo-note-test-nothink/dghd/predictions.tsv",
    "gemma_ktc": REPO_ROOT / "results/exp-012c-gemma4-ktc-combined-test-nothink/ktc/predictions.tsv",
    "qwen25_ghc": results_dir("exp-008-qwen-lora/ghc") / "predictions_test.tsv",
    "qwen25_dghd": results_dir("exp-008a-qwen-lora-dghd-identity/dghd") / "predictions_test.tsv",
    "qwen25_ktc": results_dir("exp-008-qwen-lora/ktc") / "predictions_test.tsv",
    "qwen35_ghc": results_dir("exp-017-qwen35-ghc-recipe008/ghc") / "predictions_test.tsv",
    "qwen35_dghd": results_dir("exp-015-qwen35-dghd/dghd") / "predictions_test.tsv",
    "qwen35_ktc": results_dir("exp-014-qwen35-ktc/ktc") / "predictions_test.tsv",
}

NAMES: dict[str, str] = {
    "deberta_ghc": "DeBERTa-large",
    "deberta_dghd": "DeBERTa-large",
    "deberta_ktc": "DeBERTa-large",
    "gemma_ghc": "Gemma 4 12B",
    "gemma_dghd": "Gemma 4 12B",
    "gemma_ktc": "Gemma 4 12B",
    "qwen25_ghc": "Qwen2.5-3B",
    "qwen25_dghd": "Qwen2.5-3B",
    "qwen25_ktc": "Qwen2.5-3B",
    "qwen35_ghc": "Qwen3.5",
    "qwen35_dghd": "Qwen3.5",
    "qwen35_ktc": "Qwen3.5",
}

COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("Gemma vs DeBERTa", "GHC", "gemma_ghc", "deberta_ghc"),
    ("Gemma vs DeBERTa", "DGHD", "gemma_dghd", "deberta_dghd"),
    ("Gemma vs DeBERTa", "KTC", "gemma_ktc", "deberta_ktc"),
    ("Qwen2.5 vs Qwen3.5", "GHC", "qwen25_ghc", "qwen35_ghc"),
    ("Qwen2.5 vs Qwen3.5", "DGHD", "qwen25_dghd", "qwen35_dghd"),
    ("Qwen2.5 vs Qwen3.5", "KTC", "qwen25_ktc", "qwen35_ktc"),
    ("Qwen3.5 vs DeBERTa", "GHC", "qwen35_ghc", "deberta_ghc"),
    ("Qwen3.5 vs DeBERTa", "DGHD", "qwen35_dghd", "deberta_dghd"),
    ("Qwen3.5 vs DeBERTa", "KTC", "qwen35_ktc", "deberta_ktc"),
)

DATASET_KEY = {"GHC": "ghc", "DGHD": "dghd", "KTC": "ktc"}


def _load_pair(dataset: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    _, _, test = load_canonical_dataset(DATASET_KEY[dataset])
    y_true = test["hate"].astype(int).to_numpy()
    preds = load_pred_tsv(PRED_PATHS[key])
    if len(preds) != len(y_true):
        msg = f"{PRED_PATHS[key]}: expected {len(y_true)} rows, got {len(preds)}"
        raise ValueError(msg)
    if "y_true" in preds.columns and not (preds["y_true"].astype(int).to_numpy() == y_true).all():
        msg = f"{PRED_PATHS[key]}: y_true does not match canonical test order"
        raise ValueError(msg)
    return y_true, preds["y_pred"].astype(int).to_numpy()


def mcnemar(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> dict[str, Any]:
    """Run exact two-sided McNemar on examples where the models disagree.

    The binomial test asks whether either model wins more discordant pairs than expected by chance.
    """
    a_ok = y_a == y_true
    b_ok = y_b == y_true
    a_win = int(np.sum(a_ok & ~b_ok))
    b_win = int(np.sum(~a_ok & b_ok))
    n_disc = a_win + b_win
    p_value = 1.0 if n_disc == 0 else float(
        binomtest(a_win, n_disc, 0.5, alternative="two-sided").pvalue
    )
    return {
        "a_correct_b_wrong": a_win,
        "a_wrong_b_correct": b_win,
        "n_discordant": n_disc,
        "mcnemar_p": p_value,
        "mcnemar_sig": p_value < ALPHA,
    }


def bootstrap_delta(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
) -> dict[str, Any]:
    """Estimate uncertainty for model A minus model B macro-F1.

    Each resample uses the same example indices for both models, preserving pairing.
    """
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    f1_a = float(f1_score(y_true, y_a, average="macro", zero_division=0))
    f1_b = float(f1_score(y_true, y_b, average="macro", zero_division=0))
    observed = f1_a - f1_b
    deltas = np.empty(BOOTSTRAP_B, dtype=float)
    for i in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        deltas[i] = f1_score(y_true[idx], y_a[idx], average="macro", zero_division=0) - f1_score(
            y_true[idx], y_b[idx], average="macro", zero_division=0
        )
    lo = float(np.quantile(deltas, ALPHA / 2))
    hi = float(np.quantile(deltas, 1 - ALPHA / 2))
    return {
        "f1_a": round(f1_a, 6),
        "f1_b": round(f1_b, 6),
        "delta_f1": round(observed, 6),
        "ci_low": round(lo, 6),
        "ci_high": round(hi, 6),
        "bootstrap_sig": bool(lo > 0 or hi < 0),
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for family, corpus, key_a, key_b in COMPARISONS:
        y_true, y_a = _load_pair(corpus, key_a)
        _, y_b = _load_pair(corpus, key_b)
        mc = mcnemar(y_true, y_a, y_b)
        boot = bootstrap_delta(y_true, y_a, y_b)
        rows.append(
            {
                "layer": "2-stats",
                "family": family,
                "train_corpus": corpus,
                "model_a": NAMES[key_a],
                "model_b": NAMES[key_b],
                "n": len(y_true),
                **boot,
                **mc,
                # Require both tests to pass so the combined flag is conservative.
                "sig": bool(mc["mcnemar_sig"] and boot["bootstrap_sig"]),
            }
        )
        print(
            f"{family} / {corpus}: F1 {boot['f1_a']:.4f} vs {boot['f1_b']:.4f} "
            f"d={boot['delta_f1']:+.4f} p={mc['mcnemar_p']:.4g}"
        )
    write_tsv(pd.DataFrame(rows), TABLES_DIR / "stats_mcnemar_bootstrap.tsv")


if __name__ == "__main__":
    main()
