"""Shared helpers for thesis TSV tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hsd.data.paths import REPO_ROOT
from hsd.evaluation.metrics import classification_metrics, confusion_counts

TABLES_DIR = REPO_ROOT / "results" / "tables"

OVERALL_COLUMNS: tuple[str, ...] = (
    "layer",
    "train_corpus",
    "eval_set",
    "model",
    "decode",
    "n",
    "n_hate",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "hate_P",
    "hate_R",
    "hate_F1",
    "FP",
    "FN",
    "FPR",
    "FNR",
    "threshold",
)

SLICE_COLUMNS: tuple[str, ...] = (
    "layer",
    "train_corpus",
    "eval_set",
    "model",
    "stratum",
    "n",
    "n_hate",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "hate_P",
    "hate_R",
    "hate_F1",
    "FP",
    "FN",
    "FPR",
    "FNR",
)

# Six checkpoints used in Layers 3-6.
SIX_CHECKPOINTS: tuple[dict[str, str], ...] = (
    {
        "train_corpus": "GHC",
        "model": "DeBERTa-large",
        "decode": "threshold",
        "results_subdir": "exp-005a-deberta-large-lr2e-5/ghc",
        "dataset": "ghc",
    },
    {
        "train_corpus": "DGHD",
        "model": "DeBERTa-large",
        "decode": "threshold",
        "results_subdir": "exp-005b-deberta-large-lr3e-5/dghd",
        "dataset": "dghd",
    },
    {
        "train_corpus": "KTC",
        "model": "DeBERTa-large",
        "decode": "threshold",
        "results_subdir": "exp-005c-deberta-large-lr1e-5/ktc",
        "dataset": "ktc",
    },
    {
        "train_corpus": "GHC",
        "model": "Qwen3.5",
        "decode": "greedy",
        "results_subdir": "exp-017-qwen35-ghc-recipe008/ghc",
        "dataset": "ghc",
    },
    {
        "train_corpus": "DGHD",
        "model": "Qwen3.5",
        "decode": "logit",
        "results_subdir": "exp-015-qwen35-dghd/dghd",
        "dataset": "dghd",
    },
    {
        "train_corpus": "KTC",
        "model": "Qwen3.5",
        "decode": "logit",
        "results_subdir": "exp-014-qwen35-ktc/ktc",
        "dataset": "ktc",
    },
)

TARGET_LABELS: dict[str, str] = {
    "tgt_rae": "race/ethnicity",
    "tgt_rel": "religion",
    "tgt_nat": "nationality",
    "tgt_gen": "gender",
    "tgt_sxo": "sexuality",
    "tgt_opc": "other protected",
    "tgt_other": "other",
    "tgt_unknown": "unknown",
}


def results_dir(subdir: str) -> Path:
    return REPO_ROOT / "results" / subdir


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pred_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        msg = f"Missing predictions: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, sep="\t")


def write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path} ({len(frame)} rows)")
    return path


def _fmt(value: float | None, digits: int = 6) -> str | float:
    if value is None:
        return ""
    return round(float(value), digits)


def row_from_preds(
    y_true: pd.Series | list[int],
    y_pred: pd.Series | list[int],
    *,
    layer: str,
    train_corpus: str,
    eval_set: str,
    model: str,
    decode: str,
    threshold: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y_true_s = pd.Series(y_true).astype(int)
    y_pred_s = pd.Series(y_pred).astype(int)
    metrics = classification_metrics(y_true_s.to_numpy(), y_pred_s.to_numpy())
    tp, fp, fn, tn = confusion_counts(y_true_s.to_numpy(), y_pred_s.to_numpy())
    n = int(len(y_true_s))
    n_hate = int((y_true_s == 1).sum())
    n_nonhate = n - n_hate
    hate = metrics["per_class"]["1"]
    row: dict[str, Any] = {
        "layer": layer,
        "train_corpus": train_corpus,
        "eval_set": eval_set,
        "model": model,
        "decode": decode,
        "n": n,
        "n_hate": n_hate,
        "accuracy": _fmt(metrics["accuracy"]),
        "precision_macro": _fmt(metrics["precision_macro"]),
        "recall_macro": _fmt(metrics["recall_macro"]),
        "f1_macro": _fmt(metrics["f1_macro"]),
        "hate_P": _fmt(hate["precision"]),
        "hate_R": _fmt(hate["recall"]),
        "hate_F1": _fmt(hate["f1"]),
        "FP": fp,
        "FN": fn,
        "FPR": _fmt(fp / n_nonhate if n_nonhate else None),
        "FNR": _fmt(fn / n_hate if n_hate else None),
        "threshold": "" if threshold is None else _fmt(threshold, 3),
        "TP": tp,
        "TN": tn,
    }
    if extra:
        row.update(extra)
    return row


def row_from_metrics_split(
    payload: dict[str, Any],
    split: str,
    *,
    layer: str,
    train_corpus: str,
    eval_set: str,
    model: str,
    decode: str,
) -> dict[str, Any]:
    """Build a table row when only saved aggregate metrics are available.

    Confusion counts are estimated from class support, precision, and recall,
    then rounded. If hate precision is zero, false positives default to zero.
    """
    block = payload["metrics"][split]
    hate = block["per_class"]["1"]
    support_hate = int(hate["support"])
    support_non = int(block["per_class"]["0"]["support"])
    n = support_hate + support_non
    hate_p = float(hate["precision"])
    hate_r = float(hate["recall"])
    tp = hate_r * support_hate
    fn = support_hate - tp
    fp = (tp / hate_p - tp) if hate_p > 0 else 0.0
    threshold = payload.get("threshold")
    return {
        "layer": layer,
        "train_corpus": train_corpus,
        "eval_set": eval_set,
        "model": model,
        "decode": decode,
        "n": n,
        "n_hate": support_hate,
        "accuracy": _fmt(block["accuracy"]),
        "precision_macro": _fmt(block["precision_macro"]),
        "recall_macro": _fmt(block["recall_macro"]),
        "f1_macro": _fmt(block["f1_macro"]),
        "hate_P": _fmt(hate_p),
        "hate_R": _fmt(hate_r),
        "hate_F1": _fmt(hate["f1"]),
        "FP": int(round(fp)),
        "FN": int(round(fn)),
        "FPR": _fmt(fp / support_non if support_non else None),
        "FNR": _fmt(fn / support_hate if support_hate else None),
        "threshold": "" if threshold is None else _fmt(float(threshold), 3),
    }


def join_eval_preds(eval_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Attach predictions by sample ID when possible, otherwise by checked row order.

    When saved gold labels are available, the fallback checks them before using
    positional alignment.
    """
    out = eval_df.copy()
    if "sample_id" in pred_df.columns and "sample_id" in out.columns:
        merged = out.merge(
            pred_df[["sample_id", "y_pred"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if merged["y_pred"].isna().any():
            msg = "Prediction sample_id join left null y_pred"
            raise ValueError(msg)
        return merged

    if len(pred_df) != len(out):
        msg = f"Pred length {len(pred_df)} != eval {len(out)}"
        raise ValueError(msg)
    if "y_true" in pred_df.columns:
        gold = out["hate"].astype(int).to_numpy()
        pred_true = pred_df["y_true"].astype(int).to_numpy()
        if not (gold == pred_true).all():
            msg = "y_true in predictions does not match eval gold order"
            raise ValueError(msg)
    out = out.copy()
    out["y_pred"] = pred_df["y_pred"].astype(int).to_numpy()
    return out
