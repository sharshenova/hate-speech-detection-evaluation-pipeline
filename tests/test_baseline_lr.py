"""Tests for baseline LR training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hsd.training.baseline import run_baseline_lr_experiment


def _write_canonical(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, sep="\t", index=False)


def _canonical_row(sample_id: str, text: str, hate: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "dataset": "ghc",
        "split": "train",
        "text": text,
        "hate": hate,
        "source": pd.NA,
        "tgt_rae": 0,
        "tgt_rel": 0,
        "tgt_nat": 0,
        "tgt_gen": 0,
        "tgt_sxo": 0,
        "tgt_opc": 0,
        "tgt_other": 0,
        "tgt_unknown": 0,
        "target_raw": pd.NA,
        "hate_function": pd.NA,
        "framing_explicit": pd.NA,
        "framing_implicit": pd.NA,
        "source_row_id": pd.NA,
        "identity_term": pd.NA,
        "template_type": pd.NA,
        "bias_eval_role": pd.NA,
    }


def test_run_baseline_lr_experiment_writes_metrics(tmp_path: Path) -> None:
    train_rows = [
        _canonical_row("t0", "benign example one", 0),
        _canonical_row("t1", "benign example two", 0),
        _canonical_row("t2", "hateful slur attack", 1),
        _canonical_row("t3", "another hateful attack", 1),
    ]
    dev_rows = [
        _canonical_row("d0", "benign dev example", 0),
        _canonical_row("d1", "hateful dev attack", 1),
    ]
    test_rows = [
        _canonical_row("s0", "benign test example", 0),
        _canonical_row("s1", "hateful test attack", 1),
    ]

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.tsv"
    dev_path = data_dir / "dev.tsv"
    test_path = data_dir / "test.tsv"
    _write_canonical(train_path, train_rows)
    _write_canonical(dev_path, dev_rows)
    _write_canonical(test_path, test_rows)

    config = {
        "experiment_id": "exp-test-baseline",
        "dataset": "ghc",
        "seed": 42,
        "data": {
            "text_column": "text",
            "label_column": "hate",
            "train_path": str(train_path),
            "dev_path": str(dev_path),
            "test_path": str(test_path),
        },
        "model": {
            "type": "tfidf_logistic_regression",
            "tfidf": {
                "max_features": 1000,
                "ngram_range": [1, 2],
                "min_df": 1,
                "max_df": 1.0,
                "sublinear_tf": True,
            },
            "logistic_regression": {
                "C": 1.0,
                "class_weight": None,
                "max_iter": 200,
                "solver": "lbfgs",
            },
        },
        "training": {
            "primary_metric": "macro_f1",
            "threshold_tuning": {
                "enabled": True,
                "split": "dev",
                "metric": "macro_f1",
                "min_threshold": 0.1,
                "max_threshold": 0.9,
                "step": 0.1,
            },
        },
        "results_dir": str(tmp_path / "results"),
    }

    results_dir = run_baseline_lr_experiment(config)
    metrics_path = results_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["test"]["f1_macro"] >= 0.5
    assert (results_dir / "model.joblib").exists()
