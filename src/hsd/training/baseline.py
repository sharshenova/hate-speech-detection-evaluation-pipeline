"""Run classical baseline experiments from YAML configs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hsd.data.canonical import load_canonical_dataset, xy_from_frame
from hsd.evaluation.metrics import (
    apply_threshold,
    classification_metrics,
    tune_binary_threshold,
)
from hsd.models.baseline_lr import build_tfidf_logistic_pipeline, positive_class_proba
from hsd.utils.config import resolve_results_dir
from hsd.utils.seed import set_global_seed


def _split_class_distribution(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


def run_baseline_lr_experiment(config: dict[str, Any], *, config_path: Path | None = None) -> Path:
    """Train/evaluate a TF-IDF + logistic regression baseline and write artifacts."""
    seed = int(config["seed"])
    set_global_seed(seed)

    data_cfg = config["data"]
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))

    train_df, dev_df, test_df = load_canonical_dataset(
        str(config["dataset"]),
        train_path=data_cfg.get("train_path"),
        dev_path=data_cfg.get("dev_path"),
        test_path=data_cfg.get("test_path"),
    )
    x_train, y_train = xy_from_frame(train_df, text_column=text_column, label_column=label_column)
    x_dev, y_dev = xy_from_frame(dev_df, text_column=text_column, label_column=label_column)
    x_test, y_test = xy_from_frame(test_df, text_column=text_column, label_column=label_column)

    pipeline = build_tfidf_logistic_pipeline(config["model"], seed=seed)
    pipeline.fit(x_train, y_train)

    threshold_cfg = config["training"].get("threshold_tuning", {})
    # Tune on dev only, then keep one threshold for train, dev, and test.
    threshold = 0.5
    dev_threshold_metric = None
    if threshold_cfg.get("enabled", False):
        dev_scores = positive_class_proba(pipeline, x_dev)
        threshold, dev_threshold_metric = tune_binary_threshold(
            y_dev,
            dev_scores,
            metric=str(threshold_cfg.get("metric", "macro_f1")),
            min_threshold=float(threshold_cfg.get("min_threshold", 0.05)),
            max_threshold=float(threshold_cfg.get("max_threshold", 0.95)),
            step=float(threshold_cfg.get("step", 0.01)),
        )

    split_metrics: dict[str, Any] = {}
    for split_name, texts, labels in (
        ("train", x_train, y_train),
        ("dev", x_dev, y_dev),
        ("test", x_test, y_test),
    ):
        scores = positive_class_proba(pipeline, texts)
        preds = apply_threshold(scores, threshold)
        split_metrics[split_name] = classification_metrics(labels, preds)

    results_dir = resolve_results_dir(config)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "seed": seed,
        "config_path": str(config_path) if config_path is not None else None,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "primary_metric": config["training"].get("primary_metric", "macro_f1"),
        "threshold": threshold,
        "dev_threshold_macro_f1": dev_threshold_metric,
        "data": {
            "train_rows": len(train_df),
            "dev_rows": len(dev_df),
            "test_rows": len(test_df),
            "train_class_distribution": _split_class_distribution(y_train),
            "dev_class_distribution": _split_class_distribution(y_dev),
            "test_class_distribution": _split_class_distribution(y_test),
        },
        "metrics": split_metrics,
        "model": config["model"],
        "training": config["training"],
    }

    metrics_path = results_dir / "metrics.json"
    model_path = results_dir / "model.joblib"
    config_copy_path = results_dir / "config.resolved.json"

    metrics_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    config_copy_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, model_path)

    return results_dir
