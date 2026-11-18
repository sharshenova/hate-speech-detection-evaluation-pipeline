"""Score saved checkpoints on HateCheck or IHC."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsd.data.canonical import load_eval_dataset, xy_from_frame
from hsd.evaluation.metrics import apply_threshold, classification_metrics
from hsd.training.lora_lm import (
    _evaluate_split,
    _load_inference_peft_model,
    _resolve_model_settings,
    predict_hate_scores,
    resolve_lora_prompt_config,
)
from hsd.training.transformer import _make_loader, _predict_proba, _TextDataset, select_device
from hsd.utils.config import load_experiment_config, resolve_results_dir

_PRED_META_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "text",
    "target_raw",
    "hate_function",
    "ihc_class",
)


def resolve_checkpoint_threshold(results_dir: Path, *, logit: bool = False) -> float:
    """Load the threshold chosen on in-domain development data.

    External labels never affect this value, so external evaluation stays frozen.
    """
    if logit:
        path = results_dir / "metrics_reeval_logit.json"
        if not path.is_file():
            msg = (
                f"Missing {path}; run eval_lora_adapter.py --logit-threshold first "
                "or omit --logit-threshold for greedy decode."
            )
            raise FileNotFoundError(msg)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["threshold"])

    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        msg = f"Missing training metrics: {metrics_path}"
        raise FileNotFoundError(msg)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = payload.get("threshold")
    if threshold is None:
        return 0.5
    return float(threshold)


def _fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))
    return fp, fn


def evaluate_deberta_external(
    config: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Score a saved DeBERTa checkpoint using its frozen in-domain threshold."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    results_dir = results_dir or resolve_results_dir(config)
    model_dir = results_dir / "model"
    if not model_dir.is_dir():
        msg = f"No saved model at {model_dir}"
        raise FileNotFoundError(msg)

    data_cfg = config.get("data", {})
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))
    texts, labels = xy_from_frame(eval_df, text_column=text_column, label_column=label_column)
    y_true = np.asarray(labels, dtype=int)

    threshold = resolve_checkpoint_threshold(results_dir, logit=False)
    device = select_device(str(config["model"].get("device", "auto")))
    max_length = int(config["model"].get("max_length", 128))
    batch_size = int(config["model"].get("cuda", {}).get("batch_size", 32))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    enc = dict(tokenizer(list(texts), truncation=True, max_length=max_length))
    loader = _make_loader(
        _TextDataset(enc, y_true),
        tokenizer,
        batch_size=batch_size,
        shuffle=False,
        seed=int(config["seed"]),
    )
    scores, _, _ = _predict_proba(model, loader, device)
    y_pred = apply_threshold(scores, threshold)
    metrics = classification_metrics(y_true, y_pred)
    fp, fn = _fp_fn(y_true, y_pred)

    return {
        "decode_mode": f"threshold={threshold:.3f}",
        "threshold": threshold,
        "metrics": metrics,
        "fp": fp,
        "fn": fn,
        "scores": scores.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def evaluate_lora_external(
    config: dict[str, Any],
    eval_df: pd.DataFrame,
    *,
    results_dir: Path | None = None,
    logit_threshold: bool = False,
) -> dict[str, Any]:
    """Score a saved LoRA adapter without tuning on the external frame.

    Logit decoding uses the saved in-domain threshold; greedy decoding uses text labels.
    """
    results_dir = results_dir or resolve_results_dir(config)
    adapter_dir = results_dir / "model"
    if not adapter_dir.is_dir():
        msg = f"No saved adapter at {adapter_dir}"
        raise FileNotFoundError(msg)

    data_cfg = config.get("data", {})
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))
    texts, labels = xy_from_frame(eval_df, text_column=text_column, label_column=label_column)
    y_true = np.asarray(labels, dtype=int)

    prompt_cfg = resolve_lora_prompt_config(config)
    device = select_device(str(config["model"].get("device", "auto")))
    model_cfg = _resolve_model_settings(config["model"], device)
    model_cfg["_device"] = device

    model, tokenizer = _load_inference_peft_model(model_cfg, adapter_dir, device)

    if logit_threshold:
        threshold = resolve_checkpoint_threshold(results_dir, logit=True)
        scores = predict_hate_scores(model, tokenizer, prompt_cfg, list(texts), model_cfg)
        y_pred = apply_threshold(scores, threshold)
        metrics = classification_metrics(y_true, y_pred)
        fp, fn = _fp_fn(y_true, y_pred)
        parse_failures = 0
        decode_mode = f"logit_threshold={threshold:.3f}"
    else:
        metrics, y_pred, parse_failures = _evaluate_split(
            model, tokenizer, prompt_cfg, list(texts), y_true, model_cfg
        )
        y_pred = np.asarray(y_pred, dtype=int)
        fp, fn = _fp_fn(y_true, y_pred)
        threshold = None
        scores = None
        decode_mode = "greedy"

    result: dict[str, Any] = {
        "decode_mode": decode_mode,
        "threshold": threshold,
        "metrics": metrics,
        "fp": fp,
        "fn": fn,
        "parse_failures": parse_failures,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    if scores is not None:
        result["scores"] = scores.tolist()
    return result


def run_external_eval(
    config_path: str | Path,
    *,
    eval_dataset: str,
    eval_path: str | Path | None = None,
    logit_threshold: bool = False,
    limit: int | None = None,
    train_dataset: str | None = None,
) -> dict[str, Any]:
    """Load checkpoint + external eval data; return metrics artifact dict."""
    config = load_experiment_config(config_path)
    if train_dataset is not None:
        config["dataset"] = train_dataset
    train_dataset = str(config["dataset"])
    eval_df = load_eval_dataset(eval_dataset, path=eval_path)
    if limit is not None:
        eval_df = eval_df.head(limit).copy()

    model_type = str(config["model"].get("type", ""))
    results_dir = resolve_results_dir(config)

    if model_type == "deberta_sequence_classifier":
        eval_result = evaluate_deberta_external(config, eval_df, results_dir=results_dir)
    elif model_type == "lora_causal_lm":
        eval_result = evaluate_lora_external(
            config,
            eval_df,
            results_dir=results_dir,
            logit_threshold=logit_threshold,
        )
    else:
        msg = f"Unsupported model type for external eval: {model_type!r}"
        raise ValueError(msg)

    artifact: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "eval_rows": len(eval_df),
        "note": "External eval; threshold from in-domain dev only (not refit on eval set)",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "results_dir": str(results_dir),
        **eval_result,
    }

    artifact["_eval_meta"] = {
        col: eval_df[col].tolist() for col in _PRED_META_COLUMNS if col in eval_df.columns
    }
    return artifact


def write_external_eval_artifacts(
    artifact: dict[str, Any],
    results_dir: Path,
    *,
    eval_dataset: str,
) -> tuple[Path, Path | None]:
    """Write external metrics and row predictions beside the training artifacts.

    The JSON keeps summary results while the TSV keeps predictions and available eval metadata.
    """
    metrics_path = results_dir / f"metrics_eval_{eval_dataset}.json"
    metrics_payload = {
        key: value for key, value in artifact.items() if key not in {"scores", "_eval_meta"}
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    preds_path: Path | None = None
    if "y_pred" in artifact:
        preds_path = results_dir / f"predictions_eval_{eval_dataset}.tsv"
        frame = pd.DataFrame(
            {
                "y_true": artifact["y_true"],
                "y_pred": artifact["y_pred"],
            }
        )
        if "scores" in artifact:
            frame["score"] = artifact["scores"]
        meta = artifact.get("_eval_meta") or {}
        for col, values in meta.items():
            if len(values) == len(frame):
                frame[col] = values
        frame.to_csv(preds_path, sep="\t", index=False)

    return metrics_path, preds_path
