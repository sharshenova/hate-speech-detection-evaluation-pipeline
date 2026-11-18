"""Re-evaluate a saved LoRA adapter on dev/test (no retraining).

Usage:
    python scripts/training/eval_lora_adapter.py --config configs/exp-008-qwen-lora.yaml --dataset ghc
    python scripts/training/eval_lora_adapter.py --config configs/exp-014-qwen35-ktc.yaml --dataset ktc \\
        --logit-threshold
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hsd.data.canonical import load_canonical_dataset, xy_from_frame
from hsd.training.lora_lm import (
    _evaluate_split,
    _load_inference_peft_model,
    _resolve_model_settings,
    evaluate_lora_logit_threshold,
    resolve_lora_prompt_config,
)
from hsd.training.transformer import select_device
from hsd.utils.config import load_experiment_config, resolve_results_dir
from hsd.utils.seed import set_global_seed


def _fp_fn(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))
    return fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per split for smoke tests",
    )
    parser.add_argument(
        "--logit-threshold",
        action="store_true",
        help="Score first-token logits, tune threshold on dev, apply to all splits",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.dataset is not None:
        config["dataset"] = args.dataset

    set_global_seed(int(config["seed"]))
    prompt_cfg = resolve_lora_prompt_config(config)
    device = select_device(str(config["model"].get("device", "auto")))
    model_cfg = _resolve_model_settings(config["model"], device)
    model_cfg["_device"] = device

    results_dir = resolve_results_dir(config)
    adapter_dir = results_dir / "model"
    if not adapter_dir.is_dir():
        msg = f"No saved adapter at {adapter_dir}"
        raise SystemExit(msg)

    data_cfg = config.get("data", {})
    text_column = str(data_cfg.get("text_column", "text"))
    label_column = str(data_cfg.get("label_column", "hate"))
    _, dev_df, test_df = load_canonical_dataset(str(config["dataset"]))
    x_dev, y_dev = xy_from_frame(dev_df, text_column=text_column, label_column=label_column)
    x_test, y_test = xy_from_frame(test_df, text_column=text_column, label_column=label_column)
    if args.limit is not None:
        x_dev, y_dev = x_dev[: args.limit], y_dev[: args.limit]
        x_test, y_test = x_test[: args.limit], y_test[: args.limit]

    model, tokenizer = _load_inference_peft_model(model_cfg, adapter_dir, device)

    if args.logit_threshold:
        # Keep this optional scoring method in a separate artifact from greedy re-evaluation.
        threshold_cfg = dict(config.get("training", {}).get("threshold_tuning", {}))
        threshold_cfg.setdefault("enabled", True)
        eval_config = dict(config)
        eval_config["training"] = dict(config.get("training", {}))
        eval_config["training"]["threshold_tuning"] = threshold_cfg

        result = evaluate_lora_logit_threshold(
            model,
            tokenizer,
            prompt_cfg,
            model_cfg,
            eval_config,
            splits={
                "dev": (x_dev, np.asarray(y_dev, dtype=int)),
                "test": (x_test, np.asarray(y_test, dtype=int)),
            },
        )
        artifact = {
            "experiment_id": config["experiment_id"],
            "dataset": config["dataset"],
            "note": "Logit-based re-eval with dev-tuned hate threshold",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "adapter_dir": str(adapter_dir),
            **result,
        }
        out_path = results_dir / "metrics_reeval_logit.json"
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        for split_name in ("dev", "test"):
            labels = np.asarray(y_dev if split_name == "dev" else y_test, dtype=int)
            scores = np.asarray(result["scores"][split_name], dtype=float)
            threshold = float(result["threshold"])
            preds = (scores >= threshold).astype(int)
            fp, fn = _fp_fn(labels, preds)
            metrics = result["metrics"][split_name]
            print(
                f"  {split_name}: f1_macro={metrics['f1_macro']:.4f} "
                f"FP={fp} FN={fn} threshold={threshold:.3f}",
                flush=True,
            )
        return

    split_metrics: dict[str, dict] = {}
    split_parse_failures: dict[str, int] = {}
    for split_name, texts, labels in (
        ("dev", x_dev, np.asarray(y_dev, dtype=int)),
        ("test", x_test, np.asarray(y_test, dtype=int)),
    ):
        print(f"Re-eval: {split_name} ({len(texts)} rows)...", flush=True)
        metrics, _, parse_failures = _evaluate_split(
            model, tokenizer, prompt_cfg, texts, labels, model_cfg
        )
        split_metrics[split_name] = metrics
        split_parse_failures[split_name] = parse_failures
        print(
            f"  {split_name}: acc={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f} parse_failures={parse_failures}",
            flush=True,
        )

    artifact = {
        "experiment_id": config["experiment_id"],
        "dataset": config["dataset"],
        "note": "Re-evaluation of saved adapter (generative greedy decode)",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "adapter_dir": str(adapter_dir),
        "parse_failures": split_parse_failures,
        "metrics": split_metrics,
    }
    out_path = results_dir / "metrics_reeval.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
