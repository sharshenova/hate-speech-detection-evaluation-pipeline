"""Train a hate-speech model from a YAML experiment config.

    python scripts/training/train.py --config configs/exp-001-baseline-lr.yaml --dataset ghc
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.training.baseline import run_baseline_lr_experiment
from hsd.utils.config import load_experiment_config


def _print_metrics_summary(results_dir: Path) -> None:
    import json

    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"Experiment: {metrics['experiment_id']} ({metrics['dataset']})")
    threshold = metrics.get("threshold")
    if threshold is not None:
        print(f"Threshold (dev-tuned): {threshold:.3f}")
    for split_name in ("train", "dev", "test"):
        split = metrics["metrics"][split_name]
        print(
            f"  {split_name:5s}  "
            f"acc={split['accuracy']:.4f}  "
            f"precision={split['precision_macro']:.4f}  "
            f"recall={split['recall_macro']:.4f}  "
            f"f1_macro={split['f1_macro']:.4f}"
        )
    print(f"Wrote {results_dir / 'metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config (e.g. configs/exp-001-baseline-lr.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset id in config (ghc, dghd, ktc)",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        nargs="?",
        const=30,
        default=0,
        metavar="STEPS",
        help="Transformers only: time STEPS optimizer steps (default 30) and print "
        "a full-training estimate instead of training",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.dataset is not None:
        config["dataset"] = args.dataset
    model_type = str(config["model"].get("type", ""))

    if model_type == "tfidf_logistic_regression":
        results_dir = run_baseline_lr_experiment(config, config_path=args.config)
        _print_metrics_summary(results_dir)
        return

    if model_type in {
        "bert_sequence_classifier",
        "roberta_sequence_classifier",
        "fbert_sequence_classifier",
        "deberta_sequence_classifier",
    }:
        # Import backends only when selected so baseline runs do not need their dependencies.
        from hsd.training.transformer import run_transformer_experiment

        results_dir = run_transformer_experiment(
            config,
            config_path=args.config,
            benchmark_steps=args.benchmark,
        )
        if args.benchmark == 0:
            _print_metrics_summary(results_dir)
        return

    if model_type == "lora_causal_lm":
        from hsd.training.lora_lm import run_lora_experiment

        results_dir = run_lora_experiment(config, config_path=args.config)
        _print_metrics_summary(results_dir)
        return

    msg = f"Unsupported model type: {model_type!r}"
    raise ValueError(msg)


if __name__ == "__main__":
    main()
