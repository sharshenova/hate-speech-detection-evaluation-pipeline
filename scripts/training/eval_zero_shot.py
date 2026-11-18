"""Run zero-shot LLM hate-speech evaluation from a YAML experiment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsd.training.zero_shot import load_zero_shot_config, run_zero_shot_experiment


def _print_metrics_summary(results_dir: Path) -> None:
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    split_name = str(metrics["eval_split"])
    split = metrics["metrics"][split_name]
    print(f"Experiment: {metrics['experiment_id']} ({metrics['dataset']}, {split_name})")
    print(f"Prompt: {metrics['prompt_id']}")
    print(f"Parse failures: {metrics['parse_failures']} / {metrics['eval_rows']}")
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
        help="Path to zero-shot experiment YAML (e.g. configs/exp-007a-zero-shot-minimal.yaml)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests",
    )
    args = parser.parse_args()

    config = load_zero_shot_config(args.config)
    results_dir = run_zero_shot_experiment(
        config,
        config_path=args.config,
        limit=args.limit,
    )
    _print_metrics_summary(results_dir)


if __name__ == "__main__":
    main()
