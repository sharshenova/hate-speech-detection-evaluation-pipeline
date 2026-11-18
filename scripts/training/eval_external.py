"""Evaluate a saved checkpoint on HateCheck or ImplicitHate.

Thresholds and decode settings come from in-domain training only. Examples:

    python scripts/training/eval_external.py \\
        --config configs/exp-005a-deberta-large-lr2e-5.yaml --dataset ghc \\
        --eval-dataset hatecheck

    python scripts/training/eval_external.py \\
        --config configs/exp-015-qwen35-dghd.yaml --dataset dghd \\
        --eval-dataset ihc --logit-threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.evaluation.external_eval import run_external_eval, write_external_eval_artifacts
from hsd.utils.config import load_experiment_config, resolve_results_dir
from hsd.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Training experiment config")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Train corpus override (ghc, dghd, ktc); default from config",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="hatecheck",
        help="External eval corpus id (hatecheck or ihc; default: hatecheck)",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=None,
        help="Optional path to eval TSV",
    )
    parser.add_argument(
        "--logit-threshold",
        action="store_true",
        help="LoRA only: use saved logit threshold (metrics_reeval_logit.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.dataset is not None:
        config["dataset"] = args.dataset
    set_global_seed(int(config["seed"]))

    artifact = run_external_eval(
        args.config,
        eval_dataset=args.eval_dataset,
        eval_path=args.eval_path,
        logit_threshold=args.logit_threshold,
        limit=args.limit,
        train_dataset=args.dataset,
    )
    results_dir = resolve_results_dir(config)
    metrics_path, preds_path = write_external_eval_artifacts(
        artifact,
        results_dir,
        eval_dataset=args.eval_dataset,
    )

    metrics = artifact["metrics"]
    print(
        f"{artifact['experiment_id']} ({artifact['train_dataset']} -> {args.eval_dataset}): "
        f"n={artifact['eval_rows']:,}  {artifact['decode_mode']}  "
        f"f1_macro={metrics['f1_macro']:.4f}  FP={artifact['fp']}  FN={artifact['fn']}",
        flush=True,
    )
    if artifact.get("parse_failures") is not None:
        print(f"  parse_failures={artifact['parse_failures']}", flush=True)
    print(f"Wrote {metrics_path}")
    if preds_path is not None:
        print(f"Wrote {preds_path}")


if __name__ == "__main__":
    main()
