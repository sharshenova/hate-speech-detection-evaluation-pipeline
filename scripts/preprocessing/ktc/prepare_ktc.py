"""Export canonical KTC train, dev, and test TSVs from a deduplicated subsample."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.harmonize import (
    KTC_TRAIN_FRACTION,
    SPLIT_RANDOM_SEED,
    build_ktc_canonical_splits,
    ktc_split_distribution_report,
    write_canonical_tsv,
)
from hsd.data.ktc import load_ktc_subsample, subsample_summary
from hsd.data.paths import (
    KTC_DEV_CANONICAL,
    KTC_PROCESSED_DIR,
    KTC_SUBSAMPLE_DEDUP,
    KTC_TEST_CANONICAL,
    KTC_TRAIN_CANONICAL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=KTC_SUBSAMPLE_DEDUP,
        help="Deduplicated KTC subsample CSV (default: ktc_subsample_dedup.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KTC_PROCESSED_DIR,
        help="Directory for canonical TSV outputs",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=KTC_TRAIN_FRACTION,
        help="Fraction assigned to train before dev/test holdout split (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_RANDOM_SEED,
        help="Random seed for stratified splits (default: 42)",
    )
    args = parser.parse_args()

    frame = load_ktc_subsample(args.input)
    train_can, dev_can, test_can = build_ktc_canonical_splits(
        frame,
        train_fraction=args.train_fraction,
        random_state=args.seed,
    )
    train_path = write_canonical_tsv(train_can, args.output_dir / KTC_TRAIN_CANONICAL.name)
    dev_path = write_canonical_tsv(dev_can, args.output_dir / KTC_DEV_CANONICAL.name)
    test_path = write_canonical_tsv(test_can, args.output_dir / KTC_TEST_CANONICAL.name)
    split_report = ktc_split_distribution_report(train_can, dev_can, test_can)

    print(f"Input: {args.input} ({len(frame):,} rows)")
    print("Input by source:")
    print(subsample_summary(frame).to_string(index=False))
    print()
    print("Canonical splits:")
    print(split_report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"Wrote {train_path}")
    print(f"Wrote {dev_path}")
    print(f"Wrote {test_path}")
    print(f"train_fraction={args.train_fraction}, seed={args.seed}")


if __name__ == "__main__":
    main()
