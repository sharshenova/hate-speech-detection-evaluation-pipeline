"""Export canonical GHC train, dev, and test TSVs from deduplicated labeled splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.ghc import load_labeled_ghc_split
from hsd.data.harmonize import (
    GHC_DEV_FRACTION,
    SPLIT_RANDOM_SEED,
    build_ghc_canonical_splits,
    write_canonical_tsv,
)
from hsd.data.paths import (
    GHC_DEV_CANONICAL,
    GHC_PROCESSED_DIR,
    GHC_TEST_CANONICAL,
    GHC_TEST_LABELED_DEDUP,
    GHC_TRAIN_CANONICAL,
    GHC_TRAIN_LABELED_DEDUP,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-in",
        type=Path,
        default=GHC_TRAIN_LABELED_DEDUP,
        help="Deduplicated labeled GHC train TSV",
    )
    parser.add_argument(
        "--test-in",
        type=Path,
        default=GHC_TEST_LABELED_DEDUP,
        help="Deduplicated labeled GHC test TSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GHC_PROCESSED_DIR,
        help="Directory for canonical TSV outputs",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=GHC_DEV_FRACTION,
        help="Fraction of upstream train rows assigned to dev (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_RANDOM_SEED,
        help="Random seed for train/dev split (default: 42)",
    )
    args = parser.parse_args()

    train_in = load_labeled_ghc_split(args.train_in)
    test_in = load_labeled_ghc_split(args.test_in)

    train_can, dev_can, test_can = build_ghc_canonical_splits(
        train_in,
        test_in,
        dev_fraction=args.dev_fraction,
        random_state=args.seed,
    )
    train_path = write_canonical_tsv(train_can, args.output_dir / GHC_TRAIN_CANONICAL.name)
    dev_path = write_canonical_tsv(dev_can, args.output_dir / GHC_DEV_CANONICAL.name)
    test_path = write_canonical_tsv(test_can, args.output_dir / GHC_TEST_CANONICAL.name)

    print(f"Input train: {len(train_in):,} rows")
    print(f"  canonical train: {len(train_can):,}")
    print(f"  canonical dev:   {len(dev_can):,}")
    print(f"Input test:  {len(test_in):,} rows -> canonical test: {len(test_can):,}")
    print(f"Wrote {train_path}")
    print(f"Wrote {dev_path}")
    print(f"Wrote {test_path}")
    print(f"dev_fraction={args.dev_fraction}, seed={args.seed}")


if __name__ == "__main__":
    main()
