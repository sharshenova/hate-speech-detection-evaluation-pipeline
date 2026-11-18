"""Deduplicate labeled GHC train/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.ghc import (
    deduplicate_labeled_ghc_splits,
    load_labeled_ghc_split,
    write_deduplicated_ghc_splits,
)
from hsd.data.paths import (
    GHC_PROCESSED_DIR,
    GHC_TEST_LABELED,
    GHC_TRAIN_LABELED,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-in",
        type=Path,
        default=GHC_TRAIN_LABELED,
        help="Input labeled train TSV",
    )
    parser.add_argument(
        "--test-in",
        type=Path,
        default=GHC_TEST_LABELED,
        help="Input labeled test TSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GHC_PROCESSED_DIR,
        help="Directory for deduplicated TSV outputs",
    )
    args = parser.parse_args()

    train_in = load_labeled_ghc_split(args.train_in)
    test_in = load_labeled_ghc_split(args.test_in)
    train_out, test_out = write_deduplicated_ghc_splits(
        train_in=args.train_in,
        test_in=args.test_in,
        output_dir=args.output_dir,
    )
    train, test = deduplicate_labeled_ghc_splits(train_in, test_in)

    print(f"Train: {len(train_in):,} -> {len(train):,} rows")
    print(f"Test:  {len(test_in):,} -> {len(test):,} rows")
    print(f"Wrote {train_out}")
    print(f"Wrote {test_out}")


if __name__ == "__main__":
    main()
