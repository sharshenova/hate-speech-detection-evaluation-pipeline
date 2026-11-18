"""Export canonical DGHD train, dev, and test TSVs from a DGHD CSV input."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.dghd import load_dghd, split_by_column
from hsd.data.harmonize import build_dghd_canonical_splits, write_canonical_tsv
from hsd.data.paths import (
    DGHD_DEV_CANONICAL,
    DGHD_DEDUP,
    DGHD_PROCESSED_DIR,
    DGHD_TEST_CANONICAL,
    DGHD_TRAIN_CANONICAL,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DGHD_DEDUP,
        help="DGHD CSV path (raw or deduplicated; default: dedup export)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DGHD_PROCESSED_DIR,
        help="Directory for canonical TSV outputs",
    )
    args = parser.parse_args()

    frame = load_dghd(args.input)
    train_in, dev_in, test_in = split_by_column(frame)
    train_can, dev_can, test_can = build_dghd_canonical_splits(frame)

    train_path = write_canonical_tsv(train_can, args.output_dir / DGHD_TRAIN_CANONICAL.name)
    dev_path = write_canonical_tsv(dev_can, args.output_dir / DGHD_DEV_CANONICAL.name)
    test_path = write_canonical_tsv(test_can, args.output_dir / DGHD_TEST_CANONICAL.name)

    print(f"Input:  {args.input}")
    print(f"  train: {len(train_in):,} | dev: {len(dev_in):,} | test: {len(test_in):,}")
    print(f"Canonical train: {len(train_can):,}")
    print(f"Canonical dev:   {len(dev_can):,}")
    print(f"Canonical test:  {len(test_can):,}")
    print(f"Wrote {train_path}")
    print(f"Wrote {dev_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
