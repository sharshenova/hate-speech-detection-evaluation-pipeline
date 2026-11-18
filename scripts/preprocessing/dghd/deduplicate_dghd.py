"""Deduplicate DGHD and write a single processed CSV (same schema as raw)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.dghd import deduplicate_dghd_frame, load_dghd, split_by_column, write_deduplicated_dghd
from hsd.data.paths import DGHD_DEDUP, DGHD_RAW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DGHD_RAW,
        help="DGHD v0.2.3 CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DGHD_DEDUP,
        help="Deduplicated DGHD CSV path (same columns as input)",
    )
    args = parser.parse_args()

    frame = load_dghd(args.input)
    train, dev, test = split_by_column(frame)
    deduped = deduplicate_dghd_frame(frame)
    output_path = write_deduplicated_dghd(input_path=args.input, output_path=args.output)

    train_out, dev_out, test_out = split_by_column(deduped)
    print(f"Input:  {len(frame):,} rows (train={len(train):,} dev={len(dev):,} test={len(test):,})")
    print(f"Output: {len(deduped):,} rows (train={len(train_out):,} dev={len(dev_out):,} test={len(test_out):,})")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
