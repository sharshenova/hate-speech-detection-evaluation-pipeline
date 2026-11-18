"""Deduplicate KTC and write processed CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hsd.data.ktc import deduplicate_ktc_frame, subsample_summary, write_deduplicated_ktc_subsample
from hsd.data.paths import KTC_SUBSAMPLE, KTC_SUBSAMPLE_DEDUP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=KTC_SUBSAMPLE,
        help="KTC subsample CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=KTC_SUBSAMPLE_DEDUP,
        help="Deduplicated KTC subsample output CSV",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    deduped = deduplicate_ktc_frame(frame)
    output_path = write_deduplicated_ktc_subsample(input_path=args.input, output_path=args.output)

    print(f"Input:  {args.input} ({len(frame):,} rows)")
    print(f"Output: {output_path} ({len(deduped):,} rows)")
    print(f"Removed: {len(frame) - len(deduped):,} rows")
    print("\nBefore:")
    print(subsample_summary(frame).to_string(index=False))
    print("\nAfter:")
    print(subsample_summary(deduped).to_string(index=False))


if __name__ == "__main__":
    main()
