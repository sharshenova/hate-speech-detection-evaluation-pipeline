"""Prepare GHC train/test splits with majority-voted annotation labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.ghc import build_labeled_ghc_splits, write_labeled_ghc_splits
from hsd.data.paths import GHC_PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GHC_PROCESSED_DIR,
        help="Directory for labeled TSV outputs",
    )
    args = parser.parse_args()

    train_out, test_out = write_labeled_ghc_splits(args.output_dir)
    train, test = build_labeled_ghc_splits()

    print(f"Wrote {train_out} ({len(train)} rows)")
    print(f"Wrote {test_out} ({len(test)} rows)")
    added = [col for col in train.columns if col not in ("text", "hd", "cv", "vo")]
    print("Added columns:", ", ".join(added))


if __name__ == "__main__":
    main()
