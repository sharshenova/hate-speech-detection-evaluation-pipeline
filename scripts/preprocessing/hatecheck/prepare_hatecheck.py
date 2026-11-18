"""Export canonical HateCheck eval TSV (full functional test suite)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.hatecheck import prepare_hatecheck_eval
from hsd.data.paths import HATECHECK_EVAL, HATECHECK_RAW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=HATECHECK_RAW,
        help="HateCheck test_suite_cases.csv path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HATECHECK_EVAL,
        help="Canonical eval TSV output path",
    )
    args = parser.parse_args()

    sliced, canonical = prepare_hatecheck_eval(
        input_path=args.input,
        output_path=args.output,
    )

    hate = int(canonical["hate"].sum())
    non_hate = len(canonical) - hate
    n_funcs = int(canonical["hate_function"].nunique())
    n_targets = int(canonical["target_raw"].dropna().nunique())

    print(f"Input:  {args.input} ({len(sliced):,} cases)")
    print("Filter: full suite (no subsample); empty text dropped if any")
    print(f"Labels: hate={hate:,} | non-hate={non_hate:,}")
    print(f"Functionalities: {n_funcs} | target_ident values: {n_targets}")
    print(f"Wrote {args.output} ({len(canonical):,} rows)")


if __name__ == "__main__":
    main()
