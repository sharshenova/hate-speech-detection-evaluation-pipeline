"""Export canonical Implicit Hate Corpus eval TSV (Twitter Stage 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsd.data.ihc import (
    IHC_CLASS_EXPLICIT,
    IHC_CLASS_IMPLICIT,
    IHC_CLASS_NOT_HATE,
    prepare_ihc_eval,
)
from hsd.data.paths import IHC_EVAL, IHC_STG1, IHC_STG1_POSTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, default=IHC_STG1, help="Stage-1 ID TSV")
    parser.add_argument("--posts", type=Path, default=IHC_STG1_POSTS, help="Stage-1 posts TSV")
    parser.add_argument("--output", type=Path, default=IHC_EVAL, help="Canonical eval TSV")
    args = parser.parse_args()

    sliced, canonical = prepare_ihc_eval(
        ids_path=args.ids,
        posts_path=args.posts,
        output_path=args.output,
    )

    n_explicit = int((canonical["ihc_class"] == IHC_CLASS_EXPLICIT).sum())
    n_implicit = int((canonical["ihc_class"] == IHC_CLASS_IMPLICIT).sum())
    n_not = int((canonical["ihc_class"] == IHC_CLASS_NOT_HATE).sum())
    hate = int(canonical["hate"].sum())

    print(f"Input:  {args.ids} + {args.posts} ({len(sliced):,} Twitter Stage-1 rows)")
    print("Filter: drop GAB_ / SAP_ IDs; drop empty text")
    print(
        f"Labels: hate={hate:,} (explicit={n_explicit:,} + implicit={n_implicit:,}) | "
        f"non-hate={n_not:,}"
    )
    print(f"Wrote {args.output} ({len(canonical):,} rows)")


if __name__ == "__main__":
    main()
