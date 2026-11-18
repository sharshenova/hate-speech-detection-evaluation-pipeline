"""Rebuild every thesis TSV table from saved predictions (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tables_fairness
import tables_hatecheck
import tables_ihc
import tables_indomain
import tables_slices
import tables_stats


def main() -> None:
    tables_indomain.main()
    tables_stats.main()
    tables_slices.main()
    tables_hatecheck.main()
    tables_ihc.main()
    tables_fairness.main()


if __name__ == "__main__":
    main()
