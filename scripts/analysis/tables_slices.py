"""Layer 3: in-domain slices by target, hate function, GHC framing, KTC source."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    SIX_CHECKPOINTS,
    SLICE_COLUMNS,
    TABLES_DIR,
    TARGET_LABELS,
    join_eval_preds,
    load_pred_tsv,
    results_dir,
    row_from_preds,
    write_tsv,
)
from hsd.data.canonical import load_canonical_dataset
from hsd.data.schema import TARGET_FLAG_COLUMNS


def _attach(spec: dict[str, str]) -> pd.DataFrame:
    _, _, test = load_canonical_dataset(spec["dataset"])
    preds = load_pred_tsv(results_dir(spec["results_subdir"]) / "predictions_test.tsv")
    return join_eval_preds(test, preds)


def _slice_row(spec: dict[str, str], frame: pd.DataFrame, stratum: str) -> dict:
    row = row_from_preds(
        frame["hate"],
        frame["y_pred"],
        layer="3-slice",
        train_corpus=spec["train_corpus"],
        eval_set="test",
        model=spec["model"],
        decode=spec["decode"],
        extra={"stratum": stratum},
    )
    return {col: row.get(col, "") for col in SLICE_COLUMNS}


def main() -> None:
    by_tgt: list[dict] = []
    by_function: list[dict] = []
    by_framing: list[dict] = []
    by_source: list[dict] = []

    for spec in SIX_CHECKPOINTS:
        frame = _attach(spec)
        for col in TARGET_FLAG_COLUMNS:
            part = frame.loc[frame[col] == 1]
            if part.empty:
                continue
            by_tgt.append(_slice_row(spec, part, TARGET_LABELS.get(col, col)))

        if spec["dataset"] == "dghd" and "hate_function" in frame.columns:
            # Hate-function labels describe positive examples, so non-hate rows
            # do not belong to any function slice.
            hate = frame.loc[frame["hate"] == 1]
            for name, part in hate.groupby(hate["hate_function"].fillna("(missing)"), sort=True):
                by_function.append(_slice_row(spec, part, str(name)))

        if spec["dataset"] == "ghc":
            hate = frame.loc[frame["hate"] == 1]
            explicit = hate["framing_explicit"].fillna(0).astype(int) == 1
            implicit = hate["framing_implicit"].fillna(0).astype(int) == 1
            # Flags are kept as independent slices. "Neither" means that both
            # annotations are off, not that the example is non-hate.
            masks = {
                "explicit": explicit,
                "implicit": implicit,
                "neither": (~explicit) & (~implicit),
            }
            for name, mask in masks.items():
                part = hate.loc[mask]
                if not part.empty:
                    by_framing.append(_slice_row(spec, part, name))

        if spec["dataset"] == "ktc" and "source" in frame.columns:
            # Keep the full KTC result beside source subsets as their reference.
            by_source.append(_slice_row(spec, frame, "overall"))
            for name, part in frame.groupby("source", sort=True):
                by_source.append(_slice_row(spec, part, str(name)))

    write_tsv(pd.DataFrame(by_tgt)[list(SLICE_COLUMNS)], TABLES_DIR / "indomain_by_target.tsv")
    write_tsv(
        pd.DataFrame(by_function)[list(SLICE_COLUMNS)],
        TABLES_DIR / "indomain_by_hate_function.tsv",
    )
    write_tsv(
        pd.DataFrame(by_framing)[list(SLICE_COLUMNS)],
        TABLES_DIR / "indomain_by_framing.tsv",
    )
    write_tsv(
        pd.DataFrame(by_source)[list(SLICE_COLUMNS)],
        TABLES_DIR / "indomain_ktc_by_source.tsv",
    )


if __name__ == "__main__":
    main()
