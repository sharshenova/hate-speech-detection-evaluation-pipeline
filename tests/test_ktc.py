"""Tests for KTC subsample deduplication."""

from __future__ import annotations

import pandas as pd

from hsd.data.ktc import deduplicate_ktc_frame, load_ktc_subsample, subsample_summary
from hsd.data.schema import KTC_SUBSAMPLE_COLUMNS


def test_load_ktc_subsample_validates_columns(tmp_path) -> None:
    path = tmp_path / "ktc_subsample.csv"
    pd.DataFrame(
        [{"source": "hatexplain", "text": "x", "hate": 0, "target": -1}]
    ).to_csv(path, index=False)
    frame = load_ktc_subsample(path)
    assert list(frame.columns) == list(KTC_SUBSAMPLE_COLUMNS)


def test_deduplicate_ktc_keeps_one_full_duplicate() -> None:
    frame = pd.DataFrame(
        [
            {
                "source": "hatexplain",
                "text": "Same text",
                "hate": 0,
                "target": -1,
            },
            {
                "source": "mmhs150k",
                "text": "Same text",
                "hate": 0,
                "target": -1,
            },
        ]
    )
    result = deduplicate_ktc_frame(frame)
    assert len(result) == 1
    assert result.iloc[0]["source"] == "hatexplain"


def test_deduplicate_ktc_drops_all_on_conflict() -> None:
    frame = pd.DataFrame(
        [
            {
                "source": "hatexplain",
                "text": "Conflict",
                "hate": 0,
                "target": -1,
            },
            {
                "source": "mmhs150k",
                "text": "Conflict",
                "hate": 1,
                "target": 0,
            },
        ]
    )
    result = deduplicate_ktc_frame(frame)
    assert result.empty


def test_subsample_summary_totals() -> None:
    frame = pd.DataFrame(
        [
            {"source": "hatexplain", "text": "a", "hate": 1, "target": 0},
            {"source": "mmhs150k", "text": "b", "hate": 0, "target": -1},
        ]
    )
    summary = subsample_summary(frame)
    assert int(summary.loc[summary["source"] == "total", "rows"].iloc[0]) == 2
