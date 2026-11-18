"""Tests for GHC annotation aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from hsd.data.ghc import (
    TARGET_LABEL_COLUMNS,
    aggregate_annotations_majority,
    derive_hate_label,
    enrich_ghc_split,
    majority_vote,
)
from hsd.data.paths import GHC_ANNOTATIONS_RAW, GHC_TEST_RAW, GHC_TRAIN_RAW


def test_derive_hate_label_from_hd_or_cv() -> None:
    hd = pd.Series([1, 0, 0, 1])
    cv = pd.Series([0, 1, 0, 1])
    assert derive_hate_label(hd, cv).tolist() == [1, 1, 0, 1]


def test_majority_vote_empty_is_zero() -> None:
    assert majority_vote(pd.Series([pd.NA, pd.NA])) == 0


def test_majority_vote_picks_most_common() -> None:
    assert majority_vote(pd.Series([1, 1, 0])) == 1
    assert majority_vote(pd.Series([0, 1, 1, 1])) == 1


def test_majority_vote_tie_goes_to_highest_count() -> None:
    assert majority_vote(pd.Series([0, 1])) in (0, 1)


def test_aggregate_annotations_majority_on_synthetic_rows() -> None:
    annotations = pd.DataFrame(
        {
            "ID": [1, 1, 1, 2, 2, 2],
            "Annotator": [1, 2, 3, 1, 2, 3],
            "Text": ["Hello", "Hello", "Hello", "Other", "Other", "Other"],
            "Hate": [1, 1, 0, 0, 0, 0],
            "HD": [1, 0, 0, 0, 0, 0],
            "CV": [0, 0, 0, 0, 0, 0],
            "VO": [0, 0, 0, 1, 1, 0],
            "REL": [1, 1, 0, pd.NA, pd.NA, pd.NA],
            "RAE": [0, 1, 1, pd.NA, pd.NA, pd.NA],
            "SXO": [0, 0, 0, pd.NA, pd.NA, pd.NA],
            "GEN": [0, 0, 0, pd.NA, pd.NA, pd.NA],
            "IDL": [0, 0, 0, pd.NA, pd.NA, pd.NA],
            "NAT": [0, 0, 0, pd.NA, pd.NA, pd.NA],
            "POL": [0, 0, 0, pd.NA, pd.NA, pd.NA],
            "MPH": [0, 0, 0, pd.NA, pd.NA, pd.NA],
            "EX": [1, 0, 0, pd.NA, pd.NA, pd.NA],
            "IM": [0, 0, 0, pd.NA, pd.NA, pd.NA],
        }
    )

    aggregated = aggregate_annotations_majority(annotations)
    hello = aggregated.loc[aggregated["text_norm"] == "hello"].iloc[0]
    other = aggregated.loc[aggregated["text_norm"] == "other"].iloc[0]

    assert hello["rel"] == 1
    assert hello["rae"] == 0
    assert hello["n_annotators"] == 3
    assert pd.isna(other["rel"])
    assert pd.isna(other["ex"])


def test_enrich_ghc_split_clears_targets_on_non_hate() -> None:
    split = pd.DataFrame({"text": ["Hello", "Bye"], "hd": [0, 1], "cv": [0, 0], "vo": [0, 0]})
    aggregated = pd.DataFrame(
        {
            "text_norm": ["hello", "bye"],
            "post_id": [1, 2],
            **{col: [1, 0] for col in TARGET_LABEL_COLUMNS},
            "n_annotators": [3, 3],
            "n_annotation_rows": [3, 3],
        }
    )

    enriched = enrich_ghc_split(split, aggregated)
    assert enriched.loc[0, "hate"] == 0
    assert enriched.loc[0, TARGET_LABEL_COLUMNS].isna().all()
    assert enriched.loc[1, "hate"] == 1
    assert enriched.loc[1, "rel"] == 0


def test_enrich_ghc_split_attaches_labels() -> None:
    split = pd.DataFrame({"text": ["Hello"], "hd": [1], "cv": [0], "vo": [0]})
    aggregated = pd.DataFrame(
        {
            "text_norm": ["hello"],
            "post_id": [1],
            **{col: [0] for col in TARGET_LABEL_COLUMNS},
            "n_annotators": [3],
            "n_annotation_rows": [3],
        }
    )

    enriched = enrich_ghc_split(split, aggregated)
    assert enriched.loc[0, "hate"] == 1
    assert enriched.loc[0, "post_id"] == 1
    assert enriched.loc[0, "rel"] == 0


@pytest.mark.skipif(not GHC_ANNOTATIONS_RAW.exists(), reason="GHC annotations file not available")
def test_build_labeled_splits_matches_all_rows() -> None:
    from hsd.data.ghc import AGGREGATED_LABEL_COLUMNS, build_labeled_ghc_splits

    train, test = build_labeled_ghc_splits(
        train_path=GHC_TRAIN_RAW,
        test_path=GHC_TEST_RAW,
        annotations_path=GHC_ANNOTATIONS_RAW,
    )

    assert len(train) == 22_036
    assert len(test) == 5_510
    assert set(AGGREGATED_LABEL_COLUMNS).issubset(train.columns)
    assert (train["hate"] == derive_hate_label(train["hd"], train["cv"])).all()

    hate0 = train[train["hate"] == 0]
    assert hate0[list(TARGET_LABEL_COLUMNS)].isna().all().all()
    assert not hate0[list(TARGET_LABEL_COLUMNS)].eq(1).any().any()
