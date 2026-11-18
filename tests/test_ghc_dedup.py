"""Tests for GHC labeled-split deduplication."""

from __future__ import annotations

import pandas as pd

from hsd.data.ghc import (
    TARGET_LABEL_COLUMNS,
    deduplicate_labeled_ghc_splits,
    deduplicate_within_test,
    deduplicate_within_train,
    remove_test_rows_overlapping_train,
)


def _row(
    post_id: int,
    text: str,
    *,
    hate: int = 0,
    hd: int = 0,
    cv: int = 0,
    vo: int = 0,
) -> dict[str, object]:
    return {
        "post_id": post_id,
        "text": text,
        "hd": hd,
        "cv": cv,
        "vo": vo,
        "hate": hate,
        **{col: pd.NA for col in TARGET_LABEL_COLUMNS},
        "n_annotators": 3,
        "n_annotation_rows": 3,
    }


def test_deduplicate_within_train_keeps_one_full_duplicate() -> None:
    train = pd.DataFrame(
        [
            _row(1, "Same text", hate=0),
            _row(2, "Same text", hate=0),
        ]
    )
    result = deduplicate_within_train(train)
    assert len(result) == 1
    assert result.iloc[0]["post_id"] == 1


def test_deduplicate_within_train_removes_all_on_hate_conflict() -> None:
    train = pd.DataFrame(
        [
            _row(1, "Conflict", hate=0),
            _row(2, "Conflict", hate=1, hd=1),
        ]
    )
    result = deduplicate_within_train(train)
    assert result.empty


def test_deduplicate_within_train_removes_all_on_vo_conflict() -> None:
    train = pd.DataFrame(
        [
            _row(1, "Vo conflict", hate=0, vo=0),
            _row(2, "Vo conflict", hate=0, vo=1),
        ]
    )
    result = deduplicate_within_train(train)
    assert result.empty


def test_deduplicate_within_test_removes_all_on_conflict() -> None:
    test = pd.DataFrame(
        [
            _row(1, "Conflict", hate=0),
            _row(2, "Conflict", hate=1, hd=1),
        ]
    )
    result = deduplicate_within_test(test)
    assert result.empty


def test_remove_test_rows_overlapping_train() -> None:
    train = pd.DataFrame([_row(1, "Shared")])
    test = pd.DataFrame(
        [
            _row(10, "Shared"),
            _row(11, "Test only"),
        ]
    )
    result = remove_test_rows_overlapping_train(train, test)
    assert len(result) == 1
    assert result.iloc[0]["text"] == "Test only"


def test_deduplicate_labeled_ghc_splits_pipeline() -> None:
    train = pd.DataFrame(
        [
            _row(1, "Dup", hate=0),
            _row(2, "Dup", hate=0),
            _row(3, "Train only"),
        ]
    )
    test = pd.DataFrame(
        [
            _row(10, "Train only"),
            _row(11, "Test dup", hate=0),
            _row(12, "Test dup", hate=0),
        ]
    )
    train_out, test_out = deduplicate_labeled_ghc_splits(train, test)
    assert len(train_out) == 2
    assert len(test_out) == 1
    assert test_out.iloc[0]["text"] == "Test dup"
