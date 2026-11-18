"""Tests for canonical dataset schema."""

from __future__ import annotations

import pandas as pd
import pytest

from hsd.data.schema import (
    CORE_COLUMNS,
    DATASET_GHC,
    SPLIT_TEST,
    SPLIT_TRAIN,
    validate_canonical_frame,
)


def test_validate_canonical_frame_accepts_minimal_row() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "ghc-1",
                "dataset": DATASET_GHC,
                "split": SPLIT_TRAIN,
                "text": "example",
                "hate": 0,
            }
        ]
    )
    validate_canonical_frame(frame)


def test_validate_canonical_frame_rejects_non_binary_hate() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "ghc-1",
                "dataset": DATASET_GHC,
                "split": SPLIT_TEST,
                "text": "example",
                "hate": 2,
            }
        ]
    )
    with pytest.raises(ValueError, match="binary"):
        validate_canonical_frame(frame)


def test_core_columns_order_is_stable() -> None:
    assert CORE_COLUMNS[0] == "sample_id"
    assert "hate" in CORE_COLUMNS
    assert "text" in CORE_COLUMNS


def test_validate_canonical_frame_rejects_non_hate_with_target_flags() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "ghc-1",
                "dataset": DATASET_GHC,
                "split": SPLIT_TEST,
                "text": "example",
                "hate": 0,
                "tgt_rae": 1,
                "tgt_rel": 0,
                "tgt_nat": 0,
                "tgt_gen": 0,
                "tgt_sxo": 0,
                "tgt_opc": 0,
                "tgt_other": 0,
                "tgt_unknown": 0,
            }
        ]
    )
    with pytest.raises(ValueError, match="Non-hate rows"):
        validate_canonical_frame(frame)
