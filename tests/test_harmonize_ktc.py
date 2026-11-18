"""Tests for KTC canonical harmonisation and stratified splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hsd.data.ghc import normalize_text
from hsd.data.harmonize import (
    KTC_TRAIN_FRACTION,
    SPLIT_RANDOM_SEED,
    assign_ktc_splits,
    build_ktc_canonical_splits,
    ktc_row_to_canonical,
    ktc_stratification_key,
)
from hsd.data.schema import SPLIT_DEV, SPLIT_TEST, SPLIT_TRAIN, TGT_NAT, TGT_UNKNOWN


def _ktc_fixture(*, rows_per_stratum: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for hate in (0, 1):
        for source in ("hatexplain", "mmhs150k"):
            for index in range(rows_per_stratum):
                rows.append(
                    {
                        "source": source,
                        "text": f"{source}-h{hate}-r{index}",
                        "hate": hate,
                        "target": -1 if hate == 0 else (0 if source == "hatexplain" else 2),
                    }
                )
    return pd.DataFrame(rows)


def test_ktc_stratification_key() -> None:
    frame = _ktc_fixture(rows_per_stratum=1)
    keys = ktc_stratification_key(frame)
    assert set(keys) == {"0_hatexplain", "0_mmhs150k", "1_hatexplain", "1_mmhs150k"}


def test_ktc_row_to_canonical_hate_row() -> None:
    row = {
        "source": "mmhs150k",
        "text": "example",
        "hate": 1,
        "target": 2,
    }
    record = ktc_row_to_canonical(row, split=SPLIT_TRAIN, sample_id="ktc-train-0")
    assert record["dataset"] == "ktc"
    assert record["source"] == "mmhs150k"
    assert record["hate"] == 1
    assert record[TGT_NAT] == 1
    assert record["target_raw"] == "2"


def test_ktc_row_to_canonical_nothate() -> None:
    row = {
        "source": "hatexplain",
        "text": "fine",
        "hate": 0,
        "target": -1,
    }
    record = ktc_row_to_canonical(row, split=SPLIT_DEV, sample_id="ktc-dev-0")
    assert record["hate"] == 0
    assert record[TGT_UNKNOWN] == 0
    assert pd.isna(record["target_raw"])


def test_assign_ktc_splits_sizes_and_strata() -> None:
    frame = _ktc_fixture()
    assigned = assign_ktc_splits(frame, random_state=SPLIT_RANDOM_SEED)
    counts = assigned["_canonical_split"].value_counts()

    expected_train = round(len(frame) * KTC_TRAIN_FRACTION)
    expected_holdout = len(frame) - expected_train
    assert counts[SPLIT_TRAIN] == expected_train
    assert counts[SPLIT_DEV] + counts[SPLIT_TEST] == expected_holdout
    assert counts[SPLIT_DEV] == counts[SPLIT_TEST]


def test_build_ktc_canonical_splits_disjoint_texts() -> None:
    frame = _ktc_fixture()
    train_can, dev_can, test_can = build_ktc_canonical_splits(frame)

    assert set(train_can["split"]) == {SPLIT_TRAIN}
    assert set(dev_can["split"]) == {SPLIT_DEV}
    assert set(test_can["split"]) == {SPLIT_TEST}
    assert len(train_can) + len(dev_can) + len(test_can) == len(frame)

    train_norm = set(normalize_text(train_can["text"]))
    dev_norm = set(normalize_text(dev_can["text"]))
    test_norm = set(normalize_text(test_can["text"]))
    assert not (train_norm & dev_norm)
    assert not (train_norm & test_norm)
    assert not (dev_norm & test_norm)


@pytest.mark.skipif(
    not Path("data/processed/ktc/ktc_subsample_dedup.csv").exists(),
    reason="KTC dedup file not available",
)
def test_build_ktc_canonical_splits_from_dedup() -> None:
    frame = pd.read_csv("data/processed/ktc/ktc_subsample_dedup.csv")
    train_can, dev_can, test_can = build_ktc_canonical_splits(frame)

    total = len(frame)
    assert len(train_can) + len(dev_can) + len(test_can) == total
    assert abs(len(train_can) / total - KTC_TRAIN_FRACTION) < 0.01
    assert abs(len(dev_can) / total - 0.1) < 0.01
    assert abs(len(test_can) / total - 0.1) < 0.01
