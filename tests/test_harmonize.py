"""Tests for canonical harmonisation (GHC)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hsd.data.harmonize import (
    GHC_DEV_FRACTION,
    SPLIT_RANDOM_SEED,
    assign_ghc_train_dev_splits,
    build_ghc_canonical_splits,
    ghc_labeled_row_to_canonical,
)
from hsd.data.schema import (
    SPLIT_DEV,
    SPLIT_TEST,
    SPLIT_TRAIN,
    TGT_GEN,
    TGT_OPC,
    TGT_OTHER,
    TGT_RAE,
    TGT_REL,
    TGT_UNKNOWN,
)
from hsd.data.target_mappings import target_flags_from_ghc


def test_target_flags_from_ghc_maps_political_to_other() -> None:
    flags = target_flags_from_ghc({"pol": 1, "idl": 0}, hate=1)
    assert flags[TGT_OTHER] == 1
    assert flags[TGT_RAE] == 0


def test_target_flags_from_ghc_multilabel() -> None:
    flags = target_flags_from_ghc({"rae": 1, "rel": 1, "pol": 0}, hate=1)
    assert flags[TGT_RAE] == 1
    assert flags[TGT_REL] == 1


def test_target_flags_from_ghc_unknown_when_no_target() -> None:
    flags = target_flags_from_ghc({"rae": 0, "rel": pd.NA}, hate=1)
    assert flags[TGT_UNKNOWN] == 1


def test_target_flags_from_ghc_mph_to_opc() -> None:
    flags = target_flags_from_ghc({"mph": 1}, hate=1)
    assert flags[TGT_OPC] == 1


def test_ghc_labeled_row_to_canonical() -> None:
    row = {
        "post_id": 99,
        "text": "example",
        "hate": 1,
        "rae": 1,
        "rel": 0,
        "nat": 0,
        "gen": 0,
        "sxo": 0,
        "idl": 0,
        "pol": 1,
        "mph": 0,
        "ex": 1,
        "im": 0,
    }
    record = ghc_labeled_row_to_canonical(row, split=SPLIT_TRAIN, sample_id="ghc-train-0")
    assert record["dataset"] == "ghc"
    assert record["hate"] == 1
    assert record["tgt_rae"] == 1
    assert record["tgt_other"] == 1
    assert record["target_raw"] == "rae|pol"
    assert record["framing_explicit"] == 1


@pytest.mark.skipif(
    not Path("data/processed/ghc/ghc_train_labeled_dedup.tsv").exists(),
    reason="GHC dedup train file not available",
)
def test_build_ghc_canonical_splits_sizes_and_stratification() -> None:
    train = pd.read_csv("data/processed/ghc/ghc_train_labeled_dedup.tsv", sep="\t")
    test = pd.read_csv("data/processed/ghc/ghc_test_labeled_dedup.tsv", sep="\t")

    train_can, dev_can, test_can = build_ghc_canonical_splits(train, test)

    assert set(train_can["split"]) == {SPLIT_TRAIN}
    assert set(dev_can["split"]) == {SPLIT_DEV}
    assert set(test_can["split"]) == {SPLIT_TEST}

    expected_dev = round(len(train) * GHC_DEV_FRACTION)
    assert len(dev_can) == expected_dev
    assert len(train_can) == len(train) - expected_dev
    assert len(test_can) == len(test)

    train_hate_rate = train_can["hate"].mean()
    dev_hate_rate = dev_can["hate"].mean()
    assert abs(train_hate_rate - dev_hate_rate) < 0.01

    assigned = assign_ghc_train_dev_splits(train, random_state=SPLIT_RANDOM_SEED)
    assert len(assigned) == len(train)
