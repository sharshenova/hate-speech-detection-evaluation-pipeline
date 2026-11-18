"""Tests for DGHD canonical harmonisation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hsd.data.harmonize import build_dghd_canonical_splits, dghd_row_to_canonical
from hsd.data.schema import SPLIT_DEV, SPLIT_TEST, SPLIT_TRAIN, TGT_NAT, TGT_UNKNOWN
from hsd.data.target_mappings import target_flags_from_dghd


def test_target_flags_from_dghd_polish_maps_to_nat() -> None:
    flags = target_flags_from_dghd("pol", hate=1)
    assert flags[TGT_NAT] == 1


def test_dghd_row_to_canonical_hate_row() -> None:
    row = {
        "acl.id": "acl1",
        "text": "example",
        "label": "hate",
        "type": "derogation",
        "target": "trans",
        "level": "original",
        "split": "train",
    }
    record = dghd_row_to_canonical(row, split=SPLIT_TRAIN, sample_id="dghd-train-0")
    assert record["dataset"] == "dghd"
    assert record["hate"] == 1
    assert record["tgt_gen"] == 1
    assert record["target_raw"] == "trans"
    assert record["hate_function"] == "derogation"
    assert record["source_row_id"] == "acl1"


def test_dghd_row_to_canonical_nothate() -> None:
    row = {
        "acl.id": "acl2",
        "text": "fine",
        "label": "nothate",
        "type": "none",
        "target": "none",
        "level": "original",
        "split": "dev",
    }
    record = dghd_row_to_canonical(row, split=SPLIT_DEV, sample_id="dghd-dev-0")
    assert record["hate"] == 0
    assert record["tgt_unknown"] == 0
    assert pd.isna(record["target_raw"])
    assert pd.isna(record["hate_function"])


def test_dghd_row_to_canonical_notgiven_target() -> None:
    row = {
        "acl.id": "acl3",
        "text": "example",
        "label": "hate",
        "type": "notgiven",
        "target": "notgiven",
        "level": "original",
        "split": "test",
    }
    record = dghd_row_to_canonical(row, split=SPLIT_TEST, sample_id="dghd-test-0")
    assert record["hate"] == 1
    assert record["tgt_unknown"] == 1
    assert pd.isna(record["hate_function"])


@pytest.mark.skipif(not Path("data/processed/dghd/Dynamically Generated Hate Dataset v0.2.3_dedup.csv").exists(), reason="DGHD dedup file not available")
def test_build_dghd_canonical_splits_from_dedup() -> None:
    frame = pd.read_csv("data/processed/dghd/Dynamically Generated Hate Dataset v0.2.3_dedup.csv")
    train_can, dev_can, test_can = build_dghd_canonical_splits(frame)

    assert set(train_can["split"]) == {SPLIT_TRAIN}
    assert set(dev_can["split"]) == {SPLIT_DEV}
    assert set(test_can["split"]) == {SPLIT_TEST}
    assert len(train_can) == (frame["split"] == "train").sum()
    assert len(dev_can) == (frame["split"] == "dev").sum()
    assert len(test_can) == (frame["split"] == "test").sum()
    assert train_can["hate"].sum() == (frame[frame["split"] == "train"]["label"] == "hate").sum()
