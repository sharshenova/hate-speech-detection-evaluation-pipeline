"""Tests for DGHD deduplication."""

from __future__ import annotations

import pandas as pd

from hsd.data.dghd import (
    deduplicate_dghd_splits,
    deduplicate_within_train,
    remove_rows_overlapping_train,
)


def _row(
    acl_id: str,
    text: str,
    *,
    label: str = "nothate",
    type_: str = "none",
    target: str = "none",
    level: str = "original",
    split: str = "train",
) -> dict[str, object]:
    return {
        "acl.id": acl_id,
        "text": text,
        "label": label,
        "type": type_,
        "target": target,
        "level": level,
        "split": split,
    }


def test_deduplicate_within_train_keeps_one_full_duplicate() -> None:
    train = pd.DataFrame(
        [
            _row("a1", "Same text"),
            _row("a2", "Same text"),
        ]
    )
    result = deduplicate_within_train(train)
    assert len(result) == 1
    assert result.iloc[0]["acl.id"] == "a1"


def test_deduplicate_within_train_drops_all_on_label_conflict() -> None:
    train = pd.DataFrame(
        [
            _row("a1", "Conflict", label="nothate"),
            _row("a2", "Conflict", label="hate", type_="derogation", target="notgiven"),
        ]
    )
    result = deduplicate_within_train(train)
    assert result.empty


def test_deduplicate_within_train_keeps_richer_target_on_metadata_conflict() -> None:
    train = pd.DataFrame(
        [
            _row(
                "a1",
                "trans women are men",
                label="hate",
                type_="notgiven",
                target="notgiven",
            ),
            _row(
                "a2",
                "trans women are men",
                label="hate",
                type_="derogation",
                target="trans",
            ),
        ]
    )
    result = deduplicate_within_train(train)
    assert len(result) == 1
    assert result.iloc[0]["acl.id"] == "a2"
    assert result.iloc[0]["target"] == "trans"


def test_remove_rows_overlapping_train() -> None:
    train = pd.DataFrame([_row("t1", "Shared", split="train")])
    dev = pd.DataFrame(
        [
            _row("d1", "Shared", split="dev"),
            _row("d2", "Dev only", split="dev"),
        ]
    )
    result = remove_rows_overlapping_train(train, dev)
    assert len(result) == 1
    assert result.iloc[0]["text"] == "Dev only"


def test_deduplicate_dghd_splits_pipeline() -> None:
    train = pd.DataFrame(
        [
            _row("t1", "Dup", label="nothate"),
            _row("t2", "Dup", label="nothate"),
            _row("t3", "Train only", label="nothate"),
        ]
    )
    dev = pd.DataFrame(
        [
            _row("d1", "Train only", label="nothate", split="dev"),
            _row("d2", "Dev dup", label="nothate", split="dev"),
            _row("d3", "Dev dup", label="nothate", split="dev"),
        ]
    )
    test = pd.DataFrame(
        [
            _row("s1", "Train only", label="nothate", split="test"),
            _row("s2", "Test only", label="nothate", split="test"),
        ]
    )
    train_out, dev_out, test_out = deduplicate_dghd_splits(train, dev, test)
    assert len(train_out) == 2
    assert len(dev_out) == 1
    assert dev_out.iloc[0]["text"] == "Dev dup"
    assert len(test_out) == 1
    assert test_out.iloc[0]["text"] == "Test only"
