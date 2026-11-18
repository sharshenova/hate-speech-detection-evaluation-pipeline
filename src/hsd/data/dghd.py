"""DGHD loading and deduplication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsd.data.ghc import normalize_text
from hsd.data.paths import DGHD_DEDUP, DGHD_RAW
from hsd.data.target_mappings import (
    DGHD_TARGET_NONE,
    DGHD_TARGET_NOTGIVEN,
    DGHD_TARGET_NOTARGETRECORDED,
)

DGHD_COMPARE_COLS: tuple[str, ...] = ("label", "type", "target", "level")

DGHD_REQUIRED_COLUMNS: tuple[str, ...] = (
    "acl.id",
    "text",
    "label",
    "type",
    "target",
    "level",
    "split",
)

POOR_TARGETS: frozenset[str] = frozenset(
    {
        "",
        DGHD_TARGET_NONE,
        DGHD_TARGET_NOTGIVEN,
        DGHD_TARGET_NOTARGETRECORDED,
    }
)
POOR_TYPES: frozenset[str] = frozenset({"", "none", "notgiven"})


def load_dghd(path: str | Path = DGHD_RAW) -> pd.DataFrame:
    """Load a DGHD CSV export (raw or deduplicated) preserving all original columns."""
    frame = pd.read_csv(path)
    missing = [col for col in DGHD_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Missing DGHD columns: {missing}"
        raise ValueError(msg)
    return frame


def load_dghd_raw(path: str | Path = DGHD_RAW) -> pd.DataFrame:
    """Load the DGHD CSV export."""
    return load_dghd(path)


def _duplicate_groups(frame: pd.DataFrame) -> list[str]:
    mask = frame["text_norm"].duplicated(keep=False)
    if not mask.any():
        return []
    return sorted(frame.loc[mask, "text_norm"].unique())


def _has_full_duplicate(group: pd.DataFrame) -> bool:
    return group[list(DGHD_COMPARE_COLS)].drop_duplicates().shape[0] == 1


def _has_label_conflict(group: pd.DataFrame) -> bool:
    return group["label"].drop_duplicates().shape[0] > 1


def _target_richness(value: object) -> tuple[int, int]:
    text = str(value).strip().lower()
    if text in POOR_TARGETS:
        return (0, 0)
    tokens = [part.strip() for part in text.split(",") if part.strip()]
    return (len(tokens), len(text))


def _type_richness(value: object) -> tuple[int, int]:
    text = str(value).strip().lower()
    if text in POOR_TYPES:
        return (0, 0)
    return (1, len(text))


def _row_richness_key(row: pd.Series) -> tuple[tuple[int, int], tuple[int, int]]:
    return (_target_richness(row["target"]), _type_richness(row["type"]))


def _choose_representative_index(group: pd.DataFrame) -> int:
    """Prefer the row with the richest target, then type; keep the first on a tie."""
    keys = group.apply(_row_richness_key, axis=1)
    best_pos = int(keys.idxmax()) if keys.nunique() > 1 else int(group.index[0])
    return best_pos


def _indices_to_drop(group: pd.DataFrame) -> list[int]:
    """Apply DGHD's policy to one normalized-text duplicate group.

    Keep one exact metadata duplicate, drop the whole group on a label conflict,
    and otherwise keep the richest row.
    """
    if _has_full_duplicate(group):
        return group.index[1:].tolist()
    if _has_label_conflict(group):
        return group.index.tolist()
    keep_idx = _choose_representative_index(group)
    return [idx for idx in group.index if idx != keep_idx]


def _deduplicate_within_split(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["text_norm"] = normalize_text(working["text"])
    drop_indices: list[int] = []
    for text_norm in _duplicate_groups(working):
        group = working[working["text_norm"] == text_norm]
        drop_indices.extend(_indices_to_drop(group))
    return working.drop(index=drop_indices).drop(columns=["text_norm"])


def deduplicate_within_train(frame: pd.DataFrame) -> pd.DataFrame:
    """Make within-train deduplication."""
    return _deduplicate_within_split(frame)


def deduplicate_within_dev(frame: pd.DataFrame) -> pd.DataFrame:
    """Make within-dev deduplication."""
    return _deduplicate_within_split(frame)


def deduplicate_within_test(frame: pd.DataFrame) -> pd.DataFrame:
    """Make within-test deduplication."""
    return _deduplicate_within_split(frame)


def remove_rows_overlapping_train(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose normalised text also appears in ``train``."""
    train_texts = set(normalize_text(train["text"]))
    frame_texts = normalize_text(frame["text"])
    return frame.loc[~frame_texts.isin(train_texts)].copy()


def split_by_column(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, dev, and test frames from a combined DGHD table."""
    train = frame[frame["split"] == "train"].copy()
    dev = frame[frame["split"] == "dev"].copy()
    test = frame[frame["split"] == "test"].copy()
    return train, dev, test


def deduplicate_dghd_splits(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deduplicate official splits while giving train priority over dev and test.

    Train is cleaned first. Any matching dev or test text is then removed before
    duplicates inside those splits are handled.
    """
    train_dedup = deduplicate_within_train(train)
    dev_no_overlap = remove_rows_overlapping_train(train_dedup, dev)
    dev_dedup = deduplicate_within_dev(dev_no_overlap)
    test_no_overlap = remove_rows_overlapping_train(train_dedup, test)
    test_dedup = deduplicate_within_test(test_no_overlap)
    return train_dedup, dev_dedup, test_dedup


def deduplicate_dghd_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate a full DGHD frame that already contains a ``split`` column."""
    train, dev, test = split_by_column(frame)
    train_out, dev_out, test_out = deduplicate_dghd_splits(train, dev, test)
    deduped = pd.concat([train_out, dev_out, test_out], ignore_index=True)
    return deduped[frame.columns]


def write_deduplicated_dghd(
    *,
    input_path: str | Path = DGHD_RAW,
    output_path: str | Path = DGHD_DEDUP,
) -> Path:
    """Write one deduplicated DGHD CSV with the same columns as the input file."""
    frame = load_dghd(input_path)
    deduped = deduplicate_dghd_frame(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(output_path, index=False)
    return output_path
