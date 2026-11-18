"""Canonical dataset schema for processed hate-speech corpora."""

from __future__ import annotations

from typing import Final

import pandas as pd

# Dataset identifiers used in processed exports and experiment configs.
DATASET_GHC = "ghc"
DATASET_DGHD = "dghd"
DATASET_KTC = "ktc"
DATASET_HATECHECK = "hatecheck"
DATASET_IHC = "ihc"

TRAIN_DATASETS: Final[tuple[str, ...]] = (DATASET_GHC, DATASET_DGHD, DATASET_KTC)
EVAL_DATASETS: Final[tuple[str, ...]] = (DATASET_HATECHECK, DATASET_IHC)

# Official split names in processed files.
SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_TEST = "test"
SPLIT_EVAL = "eval"

SPLITS_WITH_HOLDOUT: Final[tuple[str, ...]] = (SPLIT_TRAIN, SPLIT_DEV, SPLIT_TEST)
ALL_SPLITS: Final[tuple[str, ...]] = (*SPLITS_WITH_HOLDOUT, SPLIT_EVAL)

# Harmonised multilabel target flags (Tier 2). Values: 0/1.

TGT_RAE = "tgt_rae"
TGT_REL = "tgt_rel"
TGT_NAT = "tgt_nat"
TGT_GEN = "tgt_gen"
TGT_SXO = "tgt_sxo"
TGT_OPC = "tgt_opc"
TGT_OTHER = "tgt_other"
TGT_UNKNOWN = "tgt_unknown"

TARGET_FLAG_COLUMNS: Final[tuple[str, ...]] = (
    TGT_RAE,
    TGT_REL,
    TGT_NAT,
    TGT_GEN,
    TGT_SXO,
    TGT_OPC,
    TGT_OTHER,
    TGT_UNKNOWN,
)

# Columns on which stratified error analysis is grouped (excludes unknown).
TARGET_ANALYSIS_COLUMNS: Final[tuple[str, ...]] = (
    TGT_RAE,
    TGT_REL,
    TGT_NAT,
    TGT_GEN,
    TGT_SXO,
    TGT_OPC,
    TGT_OTHER,
)

# Tier 1: required for training and evaluation.
CORE_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "dataset",
    "split",
    "text",
    "hate",
)

# Optional metadata for error analysis (not used as training features).
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "source",
    *TARGET_FLAG_COLUMNS,
    "target_raw",
    "hate_function",
    "framing_explicit",
    "framing_implicit",
    "source_row_id",
    "identity_term",
    "template_type",
    "bias_eval_role",
)

CANONICAL_COLUMNS: Final[tuple[str, ...]] = (*CORE_COLUMNS, *METADATA_COLUMNS)

# KTC sub-sources: HateXplain and MMHS150K.
# Subsample columns: source, text, hate, target.
KTC_SOURCES: Final[tuple[str, ...]] = ("hatexplain", "mmhs150k")

KTC_CORE_COLUMNS: Final[tuple[str, ...]] = ("text", "hate", "target")

KTC_SUBSAMPLE_COLUMNS: Final[tuple[str, ...]] = ("source", *KTC_CORE_COLUMNS)


def empty_target_flags() -> dict[str, int]:
    """Return all harmonised target flags set to 0."""
    return dict.fromkeys(TARGET_FLAG_COLUMNS, 0)


def empty_metadata() -> dict[str, object]:
    """Return NA-filled metadata fields for a canonical row."""
    meta: dict[str, object] = {col: pd.NA for col in METADATA_COLUMNS}
    meta.update(empty_target_flags())
    return meta


def validate_canonical_frame(frame: pd.DataFrame, *, require_metadata: bool = False) -> None:
    """Check core columns, known dataset/split names, and binary hate labels.

    Metadata can be required for final exports. If any target flag is present,
    the full target block and its cross-column rules are also checked.
    """
    missing_core = [col for col in CORE_COLUMNS if col not in frame.columns]
    if missing_core:
        msg = f"Missing core columns: {missing_core}"
        raise ValueError(msg)

    if frame["hate"].isna().any():
        msg = "Column 'hate' must not contain null values."
        raise ValueError(msg)

    invalid_hate = set(frame["hate"].unique()) - {0, 1}
    if invalid_hate:
        msg = f"Column 'hate' must be binary 0/1; found {sorted(invalid_hate)}"
        raise ValueError(msg)

    invalid_datasets = set(frame["dataset"].unique()) - set(TRAIN_DATASETS + EVAL_DATASETS)
    if invalid_datasets:
        msg = f"Unknown dataset id(s): {sorted(invalid_datasets)}"
        raise ValueError(msg)

    invalid_splits = set(frame["split"].unique()) - set(ALL_SPLITS)
    if invalid_splits:
        msg = f"Unknown split name(s): {sorted(invalid_splits)}"
        raise ValueError(msg)

    if require_metadata:
        missing_meta = [col for col in METADATA_COLUMNS if col not in frame.columns]
        if missing_meta:
            msg = f"Missing metadata columns: {missing_meta}"
            raise ValueError(msg)

    _validate_target_flag_columns(frame)


def _validate_target_flag_columns(frame: pd.DataFrame) -> None:
    """Require a complete binary target block with dataset-wide invariants.

    Non-hate rows cannot have targets, and unknown cannot be combined with a
    known target on hate rows.
    """
    present = [col for col in TARGET_FLAG_COLUMNS if col in frame.columns]
    if not present:
        return

    missing = [col for col in TARGET_FLAG_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Partial target flags present; missing: {missing}"
        raise ValueError(msg)

    for col in TARGET_FLAG_COLUMNS:
        if frame[col].isna().any():
            msg = f"Column {col!r} must not contain null values."
            raise ValueError(msg)
        invalid = set(frame[col].unique()) - {0, 1}
        if invalid:
            msg = f"Column {col!r} must be binary 0/1; found {sorted(invalid)}"
            raise ValueError(msg)

    non_hate = frame["hate"] == 0
    if non_hate.any():
        subset = frame.loc[non_hate, TARGET_FLAG_COLUMNS]
        if (subset != 0).any().any():
            msg = "Non-hate rows must have all target flags set to 0."
            raise ValueError(msg)

    hate = frame["hate"] == 1
    if hate.any() and TGT_UNKNOWN in frame.columns:
        unknown = frame[TGT_UNKNOWN] == 1
        other_flags = [c for c in TARGET_FLAG_COLUMNS if c != TGT_UNKNOWN]
        both = hate & unknown & (frame[other_flags].sum(axis=1) > 0)
        if both.any():
            msg = "Rows with tgt_unknown=1 must not also set other target flags."
            raise ValueError(msg)
