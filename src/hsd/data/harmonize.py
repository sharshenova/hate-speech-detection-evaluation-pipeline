"""Map dataset to the canonical processed schema."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

from hsd.data.dghd import split_by_column
from hsd.data.ghc import normalize_text
from hsd.data.schema import (
    CANONICAL_COLUMNS,
    DATASET_DGHD,
    DATASET_GHC,
    DATASET_KTC,
    SPLIT_DEV,
    SPLIT_TEST,
    SPLIT_TRAIN,
    empty_metadata,
    validate_canonical_frame,
)
from hsd.data.target_mappings import (
    GHC_TARGET_SOURCE_COLUMNS,
    ghc_target_raw_from_row,
    target_flags_from_dghd,
    target_flags_from_ghc,
    target_flags_from_ktc_code,
)

# Train/eval split policy: GHC keeps official test; DGHD keeps official splits.
SPLIT_RANDOM_SEED: Final[int] = 42
GHC_DEV_FRACTION: Final[float] = 0.1
KTC_TRAIN_FRACTION: Final[float] = 0.8
KTC_DEV_FRACTION: Final[float] = 0.1
KTC_HOLDOUT_FRACTION: Final[float] = 1.0 - KTC_TRAIN_FRACTION


def _optional_int_flag(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    return int(value)


def _optional_str(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def dghd_label_to_hate(label: object) -> int:
    """Map DGHD ``label`` column to binary ``hate``."""
    return 1 if str(label).strip() == "hate" else 0


def _dghd_hate_function(row: dict[str, object], *, hate: int) -> object:
    """Keep DGHD's hate type only when it describes a hate row."""
    if hate == 0:
        return pd.NA
    type_value = _optional_str(row.get("type"))
    if type_value is pd.NA:
        return pd.NA
    if str(type_value).lower() in {"none", "notgiven"}:
        return pd.NA
    return type_value


def dghd_row_to_canonical(
    row: pd.Series | dict[str, object],
    *,
    split: str,
    sample_id: str,
) -> dict[str, object]:
    """Convert one DGHD row to a canonical record dict."""
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    hate = dghd_label_to_hate(data["label"])
    raw_target = data.get("target")
    record: dict[str, object] = {
        "sample_id": sample_id,
        "dataset": DATASET_DGHD,
        "split": split,
        "text": str(data["text"]),
        "hate": hate,
        **empty_metadata(),
        **target_flags_from_dghd(raw_target, hate=hate),
        "target_raw": _optional_str(raw_target) if hate == 1 else pd.NA,
        "hate_function": _dghd_hate_function(data, hate=hate),
        "source_row_id": str(data.get("acl.id", pd.NA)),
    }
    return record


def harmonize_dghd_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    sample_id_prefix: str,
) -> pd.DataFrame:
    """Map a DGHD dataframe slice to canonical schema rows."""
    records = [
        dghd_row_to_canonical(
            row,
            split=split,
            sample_id=f"{sample_id_prefix}-{position}",
        )
        for position, (_, row) in enumerate(frame.iterrows())
    ]
    return pd.DataFrame.from_records(records)


def build_dghd_canonical_splits(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map DGHD's official train, dev, and test assignments without resplitting."""
    train, dev, test = split_by_column(frame)

    train_rows = harmonize_dghd_frame(train, split=SPLIT_TRAIN, sample_id_prefix="dghd-train")
    dev_rows = harmonize_dghd_frame(dev, split=SPLIT_DEV, sample_id_prefix="dghd-dev")
    test_rows = harmonize_dghd_frame(test, split=SPLIT_TEST, sample_id_prefix="dghd-test")

    for part in (train_rows, dev_rows, test_rows):
        validate_canonical_frame(part, require_metadata=True)

    return train_rows, dev_rows, test_rows


def write_dghd_canonical_splits(
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Build and write canonical DGHD train, dev, and test TSV files."""
    train_rows, dev_rows, test_rows = build_dghd_canonical_splits(frame)
    output_dir = Path(output_dir)
    train_path = write_canonical_tsv(train_rows, output_dir / "dghd_train_canonical.tsv")
    dev_path = write_canonical_tsv(dev_rows, output_dir / "dghd_dev_canonical.tsv")
    test_path = write_canonical_tsv(test_rows, output_dir / "dghd_test_canonical.tsv")
    return train_path, dev_path, test_path


def ghc_labeled_row_to_canonical(
    row: pd.Series | dict[str, object],
    *,
    split: str,
    sample_id: str,
) -> dict[str, object]:
    """Convert one labeled GHC row to a canonical record dict."""
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    hate = int(data["hate"])
    record: dict[str, object] = {
        "sample_id": sample_id,
        "dataset": DATASET_GHC,
        "split": split,
        "text": str(data["text"]),
        "hate": hate,
        **empty_metadata(),
        **target_flags_from_ghc(data, hate=hate),
        "target_raw": ghc_target_raw_from_row(data) if hate == 1 else pd.NA,
        "framing_explicit": _optional_int_flag(data.get("ex")),
        "framing_implicit": _optional_int_flag(data.get("im")),
        "source_row_id": str(data.get("post_id", pd.NA)),
    }
    return record


def harmonize_ghc_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    sample_id_prefix: str,
) -> pd.DataFrame:
    """Map a labeled GHC dataframe to canonical schema rows."""
    records = [
        ghc_labeled_row_to_canonical(
            row,
            split=split,
            sample_id=f"{sample_id_prefix}-{position}",
        )
        for position, (_, row) in enumerate(frame.iterrows())
    ]
    return pd.DataFrame.from_records(records)


def assign_ghc_train_dev_splits(
    train_frame: pd.DataFrame,
    *,
    dev_fraction: float = GHC_DEV_FRACTION,
    random_state: int = SPLIT_RANDOM_SEED,
) -> pd.DataFrame:
    """Create a stratified dev set from official GHC train rows.

    The official GHC test set is handled separately and remains unchanged.
    """
    if not 0.0 < dev_fraction < 1.0:
        msg = f"dev_fraction must be between 0 and 1; got {dev_fraction}"
        raise ValueError(msg)

    indices = train_frame.index.to_numpy()
    train_idx, dev_idx = train_test_split(
        indices,
        test_size=dev_fraction,
        random_state=random_state,
        stratify=train_frame["hate"],
    )

    split_labels = pd.Series(SPLIT_TRAIN, index=train_frame.index, dtype=str)
    split_labels.loc[dev_idx] = SPLIT_DEV
    out = train_frame.copy()
    out["_canonical_split"] = split_labels
    return out


def build_ghc_canonical_splits(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    dev_fraction: float = GHC_DEV_FRACTION,
    random_state: int = SPLIT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return canonical train, dev, and test frames from labeled GHC inputs."""
    train_with_splits = assign_ghc_train_dev_splits(
        train_frame,
        dev_fraction=dev_fraction,
        random_state=random_state,
    )

    train_rows = harmonize_ghc_frame(
        train_with_splits[train_with_splits["_canonical_split"] == SPLIT_TRAIN],
        split=SPLIT_TRAIN,
        sample_id_prefix="ghc-train",
    )
    dev_rows = harmonize_ghc_frame(
        train_with_splits[train_with_splits["_canonical_split"] == SPLIT_DEV],
        split=SPLIT_DEV,
        sample_id_prefix="ghc-dev",
    )
    test_rows = harmonize_ghc_frame(
        test_frame,
        split=SPLIT_TEST,
        sample_id_prefix="ghc-test",
    )

    for part in (train_rows, dev_rows, test_rows):
        validate_canonical_frame(part, require_metadata=True)

    return train_rows, dev_rows, test_rows


def write_canonical_tsv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a canonical dataframe to TSV using the standard column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.loc[:, [col for col in CANONICAL_COLUMNS if col in frame.columns]]
    ordered.to_csv(path, sep="\t", index=False)
    return path


def write_ghc_canonical_splits(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: Path,
    *,
    dev_fraction: float = GHC_DEV_FRACTION,
    random_state: int = SPLIT_RANDOM_SEED,
) -> tuple[Path, Path, Path]:
    """Build and write canonical GHC train, dev, and test TSV files."""
    train_rows, dev_rows, test_rows = build_ghc_canonical_splits(
        train_frame,
        test_frame,
        dev_fraction=dev_fraction,
        random_state=random_state,
    )
    output_dir = Path(output_dir)
    train_path = write_canonical_tsv(train_rows, output_dir / "ghc_train_canonical.tsv")
    dev_path = write_canonical_tsv(dev_rows, output_dir / "ghc_dev_canonical.tsv")
    test_path = write_canonical_tsv(test_rows, output_dir / "ghc_test_canonical.tsv")
    return train_path, dev_path, test_path


def ktc_stratification_key(frame: pd.DataFrame) -> pd.Series:
    """Build stratification key for KTC splits: ``{hate}_{source}``."""
    return frame["hate"].astype(str) + "_" + frame["source"].astype(str)


def assign_ktc_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = KTC_TRAIN_FRACTION,
    random_state: int = SPLIT_RANDOM_SEED,
) -> pd.DataFrame:
    """Create KTC splits while preserving each hate-by-source stratum.

    The first split selects train; the remaining rows are divided equally between
    dev and test with the same random seed.
    """
    if not 0.0 < train_fraction < 1.0:
        msg = f"train_fraction must be between 0 and 1; got {train_fraction}"
        raise ValueError(msg)

    working = frame.copy()
    strat_key = ktc_stratification_key(working)
    if strat_key.value_counts().min() < 2:
        msg = "Each hate+source stratum needs at least 2 rows for stratified splitting."
        raise ValueError(msg)

    indices = working.index.to_numpy()
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=1.0 - train_fraction,
        random_state=random_state,
        stratify=strat_key,
        shuffle=True,
    )

    holdout = working.loc[holdout_idx]
    holdout_key = ktc_stratification_key(holdout)
    dev_idx, test_idx = train_test_split(
        holdout.index.to_numpy(),
        test_size=0.5,
        random_state=random_state,
        stratify=holdout_key,
        shuffle=True,
    )

    split_labels = pd.Series(SPLIT_TRAIN, index=working.index, dtype=str)
    split_labels.loc[dev_idx] = SPLIT_DEV
    split_labels.loc[test_idx] = SPLIT_TEST
    out = working.copy()
    out["_canonical_split"] = split_labels
    return out


def ktc_row_to_canonical(
    row: pd.Series | dict[str, object],
    *,
    split: str,
    sample_id: str,
) -> dict[str, object]:
    """Convert one KTC subsample row to a canonical record dict.

    Missing targets on hate rows become unknown; code -1 means no recorded target
    category and leaves every target flag at zero.
    """
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    hate = int(data["hate"])
    target_code = data.get("target")
    if hate == 1 and target_code is not None and not pd.isna(target_code):
        int_code = int(target_code)
        target_raw = pd.NA if int_code == -1 else str(int_code)
    else:
        target_raw = pd.NA

    record: dict[str, object] = {
        "sample_id": sample_id,
        "dataset": DATASET_KTC,
        "split": split,
        "text": str(data["text"]),
        "hate": hate,
        **empty_metadata(),
        **target_flags_from_ktc_code(target_code, hate=hate),
        "source": str(data["source"]),
        "target_raw": target_raw,
    }
    return record


def harmonize_ktc_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    sample_id_prefix: str,
) -> pd.DataFrame:
    """Map a KTC subsample slice to canonical schema rows."""
    records = [
        ktc_row_to_canonical(
            row,
            split=split,
            sample_id=f"{sample_id_prefix}-{position}",
        )
        for position, (_, row) in enumerate(frame.iterrows())
    ]
    return pd.DataFrame.from_records(records)


def assert_ktc_split_text_disjoint(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Raise if normalised text appears in more than one canonical split."""
    parts = {
        SPLIT_TRAIN: set(normalize_text(train["text"])),
        SPLIT_DEV: set(normalize_text(dev["text"])),
        SPLIT_TEST: set(normalize_text(test["text"])),
    }
    for left_name, left_texts in parts.items():
        for right_name, right_texts in parts.items():
            if left_name >= right_name:
                continue
            overlap = left_texts & right_texts
            if overlap:
                msg = (
                    f"Normalised text overlap between {left_name} and {right_name}: "
                    f"{len(overlap)} text(s)"
                )
                raise ValueError(msg)


def build_ktc_canonical_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = KTC_TRAIN_FRACTION,
    random_state: int = SPLIT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return canonical train, dev, and test frames from a deduplicated KTC subsample."""
    with_splits = assign_ktc_splits(
        frame,
        train_fraction=train_fraction,
        random_state=random_state,
    )

    train_rows = harmonize_ktc_frame(
        with_splits[with_splits["_canonical_split"] == SPLIT_TRAIN],
        split=SPLIT_TRAIN,
        sample_id_prefix="ktc-train",
    )
    dev_rows = harmonize_ktc_frame(
        with_splits[with_splits["_canonical_split"] == SPLIT_DEV],
        split=SPLIT_DEV,
        sample_id_prefix="ktc-dev",
    )
    test_rows = harmonize_ktc_frame(
        with_splits[with_splits["_canonical_split"] == SPLIT_TEST],
        split=SPLIT_TEST,
        sample_id_prefix="ktc-test",
    )

    for part in (train_rows, dev_rows, test_rows):
        validate_canonical_frame(part, require_metadata=True)

    assert_ktc_split_text_disjoint(train_rows, dev_rows, test_rows)
    return train_rows, dev_rows, test_rows


def ktc_split_distribution_report(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise row counts, hate rate, and source mix per canonical split."""
    rows = []
    for split_name, part in (
        (SPLIT_TRAIN, train),
        (SPLIT_DEV, dev),
        (SPLIT_TEST, test),
    ):
        rows.append(
            {
                "split": split_name,
                "rows": len(part),
                "hate_rows": int(part["hate"].sum()),
                "hate_rate": part["hate"].mean() if len(part) else 0.0,
                "hatexplain_share": ((part["source"] == "hatexplain").mean() if len(part) else 0.0),
            }
        )
    return pd.DataFrame(rows)


def write_ktc_canonical_splits(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    train_fraction: float = KTC_TRAIN_FRACTION,
    random_state: int = SPLIT_RANDOM_SEED,
) -> tuple[Path, Path, Path]:
    """Build and write canonical KTC train, dev, and test TSV files."""
    train_rows, dev_rows, test_rows = build_ktc_canonical_splits(
        frame,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    output_dir = Path(output_dir)
    train_path = write_canonical_tsv(train_rows, output_dir / "ktc_train_canonical.tsv")
    dev_path = write_canonical_tsv(dev_rows, output_dir / "ktc_dev_canonical.tsv")
    test_path = write_canonical_tsv(test_rows, output_dir / "ktc_test_canonical.tsv")
    return train_path, dev_path, test_path


def active_ghc_target_columns(row: pd.Series | dict[str, object]) -> list[str]:
    """Return GHC source target column names that are active on a row."""
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = row
    return [
        column
        for column in GHC_TARGET_SOURCE_COLUMNS
        if column in data and not pd.isna(data[column]) and int(data[column]) == 1
    ]
