"""Load canonical train/dev/test exports for model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsd.data.paths import (
    DGHD_DEV_CANONICAL,
    DGHD_TEST_CANONICAL,
    DGHD_TRAIN_CANONICAL,
    GHC_DEV_CANONICAL,
    GHC_TEST_CANONICAL,
    GHC_TRAIN_CANONICAL,
    HATECHECK_EVAL,
    IHC_EVAL,
    KTC_DEV_CANONICAL,
    KTC_TEST_CANONICAL,
    KTC_TRAIN_CANONICAL,
)
from hsd.data.schema import (
    DATASET_DGHD,
    DATASET_GHC,
    DATASET_HATECHECK,
    DATASET_IHC,
    DATASET_KTC,
    EVAL_DATASETS,
    validate_canonical_frame,
)

CANONICAL_SPLIT_PATHS: dict[str, tuple[Path, Path, Path]] = {
    DATASET_GHC: (GHC_TRAIN_CANONICAL, GHC_DEV_CANONICAL, GHC_TEST_CANONICAL),
    DATASET_DGHD: (DGHD_TRAIN_CANONICAL, DGHD_DEV_CANONICAL, DGHD_TEST_CANONICAL),
    DATASET_KTC: (KTC_TRAIN_CANONICAL, KTC_DEV_CANONICAL, KTC_TEST_CANONICAL),
}

EVAL_DATASET_PATHS: dict[str, Path] = {
    DATASET_HATECHECK: HATECHECK_EVAL,
    DATASET_IHC: IHC_EVAL,
}


def resolve_canonical_paths(
    dataset: str,
    *,
    train_path: str | Path | None = None,
    dev_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return train, dev, and test paths for a dataset."""
    if train_path is not None and dev_path is not None and test_path is not None:
        return Path(train_path), Path(dev_path), Path(test_path)

    try:
        default_train, default_dev, default_test = CANONICAL_SPLIT_PATHS[dataset]
    except KeyError as exc:
        msg = f"Unknown dataset {dataset!r}; expected one of {sorted(CANONICAL_SPLIT_PATHS)}"
        raise ValueError(msg) from exc

    return (
        Path(train_path) if train_path is not None else default_train,
        Path(dev_path) if dev_path is not None else default_dev,
        Path(test_path) if test_path is not None else default_test,
    )


def load_canonical_split(path: str | Path) -> pd.DataFrame:
    """Load one canonical TSV split."""
    frame = pd.read_csv(path, sep="\t")
    validate_canonical_frame(frame, require_metadata=True)
    return frame


def resolve_eval_path(dataset: str, *, path: str | Path | None = None) -> Path:
    """Return the processed TSV path for an external eval corpus."""
    if path is not None:
        return Path(path)

    try:
        default = EVAL_DATASET_PATHS[dataset]
    except KeyError as exc:
        known = sorted(EVAL_DATASET_PATHS)
        msg = (
            f"Unknown eval dataset {dataset!r}; expected one of {known}. "
            f"Supported eval ids: {sorted(EVAL_DATASETS)}"
        )
        raise ValueError(msg) from exc

    return default


def load_eval_split(path: str | Path) -> pd.DataFrame:
    """Load one external eval TSV."""
    frame = pd.read_csv(path, sep="\t")
    validate_canonical_frame(frame, require_metadata=True)
    return frame


def load_eval_dataset(
    dataset: str,
    *,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a held-out eval corpus (HateCheck or IHC)."""
    eval_path = resolve_eval_path(dataset, path=path)
    if not eval_path.is_file():
        msg = f"Eval export not found: {eval_path}"
        raise FileNotFoundError(msg)
    return load_eval_split(eval_path)


def load_canonical_dataset(
    dataset: str,
    *,
    train_path: str | Path | None = None,
    dev_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical train, dev, and test frames for a dataset."""
    train_p, dev_p, test_p = resolve_canonical_paths(
        dataset,
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
    )
    return (
        load_canonical_split(train_p),
        load_canonical_split(dev_p),
        load_canonical_split(test_p),
    )


def xy_from_frame(
    frame: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "hate",
) -> tuple[list[str], pd.Series]:
    """Extract text inputs and binary labels from a canonical frame."""
    for column in (text_column, label_column):
        if column not in frame.columns:
            msg = f"Missing column {column!r} in canonical frame"
            raise ValueError(msg)
    texts = frame[text_column].astype(str).tolist()
    labels = frame[label_column].astype(int)
    return texts, labels
