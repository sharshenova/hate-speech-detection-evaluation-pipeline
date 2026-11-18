"""GHC loading, annotation aggregation, and duplicate handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsd.data.paths import (
    GHC_ANNOTATIONS_RAW,
    GHC_PROCESSED_DIR,
    GHC_TEST_LABELED,
    GHC_TEST_LABELED_DEDUP,
    GHC_TEST_RAW,
    GHC_TRAIN_LABELED,
    GHC_TRAIN_LABELED_DEDUP,
    GHC_TRAIN_RAW,
)

GHC_SPLIT_COLUMNS = ("text", "hd", "cv", "vo")

ANNOTATION_BINARY_COLUMNS = (
    "Hate",
    "HD",
    "CV",
    "VO",
    "REL",
    "RAE",
    "SXO",
    "GEN",
    "IDL",
    "NAT",
    "POL",
    "MPH",
    "EX",
    "IM",
)

AGGREGATED_LABEL_COLUMNS = (
    "hate",
    "rel",
    "rae",
    "sxo",
    "gen",
    "idl",
    "nat",
    "pol",
    "mph",
    "ex",
    "im",
)

ANNOTATION_TO_AGGREGATED = {
    "REL": "rel",
    "RAE": "rae",
    "SXO": "sxo",
    "GEN": "gen",
    "IDL": "idl",
    "NAT": "nat",
    "POL": "pol",
    "MPH": "mph",
    "EX": "ex",
    "IM": "im",
}

TARGET_LABEL_COLUMNS = AGGREGATED_LABEL_COLUMNS[1:]

LABEL_COLS = ("hate", "hd", "cv", "vo")
COMPARE_COLS = (*LABEL_COLS, *TARGET_LABEL_COLUMNS)

LABELED_SPLIT_COLUMNS = (
    "post_id",
    "text",
    "hd",
    "cv",
    "vo",
    "hate",
    *TARGET_LABEL_COLUMNS,
    "n_annotators",
    "n_annotation_rows",
)


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text for joining annotations to release splits."""
    return series.astype(str).str.strip().str.lower()


def derive_hate_label(hd: pd.Series, cv: pd.Series) -> pd.Series:
    """Derive the final hate label from the release labels, not annotator votes.

    A post is hate when it is dehumanizing (hd) or calls for violence (cv).
    """
    return ((hd.astype(int) == 1) | (cv.astype(int) == 1)).astype(int)


def majority_vote(series: pd.Series) -> int:
    """Return the most common binary label; ties go to the lowest value."""
    values = series.dropna()
    if values.empty:
        return 0
    return int(values.value_counts().idxmax())


def majority_vote_or_na(series: pd.Series) -> float:
    """Majority vote for hate-only labels; return NA when no annotator values exist."""
    values = series.dropna()
    if values.empty:
        return pd.NA
    return float(int(values.value_counts().idxmax()))


def load_ghc_split(path: str | Path = GHC_TRAIN_RAW) -> pd.DataFrame:
    """Load a GHC train or test TSV split."""
    frame = pd.read_csv(path, sep="\t")
    expected = set(GHC_SPLIT_COLUMNS)
    if set(frame.columns) != expected:
        msg = f"Unexpected GHC split columns: {list(frame.columns)}"
        raise ValueError(msg)
    return frame


def load_ghc_annotations(path: str | Path = GHC_ANNOTATIONS_RAW) -> pd.DataFrame:
    """Load annotator-level Gab Hate Corpus labels."""
    frame = pd.read_csv(path, sep="\t")
    required = ("ID", "Annotator", "Text", *ANNOTATION_BINARY_COLUMNS)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        msg = f"Missing annotation columns: {missing}"
        raise ValueError(msg)
    return frame


def aggregate_annotations_majority(
    annotations: pd.DataFrame,
    *,
    text_column: str = "Text",
) -> pd.DataFrame:
    """Aggregate annotation metadata to one row per normalized text.

    Target and framing votes come only from annotators who marked ``Hate == 1``.
    These votes do not determine the final hate label, and absent votes stay NA.
    """
    working = annotations.copy()
    working["text_norm"] = normalize_text(working[text_column])

    meta = working.groupby("text_norm", as_index=False).agg(
        post_id=("ID", "first"),
        n_annotators=("Annotator", "nunique"),
        n_annotation_rows=("Annotator", "count"),
    )

    hate_rows = working[working["Hate"] == 1]
    target_aggs = {
        out_col: (ann_col, majority_vote_or_na)
        for ann_col, out_col in ANNOTATION_TO_AGGREGATED.items()
    }
    if hate_rows.empty:
        targets = pd.DataFrame({col: pd.Series(dtype=float) for col in TARGET_LABEL_COLUMNS})
    else:
        targets = hate_rows.groupby("text_norm", as_index=False).agg(**target_aggs)

    aggregated = meta.merge(targets, on="text_norm", how="left")
    for col in TARGET_LABEL_COLUMNS:
        if col not in aggregated.columns:
            aggregated[col] = pd.NA
    return aggregated


def apply_hate_only_target_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Clear target/framing labels on non-hate rows (NA, matching raw annotations)."""
    result = frame.copy()
    non_hate = result["hate"] == 0
    result.loc[non_hate, TARGET_LABEL_COLUMNS] = pd.NA
    return result


def enrich_ghc_split(
    split: pd.DataFrame,
    aggregated_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join annotation metadata, then derive hate from the official hd/cv labels."""
    working = split.copy()
    working["text_norm"] = normalize_text(working["text"])

    enriched = working.merge(
        aggregated_labels,
        on="text_norm",
        how="left",
        validate="m:1",
    )
    if enriched["post_id"].isna().any():
        msg = "Some split rows did not match aggregated annotation labels."
        raise ValueError(msg)

    enriched["hate"] = derive_hate_label(enriched["hd"], enriched["cv"])
    enriched = apply_hate_only_target_labels(enriched)

    enriched = enriched.drop(columns=["text_norm"])
    column_order = [
        "post_id",
        "text",
        "hd",
        "cv",
        "vo",
        "hate",
        *TARGET_LABEL_COLUMNS,
        "n_annotators",
        "n_annotation_rows",
    ]
    return enriched[column_order]


def build_labeled_ghc_splits(
    train_path: str | Path = GHC_TRAIN_RAW,
    test_path: str | Path = GHC_TEST_RAW,
    annotations_path: str | Path = GHC_ANNOTATIONS_RAW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and test splits with majority-voted target labels attached."""
    annotations = load_ghc_annotations(annotations_path)
    aggregated = aggregate_annotations_majority(annotations)

    train = enrich_ghc_split(load_ghc_split(train_path), aggregated)
    test = enrich_ghc_split(load_ghc_split(test_path), aggregated)
    return train, test


def write_labeled_ghc_splits(
    output_dir: str | Path = GHC_PROCESSED_DIR,
    *,
    train_path: str | Path = GHC_TRAIN_RAW,
    test_path: str | Path = GHC_TEST_RAW,
    annotations_path: str | Path = GHC_ANNOTATIONS_RAW,
) -> tuple[Path, Path]:
    """Write labeled GHC train/test TSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = build_labeled_ghc_splits(
        train_path=train_path,
        test_path=test_path,
        annotations_path=annotations_path,
    )

    train_out = output_dir / GHC_TRAIN_LABELED.name
    test_out = output_dir / GHC_TEST_LABELED.name
    train.to_csv(train_out, sep="\t", index=False)
    test.to_csv(test_out, sep="\t", index=False)
    return train_out, test_out


def load_labeled_ghc_split(path: str | Path) -> pd.DataFrame:
    """Load a processed GHC split with annotation labels attached."""
    frame = pd.read_csv(path, sep="\t")
    missing = [col for col in LABELED_SPLIT_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Missing labeled split columns: {missing}"
        raise ValueError(msg)
    return frame


def _duplicate_groups(frame: pd.DataFrame) -> list[str]:
    mask = frame["text_norm"].duplicated(keep=False)
    if not mask.any():
        return []
    return sorted(frame.loc[mask, "text_norm"].unique())


def _has_label_conflict(group: pd.DataFrame) -> bool:
    return group[list(COMPARE_COLS)].drop_duplicates().shape[0] > 1


def _indices_to_drop(group: pd.DataFrame) -> list[int]:
    """Drop every copy on any label conflict; otherwise keep the first copy."""
    if _has_label_conflict(group):
        return group.index.tolist()
    return group.index[1:].tolist()


def _deduplicate_within_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate texts: drop all conflicting rows, keep one per full duplicate."""
    working = frame.copy()
    working["text_norm"] = normalize_text(working["text"])
    drop_indices: list[int] = []
    for text_norm in _duplicate_groups(working):
        group = working[working["text_norm"] == text_norm]
        drop_indices.extend(_indices_to_drop(group))
    return working.drop(index=drop_indices).drop(columns=["text_norm"])


def deduplicate_within_train(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply within-train duplicate removal rules."""
    return _deduplicate_within_split(frame)


def deduplicate_within_test(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply within-test duplicate removal rules."""
    return _deduplicate_within_split(frame)


def remove_test_rows_overlapping_train(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Drop test rows whose text also appears in train."""
    train_texts = set(normalize_text(train["text"]))
    test_texts = normalize_text(test["text"])
    return test.loc[~test_texts.isin(train_texts)].copy()


def deduplicate_labeled_ghc_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate train first, remove train overlap from test, then deduplicate test."""
    train_dedup = deduplicate_within_train(train)
    test_no_overlap = remove_test_rows_overlapping_train(train_dedup, test)
    test_dedup = deduplicate_within_test(test_no_overlap)
    return train_dedup, test_dedup


def write_deduplicated_ghc_splits(
    *,
    train_in: str | Path = GHC_TRAIN_LABELED,
    test_in: str | Path = GHC_TEST_LABELED,
    output_dir: str | Path = GHC_PROCESSED_DIR,
) -> tuple[Path, Path]:
    """Write deduplicated labeled GHC train/test TSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test = deduplicate_labeled_ghc_splits(
        load_labeled_ghc_split(train_in),
        load_labeled_ghc_split(test_in),
    )

    train_out = output_dir / GHC_TRAIN_LABELED_DEDUP.name
    test_out = output_dir / GHC_TEST_LABELED_DEDUP.name
    train.to_csv(train_out, sep="\t", index=False)
    test.to_csv(test_out, sep="\t", index=False)
    return train_out, test_out
