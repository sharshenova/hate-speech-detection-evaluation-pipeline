"""KTC: deduplicate subsample and load for canonical splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsd.data.ghc import normalize_text
from hsd.data.paths import KTC_SUBSAMPLE, KTC_SUBSAMPLE_DEDUP
from hsd.data.schema import KTC_SOURCES, KTC_SUBSAMPLE_COLUMNS

KTC_COMPARE_COLS: tuple[str, ...] = ("hate", "target")


def _duplicate_groups(frame: pd.DataFrame) -> list[str]:
    mask = frame["text_norm"].duplicated(keep=False)
    if not mask.any():
        return []
    return sorted(frame.loc[mask, "text_norm"].unique())


def _has_label_conflict(group: pd.DataFrame) -> bool:
    return group[list(KTC_COMPARE_COLS)].drop_duplicates().shape[0] > 1


def _indices_to_drop(group: pd.DataFrame) -> list[int]:
    """Drop all rows on hate/target conflict; otherwise keep one duplicate."""
    if _has_label_conflict(group):
        return group.index.tolist()
    return group.index[1:].tolist()


def deduplicate_ktc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate normalized text globally, including across KTC sources.

    Conflicting hate or target labels remove the full group; matching copies keep
    the first row.
    """
    missing = [col for col in KTC_SUBSAMPLE_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Missing KTC subsample columns: {missing}"
        raise ValueError(msg)

    working = frame.copy()
    working["text_norm"] = normalize_text(working["text"])
    drop_indices: list[int] = []
    for text_norm in _duplicate_groups(working):
        group = working[working["text_norm"] == text_norm]
        drop_indices.extend(_indices_to_drop(group))
    return working.drop(index=drop_indices).drop(columns=["text_norm"]).reset_index(drop=True)


def write_deduplicated_ktc_subsample(
    *,
    input_path: str | Path = KTC_SUBSAMPLE,
    output_path: str | Path = KTC_SUBSAMPLE_DEDUP,
) -> Path:
    """Deduplicate a KTC subsample CSV and write the result."""
    frame = pd.read_csv(input_path)
    deduped = deduplicate_ktc_frame(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(output_path, index=False)
    return output_path


def load_ktc_subsample(path: str | Path) -> pd.DataFrame:
    """Load a KTC subsample CSV and validate required columns."""
    frame = pd.read_csv(path)
    missing = [col for col in KTC_SUBSAMPLE_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Missing KTC subsample columns: {missing}"
        raise ValueError(msg)
    frame["hate"] = frame["hate"].astype(int)
    frame["target"] = frame["target"].astype(int)
    return frame


def subsample_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-source row and hate counts for logging."""
    rows = []
    for source in KTC_SOURCES:
        part = frame[frame["source"] == source]
        rows.append(
            {
                "source": source,
                "rows": len(part),
                "hate_rows": int(part["hate"].sum()),
                "hate_rate": part["hate"].mean() if len(part) else 0.0,
            }
        )
    rows.append(
        {
            "source": "total",
            "rows": len(frame),
            "hate_rows": int(frame["hate"].sum()),
            "hate_rate": frame["hate"].mean() if len(frame) else 0.0,
        }
    )
    return pd.DataFrame(rows)
