"""HateCheck eval corpus loading and canonical export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsd.data.paths import HATECHECK_RAW
from hsd.data.schema import (
    CANONICAL_COLUMNS,
    DATASET_HATECHECK,
    SPLIT_EVAL,
    empty_metadata,
    validate_canonical_frame,
)
from hsd.data.target_mappings import target_flags_from_hatecheck_ident

HATECHECK_EXTRA_COLUMNS: tuple[str, ...] = ("direction",)

HATECHECK_REQUIRED_COLUMNS: tuple[str, ...] = (
    "case_id",
    "test_case",
    "label_gold",
    "functionality",
    "target_ident",
    "direction",
)

HATECHECK_LABEL_HATEFUL = "hateful"
HATECHECK_LABEL_NON_HATEFUL = "non-hateful"


def hatecheck_label_to_hate(label_gold: object) -> int:
    """Map HateCheck label_gold to binary hate."""
    label = str(label_gold).strip()
    if label == HATECHECK_LABEL_HATEFUL:
        return 1
    if label == HATECHECK_LABEL_NON_HATEFUL:
        return 0
    msg = f"Unexpected HateCheck label_gold: {label_gold!r}"
    raise ValueError(msg)


def load_hatecheck_raw(path: str | Path = HATECHECK_RAW) -> pd.DataFrame:
    """Load the HateCheck test-suite cases CSV."""
    path = Path(path)
    frame = pd.read_csv(path)
    missing = [col for col in HATECHECK_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Missing HateCheck columns: {missing}"
        raise ValueError(msg)
    return frame


def filter_hatecheck_eval_slice(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep all HateCheck functionalities, removing only empty test text."""
    out = frame.copy()
    out["test_case"] = out["test_case"].astype(str).str.strip()
    out = out[out["test_case"].ne("")].reset_index(drop=True)
    return out


def _target_raw_value(target_ident: object) -> object:
    """Preserve the identity tested by a case, including non-hateful cases."""
    if target_ident is None or (isinstance(target_ident, float) and pd.isna(target_ident)):
        return pd.NA
    text = str(target_ident).strip()
    return text if text else pd.NA


def hatecheck_row_to_canonical(row: pd.Series | dict[str, object]) -> dict[str, object]:
    """Convert one HateCheck case while keeping its test-suite meaning.

    ``hate_function`` stores the linguistic functionality being tested, while
    ``target_raw`` stores the case identity rather than a gold hate target.
    """
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    hate = hatecheck_label_to_hate(data["label_gold"])
    case_id = int(data["case_id"])
    target_ident = data.get("target_ident")

    record: dict[str, object] = {
        "sample_id": f"hatecheck-eval-{case_id}",
        "dataset": DATASET_HATECHECK,
        "split": SPLIT_EVAL,
        "text": str(data["test_case"]),
        "hate": hate,
        **empty_metadata(),
        **target_flags_from_hatecheck_ident(target_ident, hate=hate),
        "target_raw": _target_raw_value(target_ident),
        "hate_function": str(data["functionality"]).strip(),
        "source_row_id": str(case_id),
        "direction": str(data["direction"]).strip(),
    }
    return record


def build_hatecheck_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Map a HateCheck dataframe to canonical schema rows."""
    records = [hatecheck_row_to_canonical(row) for _, row in frame.iterrows()]
    out = pd.DataFrame.from_records(records)
    validate_canonical_frame(out, require_metadata=True)
    return out


def write_hatecheck_eval_tsv(frame: pd.DataFrame, path: Path) -> Path:
    """Write HateCheck eval export: canonical columns plus dataset-specific metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = [col for col in CANONICAL_COLUMNS if col in frame.columns]
    extra = [col for col in HATECHECK_EXTRA_COLUMNS if col in frame.columns]
    frame.loc[:, [*canonical, *extra]].to_csv(path, sep="\t", index=False)
    return path


def prepare_hatecheck_eval(
    *,
    input_path: str | Path = HATECHECK_RAW,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, harmonise, and return raw cases and canonical export."""
    raw = load_hatecheck_raw(input_path)
    sliced = filter_hatecheck_eval_slice(raw)
    canonical = build_hatecheck_canonical(sliced)
    write_hatecheck_eval_tsv(canonical, output_path)
    return sliced, canonical
