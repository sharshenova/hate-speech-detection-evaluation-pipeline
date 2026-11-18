"""Implicit Hate Corpus eval export.

Primary slice: Twitter-only Stage 1. Binary ``hate`` joins explicit and
implicit into one positive class; the original label is kept in
``ihc_class`` (and ``framing_explicit`` / ``framing_implicit`` flags).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsd.data.paths import IHC_STG1, IHC_STG1_POSTS
from hsd.data.schema import (
    CANONICAL_COLUMNS,
    DATASET_IHC,
    SPLIT_EVAL,
    TGT_UNKNOWN,
    empty_metadata,
    empty_target_flags,
    validate_canonical_frame,
)

IHC_CLASS_EXPLICIT = "explicit_hate"
IHC_CLASS_IMPLICIT = "implicit_hate"
IHC_CLASS_NOT_HATE = "not_hate"

IHC_HATE_CLASSES: frozenset[str] = frozenset({IHC_CLASS_EXPLICIT, IHC_CLASS_IMPLICIT})
IHC_STAGE1_CLASSES: frozenset[str] = frozenset(
    {IHC_CLASS_EXPLICIT, IHC_CLASS_IMPLICIT, IHC_CLASS_NOT_HATE}
)

IHC_EXTRA_COLUMNS: tuple[str, ...] = ("ihc_class",)

IHC_SOURCE_TWITTER = "twitter"
IHC_SOURCE_GAB = "gab"
IHC_SOURCE_SAP = "sap"


def ihc_class_to_hate(class_label: object) -> int:
    """Map Stage-1 class to binary hate: explicit or implicit to 1."""
    label = str(class_label).strip()
    if label in IHC_HATE_CLASSES:
        return 1
    if label == IHC_CLASS_NOT_HATE:
        return 0
    msg = f"Unexpected IHC Stage-1 class: {class_label!r}"
    raise ValueError(msg)


def ihc_slice_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for later slices.
    Explicit-only and implicit-only rows are hate-only.
    """
    if "ihc_class" not in frame.columns:
        msg = "eval frame missing ihc_class; cannot slice explicit vs implicit"
        raise ValueError(msg)
    classes = frame["ihc_class"].astype(str)
    return {
        "explicit_hate": classes.eq(IHC_CLASS_EXPLICIT),
        "implicit_hate": classes.eq(IHC_CLASS_IMPLICIT),
        "not_hate": classes.eq(IHC_CLASS_NOT_HATE),
    }


def ihc_id_source(post_id: object) -> str:
    """Return expansion source from the Stage-1 ID prefix."""
    text = str(post_id).strip()
    if text.startswith("SAP_"):
        return IHC_SOURCE_SAP
    if text.startswith("GAB_"):
        return IHC_SOURCE_GAB
    if text.isdigit():
        return IHC_SOURCE_TWITTER
    msg = f"Unexpected IHC ID: {post_id!r}"
    raise ValueError(msg)


def _framing_flags(class_label: str) -> tuple[object, object]:
    """Canonical explicit/implicit flags; NA on non-hate."""
    if class_label == IHC_CLASS_EXPLICIT:
        return 1, 0
    if class_label == IHC_CLASS_IMPLICIT:
        return 0, 1
    return pd.NA, pd.NA


def load_ihc_stage1(
    *,
    ids_path: str | Path = IHC_STG1,
    posts_path: str | Path = IHC_STG1_POSTS,
) -> pd.DataFrame:
    """Join the two Stage-1 files by their required row alignment.

    Equal length and matching class labels guard against pairing an ID with the
    wrong post because the source files provide no separate join key.
    """
    ids = pd.read_csv(ids_path, sep="\t")
    posts = pd.read_csv(posts_path, sep="\t")
    if list(ids.columns) != ["ID", "class"]:
        msg = f"Unexpected Stage-1 ID columns: {list(ids.columns)}"
        raise ValueError(msg)
    if list(posts.columns) != ["post", "class"]:
        msg = f"Unexpected Stage-1 posts columns: {list(posts.columns)}"
        raise ValueError(msg)
    if len(ids) != len(posts):
        msg = f"Stage-1 length mismatch: ids {len(ids)} vs posts {len(posts)}"
        raise ValueError(msg)
    if not ids["class"].astype(str).eq(posts["class"].astype(str)).all():
        msg = "Stage-1 ID and posts files are not row-aligned on class"
        raise ValueError(msg)

    unknown = set(ids["class"].astype(str).unique()) - IHC_STAGE1_CLASSES
    if unknown:
        msg = f"Unknown Stage-1 class value(s): {sorted(unknown)}"
        raise ValueError(msg)

    return pd.DataFrame(
        {
            "ID": ids["ID"].astype(str),
            "class": ids["class"].astype(str),
            "post": posts["post"].astype(str),
        }
    )


def filter_ihc_twitter_stage1(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep non-empty Twitter rows and exclude the GAB and SAP expansions."""
    out = frame.copy()
    out["source"] = out["ID"].map(ihc_id_source)
    out = out.loc[out["source"] == IHC_SOURCE_TWITTER].copy()
    out["post"] = out["post"].astype(str).str.strip()
    out = out.loc[out["post"].ne("")].reset_index(drop=True)
    return out


def ihc_row_to_canonical(row: pd.Series | dict[str, object]) -> dict[str, object]:
    """Convert one Twitter Stage-1 row and preserve explicit/implicit framing.

    Stage 1 has no target annotation, so every hate row uses ``tgt_unknown``.
    """
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)

    class_label = str(data["class"]).strip()
    hate = ihc_class_to_hate(class_label)
    tweet_id = str(data["ID"]).strip()
    framing_explicit, framing_implicit = _framing_flags(class_label)
    flags = empty_target_flags()
    if hate == 1:
        flags[TGT_UNKNOWN] = 1

    record: dict[str, object] = {
        "sample_id": f"ihc-eval-{tweet_id}",
        "dataset": DATASET_IHC,
        "split": SPLIT_EVAL,
        "text": str(data["post"]),
        "hate": hate,
        **empty_metadata(),
        **flags,
        "framing_explicit": framing_explicit,
        "framing_implicit": framing_implicit,
        "source_row_id": tweet_id,
        "ihc_class": class_label,
    }
    return record


def build_ihc_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Map a filtered IHC dataframe to canonical schema rows."""
    records = [ihc_row_to_canonical(row) for _, row in frame.iterrows()]
    out = pd.DataFrame.from_records(records)
    validate_canonical_frame(out, require_metadata=True)
    return out


def write_ihc_eval_tsv(frame: pd.DataFrame, path: Path) -> Path:
    """Write IHC eval export: canonical columns plus ``ihc_class``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = [col for col in CANONICAL_COLUMNS if col in frame.columns]
    extra = [col for col in IHC_EXTRA_COLUMNS if col in frame.columns]
    frame.loc[:, [*canonical, *extra]].to_csv(path, sep="\t", index=False)
    return path


def prepare_ihc_eval(
    *,
    ids_path: str | Path = IHC_STG1,
    posts_path: str | Path = IHC_STG1_POSTS,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, filter to Twitter Stage 1, harmonise, and write eval TSV."""
    raw = load_ihc_stage1(ids_path=ids_path, posts_path=posts_path)
    sliced = filter_ihc_twitter_stage1(raw)
    canonical = build_ihc_canonical(sliced)
    write_ihc_eval_tsv(canonical, output_path)
    return sliced, canonical
