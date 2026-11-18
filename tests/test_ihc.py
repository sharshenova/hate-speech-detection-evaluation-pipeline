"""Tests for Implicit Hate Corpus eval export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hsd.data.ihc import (
    IHC_CLASS_EXPLICIT,
    IHC_CLASS_IMPLICIT,
    IHC_CLASS_NOT_HATE,
    IHC_EXTRA_COLUMNS,
    IHC_SOURCE_GAB,
    IHC_SOURCE_SAP,
    IHC_SOURCE_TWITTER,
    build_ihc_canonical,
    filter_ihc_twitter_stage1,
    ihc_class_to_hate,
    ihc_id_source,
    ihc_slice_masks,
    prepare_ihc_eval,
)
from hsd.data.paths import IHC_STG1, IHC_STG1_POSTS, REPO_ROOT
from hsd.data.schema import CANONICAL_COLUMNS, DATASET_IHC, SPLIT_EVAL, TARGET_FLAG_COLUMNS


def test_ihc_class_to_hate() -> None:
    assert ihc_class_to_hate("explicit_hate") == 1
    assert ihc_class_to_hate("implicit_hate") == 1
    assert ihc_class_to_hate("not_hate") == 0
    with pytest.raises(ValueError, match="Unexpected"):
        ihc_class_to_hate("offensive")


def test_ihc_id_source() -> None:
    assert ihc_id_source("1234567890") == IHC_SOURCE_TWITTER
    assert ihc_id_source("GAB_27086") == IHC_SOURCE_GAB
    assert ihc_id_source("SAP_abc") == IHC_SOURCE_SAP
    with pytest.raises(ValueError, match="Unexpected"):
        ihc_id_source("OTHER_1")


def test_filter_and_build_keeps_ternary_and_maps_binary_hate() -> None:
    frame = pd.DataFrame(
        [
            {
                "ID": "11",
                "class": IHC_CLASS_EXPLICIT,
                "post": "explicit example",
            },
            {
                "ID": "22",
                "class": IHC_CLASS_IMPLICIT,
                "post": "implicit example",
            },
            {
                "ID": "33",
                "class": IHC_CLASS_NOT_HATE,
                "post": "not hate example",
            },
            {
                "ID": "GAB_1",
                "class": IHC_CLASS_IMPLICIT,
                "post": "gab leak",
            },
            {
                "ID": "SAP_1",
                "class": IHC_CLASS_IMPLICIT,
                "post": "sbic leak",
            },
            {
                "ID": "44",
                "class": IHC_CLASS_IMPLICIT,
                "post": "   ",
            },
        ]
    )
    sliced = filter_ihc_twitter_stage1(frame)
    assert set(sliced["ID"]) == {"11", "22", "33"}
    canonical = build_ihc_canonical(sliced)
    assert list(canonical["dataset"].unique()) == [DATASET_IHC]
    assert list(canonical["split"].unique()) == [SPLIT_EVAL]
    assert int(canonical["hate"].sum()) == 2
    assert int((canonical["ihc_class"] == IHC_CLASS_EXPLICIT).sum()) == 1
    assert int((canonical["ihc_class"] == IHC_CLASS_IMPLICIT).sum()) == 1
    assert int((canonical["ihc_class"] == IHC_CLASS_NOT_HATE).sum()) == 1

    explicit = canonical.loc[canonical["sample_id"] == "ihc-eval-11"].iloc[0]
    assert int(explicit["hate"]) == 1
    assert int(explicit["framing_explicit"]) == 1
    assert int(explicit["framing_implicit"]) == 0
    assert int(explicit["tgt_unknown"]) == 1
    assert pd.isna(explicit["hate_function"])

    implicit = canonical.loc[canonical["sample_id"] == "ihc-eval-22"].iloc[0]
    assert int(implicit["hate"]) == 1
    assert int(implicit["framing_explicit"]) == 0
    assert int(implicit["framing_implicit"]) == 1
    assert pd.isna(implicit["hate_function"])

    non_hate = canonical.loc[canonical["sample_id"] == "ihc-eval-33"].iloc[0]
    assert int(non_hate["hate"]) == 0
    assert pd.isna(non_hate["framing_explicit"])
    assert pd.isna(non_hate["framing_implicit"])
    assert all(int(non_hate[col]) == 0 for col in TARGET_FLAG_COLUMNS)

    masks = ihc_slice_masks(canonical)
    assert int(masks["explicit_hate"].sum()) == 1
    assert int(masks["implicit_hate"].sum()) == 1
    assert int(masks["not_hate"].sum()) == 1
    assert int(canonical.loc[masks["explicit_hate"], "hate"].sum()) == 1
    assert int(canonical.loc[masks["implicit_hate"], "hate"].sum()) == 1
    assert int(canonical.loc[masks["not_hate"], "hate"].sum()) == 0


@pytest.mark.skipif(not IHC_STG1.exists(), reason="IHC raw files not available")
def test_prepare_ihc_eval_twitter_stage1(tmp_path: Path) -> None:
    out = tmp_path / "ihc.tsv"
    sliced, canonical = prepare_ihc_eval(output_path=out)
    assert len(sliced) == 20_809
    assert len(canonical) == 20_809
    assert int((canonical["ihc_class"] == IHC_CLASS_EXPLICIT).sum()) == 1_038
    assert int((canonical["ihc_class"] == IHC_CLASS_IMPLICIT).sum()) == 6_480
    assert int((canonical["ihc_class"] == IHC_CLASS_NOT_HATE).sum()) == 13_291
    assert int(canonical["hate"].sum()) == 7_518
    assert canonical["hate_function"].isna().all()
    assert (canonical["hate"] == (canonical["ihc_class"] != IHC_CLASS_NOT_HATE).astype(int)).all()
    assert out.is_file()
    written = pd.read_csv(out, sep="\t")
    assert all(col in written.columns for col in CANONICAL_COLUMNS)
    assert all(col in written.columns for col in IHC_EXTRA_COLUMNS)
    assert written["source_row_id"].astype(str).str.fullmatch(r"\d+").all()


def test_ihc_raw_path_is_under_repo() -> None:
    assert IHC_STG1 == REPO_ROOT / "data/raw/implicit-hate-corpus/implicit_hate_v1_stg1.tsv"
    assert IHC_STG1_POSTS == (
        REPO_ROOT / "data/raw/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv"
    )
