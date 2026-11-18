"""Tests for HateCheck eval export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hsd.data.hatecheck import (
    HATECHECK_EXTRA_COLUMNS,
    build_hatecheck_canonical,
    filter_hatecheck_eval_slice,
    hatecheck_label_to_hate,
    prepare_hatecheck_eval,
)
from hsd.data.paths import HATECHECK_RAW, REPO_ROOT
from hsd.data.schema import CANONICAL_COLUMNS, DATASET_HATECHECK, SPLIT_EVAL, TARGET_FLAG_COLUMNS


def test_hatecheck_label_to_hate() -> None:
    assert hatecheck_label_to_hate("hateful") == 1
    assert hatecheck_label_to_hate("non-hateful") == 0
    with pytest.raises(ValueError, match="Unexpected"):
        hatecheck_label_to_hate("toxic")


def test_filter_and_build_on_synthetic_slice() -> None:
    frame = pd.DataFrame(
        [
            {
                "case_id": 1,
                "test_case": "I hate women.",
                "label_gold": "hateful",
                "functionality": "derog_neg_emote_h",
                "target_ident": "women",
                "direction": "general",
            },
            {
                "case_id": 2,
                "test_case": "I love women.",
                "label_gold": "non-hateful",
                "functionality": "ident_pos_nh",
                "target_ident": "women",
                "direction": "-",
            },
            {
                "case_id": 3,
                "test_case": "This is fucking crazy.",
                "label_gold": "non-hateful",
                "functionality": "profanity_nh",
                "target_ident": pd.NA,
                "direction": "-",
            },
            {
                "case_id": 4,
                "test_case": "   ",
                "label_gold": "hateful",
                "functionality": "derog_neg_emote_h",
                "target_ident": "women",
                "direction": "general",
            },
        ]
    )
    sliced = filter_hatecheck_eval_slice(frame)
    assert len(sliced) == 3
    canonical = build_hatecheck_canonical(sliced)
    assert list(canonical["split"].unique()) == [SPLIT_EVAL]
    assert list(canonical["dataset"].unique()) == [DATASET_HATECHECK]
    assert int(canonical["hate"].sum()) == 1
    hate_row = canonical.loc[canonical["sample_id"] == "hatecheck-eval-1"].iloc[0]
    assert int(hate_row["tgt_gen"]) == 1
    assert hate_row["hate_function"] == "derog_neg_emote_h"
    assert hate_row["target_raw"] == "women"
    assert hate_row["direction"] == "general"
    non_hate = canonical.loc[canonical["sample_id"] == "hatecheck-eval-2"].iloc[0]
    assert int(non_hate["hate"]) == 0
    assert all(int(non_hate[col]) == 0 for col in TARGET_FLAG_COLUMNS)
    assert non_hate["target_raw"] == "women"
    no_target = canonical.loc[canonical["sample_id"] == "hatecheck-eval-3", "target_raw"]
    assert pd.isna(no_target.iloc[0])


@pytest.mark.skipif(not HATECHECK_RAW.exists(), reason="HateCheck raw file not available")
def test_prepare_hatecheck_eval_full_suite(tmp_path: Path) -> None:
    out = tmp_path / "hatecheck.tsv"
    sliced, canonical = prepare_hatecheck_eval(input_path=HATECHECK_RAW, output_path=out)
    assert len(sliced) == 3_728
    assert len(canonical) == 3_728
    assert int(canonical["hate"].sum()) == 2_563
    assert int((canonical["hate"] == 0).sum()) == 1_165
    assert canonical["hate_function"].nunique() == 29
    assert out.is_file()
    written = pd.read_csv(out, sep="\t")
    assert all(col in written.columns for col in CANONICAL_COLUMNS)
    assert all(col in written.columns for col in HATECHECK_EXTRA_COLUMNS)
    # All hate rows must have a target flag set.
    hate = written[written["hate"] == 1]
    assert (hate[list(TARGET_FLAG_COLUMNS)].sum(axis=1) >= 1).all()


def test_hatecheck_raw_path_is_under_repo() -> None:
    assert HATECHECK_RAW == REPO_ROOT / "data/raw/hatecheck/test_suite_cases.csv"
