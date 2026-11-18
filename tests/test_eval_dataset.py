"""Tests for external eval dataset loading and threshold resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hsd.data.canonical import load_eval_dataset, resolve_eval_path
from hsd.data.schema import DATASET_HATECHECK, DATASET_IHC, SPLIT_EVAL, empty_metadata
from hsd.evaluation.external_eval import resolve_checkpoint_threshold


def _minimal_eval_tsv(path: Path, dataset: str) -> None:
    meta = empty_metadata()
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"{dataset}-eval-1",
                "dataset": dataset,
                "split": SPLIT_EVAL,
                "text": "neutral example",
                "hate": 0,
                **meta,
            },
            {
                "sample_id": f"{dataset}-eval-2",
                "dataset": dataset,
                "split": SPLIT_EVAL,
                "text": "hateful example",
                "hate": 1,
                **meta,
            },
        ]
    )
    frame.to_csv(path, sep="\t", index=False)


def test_resolve_eval_path_default_hatecheck() -> None:
    path = resolve_eval_path(DATASET_HATECHECK)
    assert path.name == "hatecheck.tsv"
    assert "eval" in str(path)


def test_resolve_eval_path_default_ihc() -> None:
    path = resolve_eval_path(DATASET_IHC)
    assert path.name == "ihc.tsv"
    assert "eval" in str(path)


def test_resolve_eval_path_custom_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.tsv"
    assert resolve_eval_path(DATASET_HATECHECK, path=custom) == custom


def test_resolve_eval_path_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown eval dataset"):
        resolve_eval_path("not-a-dataset")


def test_load_eval_dataset_from_path(tmp_path: Path) -> None:
    tsv = tmp_path / "hatecheck.tsv"
    _minimal_eval_tsv(tsv, DATASET_HATECHECK)
    frame = load_eval_dataset(DATASET_HATECHECK, path=tsv)
    assert len(frame) == 2
    assert int(frame["hate"].sum()) == 1


def test_resolve_checkpoint_threshold_from_metrics(tmp_path: Path) -> None:
    results_dir = tmp_path / "exp"
    results_dir.mkdir()
    (results_dir / "metrics.json").write_text(
        json.dumps({"threshold": 0.42, "f1_macro": 0.75}),
        encoding="utf-8",
    )
    assert resolve_checkpoint_threshold(results_dir) == pytest.approx(0.42)


def test_resolve_checkpoint_threshold_default_when_missing(tmp_path: Path) -> None:
    results_dir = tmp_path / "exp"
    results_dir.mkdir()
    (results_dir / "metrics.json").write_text(json.dumps({"f1_macro": 0.75}), encoding="utf-8")
    assert resolve_checkpoint_threshold(results_dir) == pytest.approx(0.5)


def test_resolve_checkpoint_threshold_logit(tmp_path: Path) -> None:
    results_dir = tmp_path / "exp"
    results_dir.mkdir()
    (results_dir / "metrics_reeval_logit.json").write_text(
        json.dumps({"threshold": 0.76}),
        encoding="utf-8",
    )
    assert resolve_checkpoint_threshold(results_dir, logit=True) == pytest.approx(0.76)


def test_resolve_checkpoint_threshold_logit_missing(tmp_path: Path) -> None:
    results_dir = tmp_path / "exp"
    results_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="metrics_reeval_logit"):
        resolve_checkpoint_threshold(results_dir, logit=True)
