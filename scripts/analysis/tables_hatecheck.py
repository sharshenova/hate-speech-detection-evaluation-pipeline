"""Layer 4: HateCheck overall and by function."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    OVERALL_COLUMNS,
    SIX_CHECKPOINTS,
    SLICE_COLUMNS,
    TABLES_DIR,
    join_eval_preds,
    load_json,
    load_pred_tsv,
    results_dir,
    row_from_preds,
    write_tsv,
)
from hsd.data.canonical import load_eval_dataset


def _hatecheck() -> pd.DataFrame:
    return load_eval_dataset("hatecheck")


def _join(spec: dict[str, str], eval_df: pd.DataFrame) -> pd.DataFrame:
    preds = load_pred_tsv(
        results_dir(spec["results_subdir"]) / "predictions_eval_hatecheck.tsv"
    )
    return join_eval_preds(eval_df, preds)


def _threshold(spec: dict[str, str]) -> float | None:
    payload = load_json(results_dir(spec["results_subdir"]) / "metrics.json")
    if spec["decode"] == "logit":
        logit = results_dir(spec["results_subdir"]) / "metrics_reeval_logit.json"
        if logit.is_file():
            return float(load_json(logit)["threshold"])
    value = payload.get("threshold")
    return None if value is None else float(value)


def _slice_row(spec: dict[str, str], frame: pd.DataFrame, stratum: str) -> dict:
    row = row_from_preds(
        frame["hate"],
        frame["y_pred"],
        layer="4-hatecheck",
        train_corpus=spec["train_corpus"],
        eval_set="hatecheck",
        model=spec["model"],
        decode=spec["decode"],
        extra={"stratum": stratum},
    )
    return {col: row.get(col, "") for col in SLICE_COLUMNS}


def main() -> None:
    eval_df = _hatecheck()
    overall_rows: list[dict] = []
    by_function: list[dict] = []

    for spec in SIX_CHECKPOINTS:
        frame = _join(spec, eval_df)
        metrics_hc = load_json(
            results_dir(spec["results_subdir"]) / "metrics_eval_hatecheck.json"
        )
        in_domain = load_json(results_dir(spec["results_subdir"]) / "metrics.json")
        in_f1 = float(in_domain["metrics"]["test"]["f1_macro"])
        if spec["decode"] == "logit":
            logit = results_dir(spec["results_subdir"]) / "metrics_reeval_logit.json"
            if logit.is_file() and "metrics" in load_json(logit):
                in_f1 = float(load_json(logit)["metrics"]["test"]["f1_macro"])
        row = row_from_preds(
            frame["hate"],
            frame["y_pred"],
            layer="4-hatecheck",
            train_corpus=spec["train_corpus"],
            eval_set="hatecheck",
            model=spec["model"],
            decode=spec["decode"],
            threshold=_threshold(spec),
        )
        row["f1_indomain"] = round(in_f1, 6)
        # Negative values mean macro-F1 fell when the same checkpoint and
        # decoding policy moved from its own test set to HateCheck.
        row["delta_vs_indomain"] = round(float(row["f1_macro"]) - in_f1, 6)
        row["eval_rows_metrics"] = metrics_hc.get("eval_rows", "")
        overall_rows.append(row)

        if "hate_function" in frame.columns:
            for name, part in frame.groupby("hate_function", sort=True):
                by_function.append(_slice_row(spec, part, str(name)))

    overall_cols = list(OVERALL_COLUMNS) + ["f1_indomain", "delta_vs_indomain"]
    write_tsv(pd.DataFrame(overall_rows)[overall_cols], TABLES_DIR / "hatecheck_overall.tsv")
    write_tsv(
        pd.DataFrame(by_function)[list(SLICE_COLUMNS)],
        TABLES_DIR / "hatecheck_by_function.tsv",
    )


if __name__ == "__main__":
    main()
