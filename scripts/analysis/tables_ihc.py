"""Layer 5: ImplicitHate overall plus explicit vs implicit hate recall."""

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


def _threshold(spec: dict[str, str]) -> float | None:
    if spec["decode"] == "logit":
        logit = results_dir(spec["results_subdir"]) / "metrics_reeval_logit.json"
        if logit.is_file():
            return float(load_json(logit)["threshold"])
    payload = load_json(results_dir(spec["results_subdir"]) / "metrics.json")
    value = payload.get("threshold")
    return None if value is None else float(value)


def _in_domain_f1(spec: dict[str, str]) -> float:
    payload = load_json(results_dir(spec["results_subdir"]) / "metrics.json")
    f1 = float(payload["metrics"]["test"]["f1_macro"])
    if spec["decode"] == "logit":
        logit = results_dir(spec["results_subdir"]) / "metrics_reeval_logit.json"
        if logit.is_file() and "metrics" in load_json(logit):
            f1 = float(load_json(logit)["metrics"]["test"]["f1_macro"])
    return f1


def _slice_row(spec: dict[str, str], frame: pd.DataFrame, stratum: str) -> dict:
    row = row_from_preds(
        frame["hate"],
        frame["y_pred"],
        layer="5-ihc",
        train_corpus=spec["train_corpus"],
        eval_set="ihc",
        model=spec["model"],
        decode=spec["decode"],
        extra={"stratum": stratum},
    )
    return {col: row.get(col, "") for col in SLICE_COLUMNS}


def main() -> None:
    eval_df = load_eval_dataset("ihc")
    overall_rows: list[dict] = []
    by_class: list[dict] = []

    for spec in SIX_CHECKPOINTS:
        preds = load_pred_tsv(results_dir(spec["results_subdir"]) / "predictions_eval_ihc.tsv")
        frame = join_eval_preds(eval_df, preds)
        in_f1 = _in_domain_f1(spec)
        row = row_from_preds(
            frame["hate"],
            frame["y_pred"],
            layer="5-ihc",
            train_corpus=spec["train_corpus"],
            eval_set="ihc",
            model=spec["model"],
            decode=spec["decode"],
            threshold=_threshold(spec),
        )
        row["f1_indomain"] = round(in_f1, 6)
        # Negative values mean macro-F1 fell from the checkpoint's in-domain
        # test set to IHC under the same decoding policy.
        row["delta_vs_indomain"] = round(float(row["f1_macro"]) - in_f1, 6)
        overall_rows.append(row)

        # Stage 1 treats explicit and implicit hate as positive; not_hate is negative.
        for label in ("explicit_hate", "implicit_hate", "not_hate"):
            part = frame.loc[frame["ihc_class"].astype(str) == label]
            if part.empty:
                continue
            by_class.append(_slice_row(spec, part, label))

    overall_cols = list(OVERALL_COLUMNS) + ["f1_indomain", "delta_vs_indomain"]
    write_tsv(pd.DataFrame(overall_rows)[overall_cols], TABLES_DIR / "ihc_overall.tsv")
    write_tsv(pd.DataFrame(by_class)[list(SLICE_COLUMNS)], TABLES_DIR / "ihc_by_class.tsv")


if __name__ == "__main__":
    main()
