"""Layer 6: HateCheck identity-slice FPR/FNR and FPED/FNED.

FPED uses gold non-hate rows and FNED uses gold hate rows with a named target_raw.
FPR_delta / FNR_delta are signed (identity rate - overall pool rate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    SIX_CHECKPOINTS,
    TABLES_DIR,
    join_eval_preds,
    load_pred_tsv,
    results_dir,
    write_tsv,
)
from hsd.data.canonical import load_eval_dataset
from hsd.evaluation.bias import compute_hatecheck_bias_metrics


def main() -> None:
    eval_df = load_eval_dataset("hatecheck")
    overall_rows: list[dict] = []
    per_identity: list[dict] = []

    for spec in SIX_CHECKPOINTS:
        preds = load_pred_tsv(
            results_dir(spec["results_subdir"]) / "predictions_eval_hatecheck.tsv"
        )
        frame = join_eval_preds(eval_df, preds)
        bias = compute_hatecheck_bias_metrics(frame, y_true_col="hate", y_pred_col="y_pred")
        overall_rows.append(
            {
                "layer": "6-fairness",
                "train_corpus": spec["train_corpus"],
                "eval_set": "hatecheck",
                "model": spec["model"],
                "decode": spec["decode"],
                "n_identity_terms": bias["n_identity_terms_fped"],
                "n_nonhate": bias["n_fped_rows"],
                "n_hate": bias["n_fned_rows"],
                "n_missing_excluded": bias["n_missing_target_excluded"],
                "FPR": round(float(bias["fpr_overall"]), 6),
                "FNR": round(float(bias["fnr_overall"]), 6),
                "FP": bias["fp"],
                "FN": bias["fn"],
                "FPED": round(float(bias["fped"]), 6),
                "FNED": round(float(bias["fned"]), 6),
            }
        )
        fpr_map = {row["identity_term"]: row for row in bias["fped_per_term"]}
        fnr_map = {row["identity_term"]: row for row in bias["fned_per_term"]}
        # Some identities may occur in only one gold-label pool, so keep the union
        # and leave the unavailable rate blank.
        identities = sorted(set(fpr_map) | set(fnr_map))
        for ident in identities:
            fpr_row = fpr_map.get(ident, {})
            fnr_row = fnr_map.get(ident, {})
            per_identity.append(
                {
                    "layer": "6-fairness",
                    "train_corpus": spec["train_corpus"],
                    "eval_set": "hatecheck",
                    "model": spec["model"],
                    "identity": ident,
                    "n_nonhate": fpr_row.get("n", ""),
                    "FPR": round(float(fpr_row["fpr"]), 6) if "fpr" in fpr_row else "",
                    "FPR_gap": (
                        round(float(fpr_row["abs_gap_vs_overall"]), 6)
                        if "abs_gap_vs_overall" in fpr_row
                        else ""
                    ),
                    "FPR_delta": (
                        round(float(fpr_row["signed_gap_vs_overall"]), 6)
                        if "signed_gap_vs_overall" in fpr_row
                        else ""
                    ),
                    "n_hate": fnr_row.get("n", ""),
                    "FNR": round(float(fnr_row["fnr"]), 6) if "fnr" in fnr_row else "",
                    "FNR_gap": (
                        round(float(fnr_row["abs_gap_vs_overall"]), 6)
                        if "abs_gap_vs_overall" in fnr_row
                        else ""
                    ),
                    "FNR_delta": (
                        round(float(fnr_row["signed_gap_vs_overall"]), 6)
                        if "signed_gap_vs_overall" in fnr_row
                        else ""
                    ),
                }
            )

    write_tsv(pd.DataFrame(overall_rows), TABLES_DIR / "hatecheck_fped_fned.tsv")
    write_tsv(pd.DataFrame(per_identity), TABLES_DIR / "hatecheck_fpr_fnr_by_identity.tsv")


if __name__ == "__main__":
    main()
