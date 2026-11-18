"""Layer 1: in-domain test metrics from saved metrics.json / predictions.

Writes:
  results/tables/indomain_overall.tsv
  results/tables/prompt_engineering.tsv
  results/tables/lora_finetune.tsv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    OVERALL_COLUMNS,
    TABLES_DIR,
    load_json,
    load_pred_tsv,
    results_dir,
    row_from_metrics_split,
    row_from_preds,
    write_tsv,
)
from hsd.evaluation.label_parse import parse_zero_shot_label

# Checkpoints for the comparison table (best setting per family x corpus).
INDOMAIN_RUNS: tuple[dict[str, str], ...] = (
    {"model": "TF-IDF+LR", "train_corpus": "GHC", "subdir": "exp-001-baseline-lr/ghc", "decode": "threshold"},
    {"model": "TF-IDF+LR", "train_corpus": "DGHD", "subdir": "exp-001-baseline-lr/dghd", "decode": "threshold"},
    {"model": "TF-IDF+LR", "train_corpus": "KTC", "subdir": "exp-001-baseline-lr/ktc", "decode": "threshold"},
    {"model": "BERT-base", "train_corpus": "GHC", "subdir": "exp-006-bert-base/ghc", "decode": "threshold"},
    {"model": "BERT-base", "train_corpus": "DGHD", "subdir": "exp-006a-bert-lr3e-5/dghd", "decode": "threshold"},
    {"model": "BERT-base", "train_corpus": "KTC", "subdir": "exp-006-bert-base/ktc", "decode": "threshold"},
    {"model": "RoBERTa-base", "train_corpus": "GHC", "subdir": "exp-002c-roberta-lr3e5/ghc", "decode": "threshold"},
    {"model": "RoBERTa-base", "train_corpus": "DGHD", "subdir": "exp-002c-roberta-lr3e5/dghd", "decode": "threshold"},
    {"model": "RoBERTa-base", "train_corpus": "KTC", "subdir": "exp-002-roberta-base/ktc", "decode": "threshold"},
    {"model": "fBERT", "train_corpus": "GHC", "subdir": "exp-003-fbert/ghc", "decode": "threshold"},
    {"model": "fBERT", "train_corpus": "DGHD", "subdir": "exp-003-fbert/dghd", "decode": "threshold"},
    {"model": "fBERT", "train_corpus": "KTC", "subdir": "exp-003-fbert/ktc", "decode": "threshold"},
    {"model": "DeBERTa-v3-base", "train_corpus": "GHC", "subdir": "exp-004a-deberta-lr2e-5/ghc", "decode": "threshold"},
    {"model": "DeBERTa-v3-base", "train_corpus": "DGHD", "subdir": "exp-004b-deberta-lr3e-5/dghd", "decode": "threshold"},
    {"model": "DeBERTa-v3-base", "train_corpus": "KTC", "subdir": "exp-004a-deberta-lr2e-5/ktc", "decode": "threshold"},
    {"model": "DeBERTa-large", "train_corpus": "GHC", "subdir": "exp-005a-deberta-large-lr2e-5/ghc", "decode": "threshold"},
    {"model": "DeBERTa-large", "train_corpus": "DGHD", "subdir": "exp-005b-deberta-large-lr3e-5/dghd", "decode": "threshold"},
    {"model": "DeBERTa-large", "train_corpus": "KTC", "subdir": "exp-005c-deberta-large-lr1e-5/ktc", "decode": "threshold"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-012a-gemma4-ghc-vo-note-test-nothink/ghc", "decode": "zero-shot"},
    {"model": "Gemma 4 12B", "train_corpus": "DGHD", "subdir": "exp-012b-gemma4-dghd-vo-note-test-nothink/dghd", "decode": "zero-shot"},
    {"model": "Gemma 4 12B", "train_corpus": "KTC", "subdir": "exp-012c-gemma4-ktc-combined-test-nothink/ktc", "decode": "zero-shot"},
    {"model": "Qwen2.5-3B", "train_corpus": "GHC", "subdir": "exp-008-qwen-lora/ghc", "decode": "greedy"},
    {"model": "Qwen2.5-3B", "train_corpus": "DGHD", "subdir": "exp-008a-qwen-lora-dghd-identity/dghd", "decode": "greedy"},
    {"model": "Qwen2.5-3B", "train_corpus": "KTC", "subdir": "exp-008-qwen-lora/ktc", "decode": "greedy"},
    {"model": "Qwen3.5", "train_corpus": "GHC", "subdir": "exp-017-qwen35-ghc-recipe008/ghc", "decode": "greedy"},
    {"model": "Qwen3.5", "train_corpus": "DGHD", "subdir": "exp-015-qwen35-dghd/dghd", "decode": "logit"},
    {"model": "Qwen3.5", "train_corpus": "KTC", "subdir": "exp-014-qwen35-ktc/ktc", "decode": "logit"},
)

PROMPT_RUNS: tuple[dict[str, str], ...] = (
    {"model": "Llama 3.2 3B", "train_corpus": "GHC", "subdir": "exp-007a-zero-shot-minimal/ghc", "prompt": "Minimal"},
    {"model": "Llama 3.2 3B", "train_corpus": "GHC", "subdir": "exp-007d-zero-shot-short-vo-note/ghc", "prompt": "Short GHC"},
    {"model": "Llama 3.2 3B", "train_corpus": "GHC", "subdir": "exp-007b-zero-shot-strict-ghc/ghc", "prompt": "Detailed"},
    {"model": "Llama 3.2 3B", "train_corpus": "GHC", "subdir": "exp-007c-few-shot-strict-ghc/ghc", "prompt": "Detailed + few-shot"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009a-gemma4-minimal/ghc", "prompt": "Minimal (think ON)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009h-gemma4-minimal-nothink/ghc", "prompt": "Minimal (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009b-gemma4-vo-note/ghc", "prompt": "Short GHC (think ON)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009f-gemma4-vo-note-nothink/ghc", "prompt": "Short GHC (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009c-gemma4-strict/ghc", "prompt": "Detailed GHC (think ON)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009d-gemma4-few-shot-vo-hard10/ghc", "prompt": "Short + few-shot 10/10 (ON)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009e-gemma4-few-shot-vo-hard6-nothink/ghc", "prompt": "Short + few-shot 6/6 (OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-009g-gemma4-few-shot-vo-hard5/ghc", "prompt": "Short + few-shot 5/5 (ON)"},
    {"model": "Gemma 4 12B", "train_corpus": "DGHD", "subdir": "exp-010a-gemma4-dghd-minimal/dghd", "prompt": "Minimal (think ON)"},
    {"model": "Gemma 4 12B", "train_corpus": "DGHD", "subdir": "exp-010d-gemma4-dghd-minimal-nothink/dghd", "prompt": "Minimal (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "DGHD", "subdir": "exp-010b-gemma4-dghd-vo-note-nothink/dghd", "prompt": "Short DGHD (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "DGHD", "subdir": "exp-010c-gemma4-dghd-few-shot-vo-hard6-nothink/dghd", "prompt": "Short + few-shot 6/6 (OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "KTC", "subdir": "exp-011a-gemma4-ktc-minimal-nothink/ktc", "prompt": "Minimal (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "KTC", "subdir": "exp-011b-gemma4-ktc-combined-nothink/ktc", "prompt": "Short KTC (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "KTC", "subdir": "exp-011c-gemma4-ktc-few-shot-combined-hard6-nothink/ktc", "prompt": "Short + few-shot 6/6 (OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "GHC", "subdir": "exp-012a-gemma4-ghc-vo-note-test-nothink/ghc", "prompt": "Short GHC test (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "DGHD", "subdir": "exp-012b-gemma4-dghd-vo-note-test-nothink/dghd", "prompt": "Short DGHD test (think OFF)"},
    {"model": "Gemma 4 12B", "train_corpus": "KTC", "subdir": "exp-012c-gemma4-ktc-combined-test-nothink/ktc", "prompt": "Short KTC test (think OFF)"},
)

LORA_RUNS: tuple[dict[str, str], ...] = (
    {"model": "Qwen2.5-3B", "train_corpus": "GHC", "subdir": "exp-008-qwen-lora/ghc", "note": "exp 1 greedy"},
    {"model": "Qwen2.5-3B", "train_corpus": "GHC", "subdir": "exp-008c-qwen-lora-ghc-vo-note/ghc", "note": "exp 2 VO-note"},
    {"model": "Qwen2.5-3B", "train_corpus": "GHC", "subdir": "exp-008d-qwen-lora-ghc-balanced/ghc", "note": "exp 3 balanced"},
    {"model": "Qwen3.5-9B", "train_corpus": "GHC", "subdir": "exp-017-qwen35-ghc-recipe008/ghc", "note": "recipe 008 greedy"},
    {"model": "Qwen2.5-3B", "train_corpus": "DGHD", "subdir": "exp-008a-qwen-lora-dghd-identity/dghd", "note": "identity prompt"},
    {"model": "Qwen3.5-9B", "train_corpus": "DGHD", "subdir": "exp-015-qwen35-dghd/dghd", "note": "logit 0.50"},
    {"model": "Qwen2.5-3B", "train_corpus": "KTC", "subdir": "exp-008-qwen-lora/ktc", "note": "greedy"},
    {"model": "Qwen3.5-4B", "train_corpus": "KTC", "subdir": "exp-014-qwen35-ktc/ktc", "note": "logit 0.76"},
)


def _logit_payload(subdir: str) -> dict | None:
    path = results_dir(subdir) / "metrics_reeval_logit.json"
    if path.is_file():
        return load_json(path)
    return None


def _output_cfg(results_dir_path: Path) -> dict:
    prompt_path = results_dir_path / "prompt.resolved.yaml"
    if prompt_path.is_file():
        payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
        output = payload.get("output")
        if isinstance(output, dict):
            return output
    return {"parse_as_hate": ["hate"], "parse_as_non_hate": ["non-hate"]}


def _count_parse_failures(results_dir_path: Path, fallback: object) -> int:
    """Audit saved responses with the current strict label parser.

    This catches long dumps or truncated thinking that older runs may have labeled.
    The stored count is used only when raw responses are unavailable.
    """
    pred_path = results_dir_path / "predictions.tsv"
    if not pred_path.is_file():
        return 0 if fallback in (None, "") else int(fallback)
    preds = load_pred_tsv(pred_path)
    if "model_response" not in preds.columns:
        return 0 if fallback in (None, "") else int(fallback)
    output_cfg = _output_cfg(results_dir_path)
    n = 0
    for raw in preds["model_response"].fillna(""):
        label, _ = parse_zero_shot_label(str(raw), output_cfg)
        if label is None:
            n += 1
    return n


def _row_for_run(spec: dict[str, str], *, layer: str, prefer_logit: bool) -> dict:
    """Build one in-domain row from the selected decoding artifact.

    For logit runs, saved test predictions take priority over logit summary metrics.
    Other runs use the main metrics file.
    """
    rdir = results_dir(spec["subdir"])
    decode = spec.get("decode", "greedy")
    payload = load_json(rdir / "metrics.json")
    pred_path = rdir / "predictions_test.tsv"

    if decode == "logit" and prefer_logit:
        logit = _logit_payload(spec["subdir"])
        if pred_path.is_file():
            preds = load_pred_tsv(pred_path)
            return row_from_preds(
                preds["y_true"],
                preds["y_pred"],
                layer=layer,
                train_corpus=spec["train_corpus"],
                eval_set="test",
                model=spec["model"],
                decode="logit",
                threshold=None if logit is None else float(logit["threshold"]),
            )
        if logit is not None and "metrics" in logit and "test" in logit["metrics"]:
            payload = logit

    split = "test"
    if "metrics" in payload and split not in payload["metrics"]:
        split = str(payload.get("eval_split", "dev"))
    return row_from_metrics_split(
        payload,
        split,
        layer=layer,
        train_corpus=spec["train_corpus"],
        eval_set=split,
        model=spec["model"],
        decode=decode,
    )


def _zero_shot_row(spec: dict[str, str]) -> dict:
    """Prefer saved predictions so metrics and the response audit use the same rows."""
    rdir = results_dir(spec["subdir"])
    payload = load_json(rdir / "metrics.json")
    split = str(payload.get("eval_split", "dev"))
    pred_path = rdir / "predictions.tsv"
    if pred_path.is_file():
        preds = load_pred_tsv(pred_path)
        row = row_from_preds(
            preds["y_true"],
            preds["y_pred"],
            layer="1-prompt",
            train_corpus=spec["train_corpus"],
            eval_set=split,
            model=spec["model"],
            decode="zero-shot",
        )
    else:
        row = row_from_metrics_split(
            payload,
            split,
            layer="1-prompt",
            train_corpus=spec["train_corpus"],
            eval_set=split,
            model=spec["model"],
            decode="zero-shot",
        )
    row["prompt"] = spec["prompt"]
    row["parse_failures"] = _count_parse_failures(rdir, payload.get("parse_failures", ""))
    return row


def _lora_row(spec: dict[str, str]) -> dict:
    rdir = results_dir(spec["subdir"])
    payload = load_json(rdir / "metrics.json")
    greedy = row_from_metrics_split(
        payload,
        "test",
        layer="1-lora",
        train_corpus=spec["train_corpus"],
        eval_set="test",
        model=spec["model"],
        decode="greedy",
    )
    greedy["note"] = spec["note"]
    greedy["train_f1"] = round(float(payload["metrics"]["train"]["f1_macro"]), 6)
    greedy["dev_f1"] = round(float(payload["metrics"]["dev"]["f1_macro"]), 6)
    greedy["test_f1_greedy"] = greedy["f1_macro"]
    logit = _logit_payload(spec["subdir"])
    if logit is not None and "metrics" in logit and "test" in logit["metrics"]:
        greedy["test_f1_logit"] = round(float(logit["metrics"]["test"]["f1_macro"]), 6)
        greedy["logit_threshold"] = round(float(logit["threshold"]), 3)
    else:
        greedy["test_f1_logit"] = ""
        greedy["logit_threshold"] = ""
    return greedy


def main() -> None:
    overall = [_row_for_run(spec, layer="1-indomain", prefer_logit=True) for spec in INDOMAIN_RUNS]
    write_tsv(
        pd.DataFrame(overall)[list(OVERALL_COLUMNS)],
        TABLES_DIR / "indomain_overall.tsv",
    )

    prompt_rows = [_zero_shot_row(spec) for spec in PROMPT_RUNS]
    prompt_cols = list(OVERALL_COLUMNS) + ["prompt", "parse_failures"]
    write_tsv(
        pd.DataFrame(prompt_rows)[prompt_cols],
        TABLES_DIR / "prompt_engineering.tsv",
    )

    lora_rows = [_lora_row(spec) for spec in LORA_RUNS]
    lora_cols = list(OVERALL_COLUMNS) + [
        "note",
        "train_f1",
        "dev_f1",
        "test_f1_greedy",
        "test_f1_logit",
        "logit_threshold",
    ]
    write_tsv(
        pd.DataFrame(lora_rows)[lora_cols],
        TABLES_DIR / "lora_finetune.tsv",
    )


if __name__ == "__main__":
    main()
