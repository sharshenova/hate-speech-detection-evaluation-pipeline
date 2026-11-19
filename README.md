# Hate speech detection evaluation pipeline

This is the code for my thesis project, 'Multi-Layer Evaluation of Hate Speech Detectors Across Models and Datasets'.

The project proposes a six-layer evaluation pipeline to compare hate speech detection models based on their performance, functional robustness, out-of-domain generalisation, and fairness across different protected identities.

The project employed three datasets (GHC, DGHD and KTC), which were collected using various methodologies, to develop and test several model families for binary hate speech classification: BERT-based encoders, zero/few-shot prompt engineering (Llama 3.2 3B and Gemma 4 12B) and fine-tuned small instruction-based language models (Qwen2.5 3B + QLoRA and Qwen3.5 9B/4B + LoRA).

Saved metrics and predictions are already in `results/`. Table scripts do not
retrain models.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras for training or GPU eval:

```bash
pip install -e ".[dev,ml]"      # encoders
pip install -e ".[dev,ml,lora]" # Qwen LoRA
```

## Data preparation

Raw files live under `data/raw/`. Canonical TSVs are already in
`data/processed/`. To rebuild them:

```bash
# GHC
python scripts/preprocessing/ghc/prepare_ghc_labels.py
python scripts/preprocessing/ghc/deduplicate_ghc_labels.py
python scripts/preprocessing/ghc/prepare_ghc.py

# DGHD
python scripts/preprocessing/dghd/deduplicate_dghd.py
python scripts/preprocessing/dghd/prepare_dghd.py

# KTC (merged subsample: data/raw/ktc/ktc_subsample.csv)
python scripts/preprocessing/ktc/deduplicate_ktc.py
python scripts/preprocessing/ktc/prepare_ktc.py

# External eval
python scripts/preprocessing/hatecheck/prepare_hatecheck.py
python scripts/preprocessing/ihc/prepare_ihc.py
```

## Training and eval (optional)

**Note:** Model weights were removed to save 12.7GB of space. Saved encoder and
LoRA checkpoints therefore cannot be re-evaluated with
`scripts/training/eval_lora_adapter.py` or `scripts/training/eval_external.py`.
The training script remains a recipe for retraining. Running
`scripts/training/eval_zero_shot.py` requires the Hugging Face checkpoint or
Ollama model on the machine.


Configs for reported runs are in `configs/`. 

```bash
python scripts/training/train.py --config configs/exp-001-baseline-lr.yaml --dataset ghc
python scripts/training/eval_zero_shot.py --config configs/exp-012a-gemma4-ghc-vo-note-test-nothink.yaml
python scripts/training/eval_lora_adapter.py --config configs/exp-014-qwen35-ktc.yaml --dataset ktc --logit-threshold
python scripts/training/eval_external.py --config configs/exp-005a-deberta-large-lr2e-5.yaml --dataset ghc --eval-dataset hatecheck
python scripts/training/eval_external.py --config configs/exp-015-qwen35-dghd.yaml --dataset dghd --eval-dataset ihc --logit-threshold
```


## Rebuild result tables

All numbers used in the thesis tables come from saved `metrics.json` and
`predictions*.tsv` files. Layers 2-6 also use `data/processed/` for gold labels,
slice metadata, and prediction joins. Output is `results/tables/*.tsv`.

```bash
python scripts/analysis/tables_all.py
```

Or one layer at a time:

```bash
python scripts/analysis/tables_indomain.py    # Table 4.1, Appendix C, Appendix D
python scripts/analysis/tables_stats.py       # Table 4.3 McNemar + bootstrap
python scripts/analysis/tables_slices.py      # target / function / framing / KTC source
python scripts/analysis/tables_hatecheck.py   # Table 4.7 + function slices
python scripts/analysis/tables_ihc.py         # Table 4.8 + explicit/implicit
python scripts/analysis/tables_fairness.py    # FPED/FNED and FPR/FNR by identity
```

Columns are kept consistent across overall tables: `accuracy`, `precision_macro`,
`recall_macro`, `f1_macro`, `hate_P`, `hate_R`, `hate_F1`, `FP`, `FN`, `FPR`,
`FNR`, `threshold`.
