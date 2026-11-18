"""Filesystem paths for local datasets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

GHC_RAW_DIR = RAW_DIR / "ghc"
GHC_PROCESSED_DIR = PROCESSED_DIR / "ghc"

GHC_TRAIN_RAW = GHC_RAW_DIR / "ghc_train.tsv"
GHC_TEST_RAW = GHC_RAW_DIR / "ghc_test.tsv"
GHC_ANNOTATIONS_RAW = GHC_RAW_DIR / "GabHateCorpus_annotations.tsv"

GHC_TRAIN_LABELED = GHC_PROCESSED_DIR / "ghc_train_labeled.tsv"
GHC_TEST_LABELED = GHC_PROCESSED_DIR / "ghc_test_labeled.tsv"

GHC_TRAIN_LABELED_DEDUP = GHC_PROCESSED_DIR / "ghc_train_labeled_dedup.tsv"
GHC_TEST_LABELED_DEDUP = GHC_PROCESSED_DIR / "ghc_test_labeled_dedup.tsv"

GHC_TRAIN_CANONICAL = GHC_PROCESSED_DIR / "ghc_train_canonical.tsv"
GHC_DEV_CANONICAL = GHC_PROCESSED_DIR / "ghc_dev_canonical.tsv"
GHC_TEST_CANONICAL = GHC_PROCESSED_DIR / "ghc_test_canonical.tsv"

DGHD_RAW_DIR = RAW_DIR / "dghd"
DGHD_RAW = DGHD_RAW_DIR / "Dynamically Generated Hate Dataset v0.2.3.csv"
DGHD_PROCESSED_DIR = PROCESSED_DIR / "dghd"
DGHD_DEDUP = DGHD_PROCESSED_DIR / "Dynamically Generated Hate Dataset v0.2.3_dedup.csv"
DGHD_TRAIN_CANONICAL = DGHD_PROCESSED_DIR / "dghd_train_canonical.tsv"
DGHD_DEV_CANONICAL = DGHD_PROCESSED_DIR / "dghd_dev_canonical.tsv"
DGHD_TEST_CANONICAL = DGHD_PROCESSED_DIR / "dghd_test_canonical.tsv"

KTC_RAW_DIR = RAW_DIR / "ktc"
KTC_SUBSAMPLE = KTC_RAW_DIR / "ktc_subsample.csv"
KTC_PROCESSED_DIR = PROCESSED_DIR / "ktc"
KTC_SUBSAMPLE_DEDUP = KTC_PROCESSED_DIR / "ktc_subsample_dedup.csv"
KTC_TRAIN_CANONICAL = KTC_PROCESSED_DIR / "ktc_train_canonical.tsv"
KTC_DEV_CANONICAL = KTC_PROCESSED_DIR / "ktc_dev_canonical.tsv"
KTC_TEST_CANONICAL = KTC_PROCESSED_DIR / "ktc_test_canonical.tsv"

HATECHECK_RAW_DIR = RAW_DIR / "hatecheck"
HATECHECK_RAW = HATECHECK_RAW_DIR / "test_suite_cases.csv"

IHC_RAW_DIR = RAW_DIR / "implicit-hate-corpus"
IHC_STG1 = IHC_RAW_DIR / "implicit_hate_v1_stg1.tsv"
IHC_STG1_POSTS = IHC_RAW_DIR / "implicit_hate_v1_stg1_posts.tsv"

EVAL_PROCESSED_DIR = PROCESSED_DIR / "eval"
HATECHECK_EVAL = EVAL_PROCESSED_DIR / "hatecheck.tsv"
IHC_EVAL = EVAL_PROCESSED_DIR / "ihc.tsv"
