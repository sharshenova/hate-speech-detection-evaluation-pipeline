"""TF-IDF + logistic regression baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logistic_pipeline(model_config: dict[str, Any], *, seed: int) -> Pipeline:
    """Build a sklearn Pipeline from experiment config."""
    tfidf_cfg = dict(model_config.get("tfidf", {}))
    lr_cfg = dict(model_config.get("logistic_regression", {}))

    ngram_range = tfidf_cfg.get("ngram_range", [1, 2])
    if not isinstance(ngram_range, list) or len(ngram_range) != 2:
        msg = "tfidf.ngram_range must be a list of two integers"
        raise ValueError(msg)

    class_weight = lr_cfg.get("class_weight")
    if class_weight == "null":
        class_weight = None

    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features"),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        min_df=tfidf_cfg.get("min_df", 2),
        max_df=tfidf_cfg.get("max_df", 0.95),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
    )
    classifier = LogisticRegression(
        C=float(lr_cfg.get("C", 1.0)),
        class_weight=class_weight,
        max_iter=int(lr_cfg.get("max_iter", 1000)),
        solver=str(lr_cfg.get("solver", "lbfgs")),
        random_state=seed,
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


def positive_class_proba(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    """Return P(hate=1) for a fitted binary classifier pipeline."""
    proba = pipeline.predict_proba(texts)
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 not in classes:
        msg = "Classifier was not trained with hate=1 in the label space"
        raise ValueError(msg)
    hate_index = classes.index(1)
    return proba[:, hate_index]
