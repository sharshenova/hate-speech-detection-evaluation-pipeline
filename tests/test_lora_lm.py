"""Unit tests for LoRA training helpers."""

import numpy as np

from hsd.training.lora_lm import (
    _balanced_sample_weights,
    _chat_template_kwargs,
    _use_qlora,
    hate_prob_from_logits_row,
)


def test_use_qlora_defaults_true_on_cuda() -> None:
    assert _use_qlora({}, "cuda") is True
    assert _use_qlora({"qlora": True}, "cuda") is True


def test_use_qlora_disabled_for_bf16_lora() -> None:
    assert _use_qlora({"qlora": False}, "cuda") is False
    assert _use_qlora({"qlora": True}, "cpu") is False


def test_chat_template_kwargs_enable_thinking() -> None:
    assert _chat_template_kwargs({}) == {}
    assert _chat_template_kwargs({"enable_thinking": False}) == {"enable_thinking": False}
    assert _chat_template_kwargs({"enable_thinking": True}) == {"enable_thinking": True}


def test_balanced_sample_weights_upweights_minority() -> None:
    labels = np.array([0, 0, 0, 0, 1], dtype=int)
    weights = _balanced_sample_weights(labels)
    assert weights.shape == (5,)
    assert weights[4] > weights[0]
    assert np.isclose(weights[0], weights[1])
    assert np.isclose(weights.sum(), 5.0)


def test_hate_prob_from_logits_row_prefers_hate_token() -> None:
    row = np.full(8, -5.0, dtype=np.float64)
    row[2] = 4.0
    row[5] = 0.0
    prob = hate_prob_from_logits_row(row, hate_token_ids=[2], non_hate_token_ids=[5])
    assert prob > 0.9


def test_hate_prob_from_logits_row_ties_at_half() -> None:
    row = np.zeros(4, dtype=np.float64)
    prob = hate_prob_from_logits_row(row, hate_token_ids=[1], non_hate_token_ids=[2])
    assert np.isclose(prob, 0.5)
