"""Parse zero-shot / generative labels."""

from __future__ import annotations

import re
from typing import Any


def strip_thinking_content(raw: str) -> str:
    """Remove Gemma/Ollama thinking channels; keep the final answer text."""
    text = str(raw)
    text = re.sub(
        r"<\|channel>thought\n.*?<channel\|>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|think\|>", "", text)
    return text.strip()


def normalize_label(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.strip("\"'`.,:;!?")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_zero_shot_label(raw: str, output_cfg: dict[str, Any]) -> tuple[int | None, str]:
    """Map a model response to binary hate (1) / non-hate (0), or None.

    Candidate order resolves ambiguous multi-line answers by preferring the final line.
    """

    answer = strip_thinking_content(raw)
    normalized = normalize_label(answer)
    if not normalized:
        return None, normalized

    hate_labels = {
        normalize_label(str(label)) for label in output_cfg.get("parse_as_hate", ["hate"])
    }
    non_hate_labels = {
        normalize_label(str(label))
        for label in output_cfg.get("parse_as_non_hate", ["non-hate"])
    }

    lines = [line.strip() for line in str(answer).splitlines() if line.strip()]
    # Models often put the label after an explanation. The first line and full
    # answer are fallbacks for short answers and prompts that put the label first.
    candidate_strings = [*lines[-1:], *lines[:1], normalized]

    for text in candidate_strings:
        norm = normalize_label(text)
        if not norm:
            continue
        first_token = norm.split()[0]
        if first_token in hate_labels or first_token == "hate":
            return 1, norm
        if (
            first_token in non_hate_labels
            or first_token.startswith("non")
            or first_token in {"no", "not"}
        ):
            return 0, norm

    # Limit this final substring fallback so incidental labels in a long answer
    # do not decide the result.
    if len(normalized.split()) <= 6:
        if "non-hate" in normalized or "non hate" in normalized or "not hate" in normalized:
            return 0, normalized
        if re.search(r"\bhate\b", normalized):
            return 1, normalized

    return None, normalized
