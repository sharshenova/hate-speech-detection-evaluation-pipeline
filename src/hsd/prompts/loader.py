"""Load YAML prompt configs for zero-shot evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hsd.data.paths import REPO_ROOT


def _resolve_repo_path(path: str | Path) -> Path:
    config_path = Path(path)
    if config_path.is_file():
        return config_path
    resolved = REPO_ROOT / config_path
    if not resolved.is_file():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)
    return resolved


def _load_few_shot_examples_file(
    path: str | Path,
    *,
    interleave: bool = False,
) -> list[dict[str, Any]]:
    """Load hate/non-hate examples from a local YAML."""
    examples_path = _resolve_repo_path(path)
    with examples_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        msg = f"Few-shot examples root must be a mapping: {examples_path}"
        raise ValueError(msg)

    hate_rows: list[dict[str, Any]] = []
    non_hate_rows: list[dict[str, Any]] = []
    for key, label, bucket in (
        ("hate_examples", "hate", hate_rows),
        ("non_hate_examples", "non-hate", non_hate_rows),
    ):
        for row in payload.get(key) or []:
            bucket.append(
                {
                    "source_post_id": row.get("post_id"),
                    "category": row.get("category"),
                    "post": str(row["text"]).strip(),
                    "label": str(row.get("label", label)).strip(),
                }
            )

    if interleave:
        examples: list[dict[str, Any]] = []
        for index in range(max(len(hate_rows), len(non_hate_rows))):
            if index < len(hate_rows):
                examples.append(hate_rows[index])
            if index < len(non_hate_rows):
                examples.append(non_hate_rows[index])
    else:
        examples = [*hate_rows, *non_hate_rows]

    if not examples:
        msg = f"No few-shot examples found in {examples_path}"
        raise ValueError(msg)
    return examples


def load_prompt_config(path: str | Path) -> dict[str, Any]:
    """Load a prompt YAML config."""
    config_path = Path(path)
    if not config_path.is_file():
        msg = f"Prompt config not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        msg = f"Prompt config root must be a mapping: {config_path}"
        raise ValueError(msg)

    required = ("prompt_id", "dataset", "messages", "output")
    missing = [key for key in required if key not in config]
    if missing:
        msg = f"Missing prompt config keys: {missing}"
        raise ValueError(msg)

    examples_path = config.get("few_shot_examples_path")
    if examples_path:
        config = dict(config)
        config["few_shot_examples"] = _load_few_shot_examples_file(
            str(examples_path),
            interleave=bool(config.get("few_shot_interleave", False)),
        )

    return config

def resolve_prompt_config_path(path: str | Path) -> Path:
    """Resolve a prompt config path relative to the repo root."""
    return _resolve_repo_path(path)