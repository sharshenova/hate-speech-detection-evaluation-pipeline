"""YAML experiment config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from hsd.data.paths import REPO_ROOT


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a top-level experiment YAML config."""
    config_path = Path(path)
    if not config_path.is_file():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        msg = f"Config root must be a mapping: {config_path}"
        raise ValueError(msg)

    required = ("experiment_id", "dataset", "seed", "data", "model", "training", "results_dir")
    missing = [key for key in required if key not in config]
    if missing:
        msg = f"Missing config keys: {missing}"
        raise ValueError(msg)

    return config


def resolve_results_dir(config: dict[str, Any]) -> Path:
    """Resolve the per-dataset results directory for an experiment."""
    base = Path(config["results_dir"])
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / str(config["dataset"])
