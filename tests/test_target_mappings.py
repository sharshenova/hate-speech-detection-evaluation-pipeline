"""Tests for harmonised target mapping dictionaries."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hsd.data.schema import (
    TGT_GEN,
    TGT_NAT,
    TGT_OPC,
    TGT_OTHER,
    TGT_RAE,
    TGT_REL,
    TGT_SXO,
    TGT_UNKNOWN,
)
from hsd.data.target_mappings import (
    DGHD_TARGET_TOKEN_FLAGS,
    target_flags_from_dghd,
    target_flags_from_ghc,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DGHD_CSV = REPO_ROOT / "data/raw/dghd/Dynamically Generated Hate Dataset v0.2.3.csv"


def _dghd_tokens() -> set[str]:
    tokens: set[str] = set()
    with DGHD_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("target") or "").strip()
            if raw in {"", "none", "notgiven", "notargetrecorded", "NA"}:
                continue
            tokens.update(part.strip() for part in raw.split(",") if part.strip())
    return tokens


@pytest.mark.skipif(not DGHD_CSV.exists(), reason="DGHD raw file not available locally")
def test_all_dghd_tokens_are_mapped() -> None:
    tokens = _dghd_tokens()
    missing = sorted(tokens - set(DGHD_TARGET_TOKEN_FLAGS))
    assert not missing, f"Unmapped DGHD tokens: {missing}"


def test_dghd_gay_wom_sets_sxo_and_gen() -> None:
    flags = target_flags_from_dghd("gay.wom", hate=1)
    assert flags[TGT_SXO] == 1
    assert flags[TGT_GEN] == 1


def test_dghd_intersectional_example() -> None:
    flags = target_flags_from_dghd("trans, gay.wom, gay.man, bis", hate=1)
    assert flags[TGT_GEN] == 1
    assert flags[TGT_SXO] == 1
    assert flags[TGT_RAE] == 0
    assert flags[TGT_REL] == 0
    assert flags[TGT_NAT] == 0


def test_dghd_race_religion_example() -> None:
    flags = target_flags_from_dghd("mus, bla, non.white, african", hate=1)
    assert flags[TGT_REL] == 1
    assert flags[TGT_RAE] == 1
    assert flags[TGT_NAT] == 0


def test_dghd_polish_maps_to_nat_not_other() -> None:
    flags = target_flags_from_dghd("pol", hate=1)
    assert flags[TGT_NAT] == 1
    assert flags[TGT_OTHER] == 0


def test_dghd_notgiven_maps_to_unknown() -> None:
    flags = target_flags_from_dghd("notgiven", hate=1)
    assert flags[TGT_UNKNOWN] == 1
    assert sum(flags.values()) == 1


def test_dghd_nothate_all_zero() -> None:
    flags = target_flags_from_dghd("none", hate=0)
    assert all(value == 0 for value in flags.values())


def test_dghd_travellers_map_to_rae() -> None:
    flags = target_flags_from_dghd("trav", hate=1)
    assert flags[TGT_RAE] == 1
    assert flags[TGT_NAT] == 0


def test_dghd_bla_man_does_not_set_gen() -> None:
    flags = target_flags_from_dghd("bla.man", hate=1)
    assert flags[TGT_RAE] == 1
    assert flags[TGT_GEN] == 0


def test_ghc_pol_maps_to_other() -> None:
    flags = target_flags_from_ghc({"pol": 1}, hate=1)
    assert flags[TGT_OTHER] == 1


def test_ghc_non_hate_all_zero() -> None:
    flags = target_flags_from_ghc({"pol": 1, "rae": 1}, hate=0)
    assert all(value == 0 for value in flags.values())
