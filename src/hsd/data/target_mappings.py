"""Harmonised multilabel target flags and per-dataset token mappings."""

from __future__ import annotations

from typing import Final

from hsd.data.schema import (
    TARGET_FLAG_COLUMNS,
    TGT_GEN,
    TGT_NAT,
    TGT_OPC,
    TGT_OTHER,
    TGT_RAE,
    TGT_REL,
    TGT_SXO,
    TGT_UNKNOWN,
)

# Sentinel target strings in raw DGHD exports (not mapped via token dict).
DGHD_TARGET_NONE: Final[str] = "none"
DGHD_TARGET_NOTGIVEN: Final[str] = "notgiven"
DGHD_TARGET_NOTARGETRECORDED: Final[str] = "notargetrecorded"
DGHD_UNKNOWN_TARGETS: Final[frozenset[str]] = frozenset(
    {DGHD_TARGET_NOTGIVEN, DGHD_TARGET_NOTARGETRECORDED}
)

# Map each DGHD shorthand token to harmonised target flag column(s) to set to 1.
# DGHD `pol` = Polish people -> nat, not `politics`).
DGHD_TARGET_TOKEN_FLAGS: Final[dict[str, frozenset[str]]] = {
    # Race / ethnicity
    "african": frozenset({TGT_RAE}),
    "arab": frozenset({TGT_RAE}),
    "asi": frozenset({TGT_RAE}),
    "asi.east": frozenset({TGT_RAE}),
    "asi.man": frozenset({TGT_RAE}),
    "asi.south": frozenset({TGT_RAE}),
    "asi.wom": frozenset({TGT_RAE, TGT_GEN}),
    "bla": frozenset({TGT_RAE}),
    "bla.man": frozenset({TGT_RAE}),
    "bla.wom": frozenset({TGT_RAE, TGT_GEN}),
    "ethnic.minority": frozenset({TGT_RAE}),
    "hispanic": frozenset({TGT_RAE}),
    "indig": frozenset({TGT_RAE}),
    "indig.wom": frozenset({TGT_RAE, TGT_GEN}),
    "mixed.race": frozenset({TGT_RAE}),
    "non.white": frozenset({TGT_RAE}),
    "non.white.wom": frozenset({TGT_RAE, TGT_GEN}),
    "trav": frozenset({TGT_RAE}),
    # Religion
    "jew": frozenset({TGT_REL}),
    "mus": frozenset({TGT_REL}),
    "mus.wom": frozenset({TGT_REL, TGT_GEN}),
    "other.religion": frozenset({TGT_REL}),
    # Nationality / regionalism / migration (continued)
    "asi.chin": frozenset({TGT_NAT}),
    "asi.pak": frozenset({TGT_NAT}),
    "asylum": frozenset({TGT_NAT}),
    "eastern.europe": frozenset({TGT_NAT}),
    "for": frozenset({TGT_NAT}),
    "immig": frozenset({TGT_NAT}),
    "other.national": frozenset({TGT_NAT}),
    "pol": frozenset({TGT_NAT}),
    "ref": frozenset({TGT_NAT}),
    "russian": frozenset({TGT_NAT}),
    # Gender (trans -> gender identity)
    "gendermin": frozenset({TGT_GEN}),
    "trans": frozenset({TGT_GEN}),
    "wom": frozenset({TGT_GEN}),
    # Sexual orientation
    "bis": frozenset({TGT_SXO}),
    "gay": frozenset({TGT_SXO}),
    "gay.man": frozenset({TGT_SXO}),
    "gay.wom": frozenset({TGT_SXO, TGT_GEN}),
    "lgbtq": frozenset({TGT_SXO}),
    # Other protected (disability / health)
    "dis": frozenset({TGT_OPC}),
    # Extended categories (political glorification, class, age, ...)
    "hitler": frozenset({TGT_OTHER}),
    "nazis": frozenset({TGT_OTHER}),
    "old.people": frozenset({TGT_OTHER}),
    "other.glorification": frozenset({TGT_OTHER}),
    "wc": frozenset({TGT_OTHER}),
}

# Map HateCheck `target_ident` to harmonised flags.
# trans people -> gender.
HATECHECK_TARGET_IDENT_FLAGS: Final[dict[str, frozenset[str]]] = {
    "black people": frozenset({TGT_RAE}),
    "disabled people": frozenset({TGT_OPC}),
    "gay people": frozenset({TGT_SXO}),
    "immigrants": frozenset({TGT_NAT}),
    "Muslims": frozenset({TGT_REL}),
    "trans people": frozenset({TGT_GEN}),
    "women": frozenset({TGT_GEN}),
}

# Map GHC aggregated target columns to harmonised flags.
# idl/pol -> tgt_other; mph (disability) -> tgt_opc.
GHC_TARGET_COLUMN_FLAGS: Final[dict[str, frozenset[str]]] = {
    "rae": frozenset({TGT_RAE}),
    "rel": frozenset({TGT_REL}),
    "nat": frozenset({TGT_NAT}),
    "gen": frozenset({TGT_GEN}),
    "sxo": frozenset({TGT_SXO}),
    "idl": frozenset({TGT_OTHER}),
    "pol": frozenset({TGT_OTHER}),
    "mph": frozenset({TGT_OPC}),
}

GHC_TARGET_SOURCE_COLUMNS: Final[tuple[str, ...]] = tuple(GHC_TARGET_COLUMN_FLAGS.keys())

# Map KTC integer `target` code to tgt_* flags (hate rows only; HateXplain + MMHS150K).
KTC_TARGET_CODE_FLAGS: Final[dict[int, frozenset[str]]] = {
    -1: frozenset(),
    0: frozenset({TGT_RAE}),
    1: frozenset({TGT_REL}),
    2: frozenset({TGT_NAT}),
    3: frozenset({TGT_GEN}),
    4: frozenset({TGT_SXO}),
    5: frozenset({TGT_OTHER}),
    6: frozenset({TGT_OPC}),
}

def empty_target_flags() -> dict[str, int]:
    """Return all harmonised target flags set to 0."""
    return dict.fromkeys(TARGET_FLAG_COLUMNS, 0)


def _apply_token_flags(
    flags: dict[str, int],
    token_map: dict[str, frozenset[str]],
    raw_tokens: str,
    *,
    separator: str = ",",
) -> None:
    for part in raw_tokens.split(separator):
        token = part.strip()
        if not token:
            continue
        mapped = token_map.get(token)
        if mapped is None:
            msg = f"Unmapped target token: {token!r}"
            raise KeyError(msg)
        for column in mapped:
            flags[column] = 1


def target_flags_from_dghd(raw_target: object, *, hate: int) -> dict[str, int]:
    """Map DGHD target tokens while preserving its sentinel meanings.

    On hate rows, ``none`` means no target, while missing, ``notgiven``, and
    ``notargetrecorded`` set the unknown flag. Non-hate rows keep all flags zero.
    """
    flags = empty_target_flags()
    if hate == 0:
        return flags

    if raw_target is None or (isinstance(raw_target, float) and pd_is_na(raw_target)):
        flags[TGT_UNKNOWN] = 1
        return flags

    target = str(raw_target).strip()
    if target in {"", "NA", "nan"}:
        flags[TGT_UNKNOWN] = 1
        return flags
    if target == DGHD_TARGET_NONE:
        return flags
    if target in DGHD_UNKNOWN_TARGETS:
        flags[TGT_UNKNOWN] = 1
        return flags

    _apply_token_flags(flags, DGHD_TARGET_TOKEN_FLAGS, target)
    return flags


def target_flags_from_ghc(row: object, *, hate: int) -> dict[str, int]:
    """Map GHC target votes, marking a hate row unknown when none are active."""
    flags = empty_target_flags()
    if hate == 0:
        return flags

    any_active = False
    for column, mapped in GHC_TARGET_COLUMN_FLAGS.items():
        if _ghc_target_flag_is_active(_row_value(row, column)):
            any_active = True
            for target_column in mapped:
                flags[target_column] = 1

    if not any_active:
        flags[TGT_UNKNOWN] = 1
    return flags


def ghc_target_raw_from_row(row: object) -> str:
    """Build ``target_raw`` from active GHC target column names."""
    active = [
        column
        for column in GHC_TARGET_SOURCE_COLUMNS
        if _ghc_target_flag_is_active(_row_value(row, column))
    ]
    return "|".join(active)


def _row_value(row: object, column: str) -> object:
    if isinstance(row, dict):
        return row.get(column)
    return getattr(row, column, None)


def _ghc_target_flag_is_active(value: object) -> bool:
    if pd_is_na(value):
        return False
    return int(value) == 1


def target_flags_from_hatecheck_ident(raw_ident: object, *, hate: int) -> dict[str, int]:
    """Map the tested identity to target flags on hateful HateCheck cases only."""
    flags = empty_target_flags()
    if hate == 0 or raw_ident is None or (isinstance(raw_ident, float) and pd_is_na(raw_ident)):
        return flags

    ident = str(raw_ident).strip()
    if not ident:
        return flags

    mapped = HATECHECK_TARGET_IDENT_FLAGS.get(ident)
    if mapped is None:
        msg = f"Unmapped HateCheck target_ident: {ident!r}"
        raise KeyError(msg)
    for column in mapped:
        flags[column] = 1
    return flags


def target_flags_from_ktc_code(code: object, *, hate: int) -> dict[str, int]:
    """Map KTC target codes on hate rows.

    A missing code sets unknown, while the -1 sentinel means no target category
    and leaves all flags zero.
    """
    flags = empty_target_flags()
    if hate == 0:
        return flags

    if code is None or (isinstance(code, float) and pd_is_na(code)):
        flags[TGT_UNKNOWN] = 1
        return flags

    int_code = int(code)
    mapped = KTC_TARGET_CODE_FLAGS.get(int_code)
    if mapped is None:
        msg = f"Unmapped KTC target code: {int_code}"
        raise KeyError(msg)
    for column in mapped:
        flags[column] = 1
    return flags


def pd_is_na(value: object) -> bool:
    """Return True for pandas-style missing values without importing pandas here."""
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except ImportError:
        return False
