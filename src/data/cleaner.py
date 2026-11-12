"""
Data Cleaner — validation and cleanup utilities used by the loader.

Section 7: duplicate removal, schema validation, and any light
cleanup that runs on the already-subsetted files.
"""

import logging

import pandas as pd

log = logging.getLogger(__name__)

# ── Required columns per entity (Section 4) ──────────────────────────
REVIEW_REQUIRED_COLS = ["user_id", "parent_asin", "rating"]
META_REQUIRED_COLS = ["parent_asin", "title"]

REVIEW_EXPECTED_COLS = [
    "rating", "title", "text", "asin", "parent_asin",
    "user_id", "timestamp", "verified_purchase",
]
META_EXPECTED_COLS = [
    "parent_asin", "main_category", "title", "average_rating",
    "rating_number", "features", "description", "price",
    "store", "categories",
]


def validate_schema(df: pd.DataFrame, expected_cols: list[str], name: str) -> None:
    """Raise if any expected column is missing from the DataFrame."""
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    log.debug("Schema OK for %s — all %d expected columns present", name, len(expected_cols))


def check_no_nulls(df: pd.DataFrame, required_cols: list[str], name: str) -> pd.DataFrame:
    """Drop rows where any required column is null and log warnings."""
    before = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before - len(df)
    if dropped:
        log.warning("%s: dropped %d rows with nulls in required columns %s", name, dropped, required_cols)
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str],
    name: str,
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicate rows based on the given column subset."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    dropped = before - len(df)
    if dropped:
        log.info("%s: removed %d duplicate(s) on %s", name, dropped, subset)
    return df
