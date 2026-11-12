"""
Data Loader — reads the subsetted Parquet files and applies the
Section 7 data-processing pipeline (duplicate removal, schema check).

Uses parent_asin as the canonical item ID throughout (Section 5).
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REVIEWS_PATH = PROCESSED_DIR / "reviews_subset.parquet"
META_PATH = PROCESSED_DIR / "metadata_subset.parquet"

# ── Required columns (Section 4) ────────────────────────────────────
REVIEW_REQUIRED = ["user_id", "parent_asin", "rating"]
META_REQUIRED = ["parent_asin", "title"]


# ── Config helper ────────────────────────────────────────────────────
def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration dictionary."""
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log.info("Config loaded from %s", path)
    return cfg


# ── Core loader ──────────────────────────────────────────────────────
def load_reviews(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the reviews subset.

    Pipeline (Section 7):
      1. Read Parquet
      2. Duplicate removal
      3. Schema check — no nulls in required fields
    """
    path = Path(path) if path else REVIEWS_PATH
    log.info("Loading reviews from %s", path)
    df = pd.read_parquet(path)
    initial = len(df)

    # ── Duplicate removal ────────────────────────────────────────────
    # A duplicate review is the same user reviewing the same product
    # with the same rating and timestamp.
    df = df.drop_duplicates(subset=["user_id", "parent_asin", "timestamp"])
    dupes = initial - len(df)
    if dupes:
        log.info("Removed %d duplicate review(s)", dupes)

    # ── Schema check — required fields must not be null ──────────────
    for col in REVIEW_REQUIRED:
        n_null = df[col].isna().sum()
        if n_null > 0:
            log.warning("Dropping %d rows with null %s", n_null, col)
            df = df.dropna(subset=[col])

    # ── parent_asin canonicalization (Section 5) ─────────────────────
    # parent_asin is already the canonical ID in our subset.
    # Ensure it is always a string for consistent joins downstream.
    df["parent_asin"] = df["parent_asin"].astype(str)
    if "asin" in df.columns:
        df["asin"] = df["asin"].astype(str)

    log.info(
        "Reviews loaded: %d rows, %d users, %d items",
        len(df),
        df["user_id"].nunique(),
        df["parent_asin"].nunique(),
    )
    return df


def load_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the metadata subset.

    Pipeline (Section 7):
      1. Read Parquet
      2. Duplicate removal (by parent_asin — one catalog entry per product)
      3. Schema check
      4. Deserialize JSON-encoded list columns back into Python lists
    """
    path = Path(path) if path else META_PATH
    log.info("Loading metadata from %s", path)
    df = pd.read_parquet(path)
    initial = len(df)

    # ── Duplicate removal — one row per parent_asin ──────────────────
    df = df.drop_duplicates(subset=["parent_asin"], keep="first")
    dupes = initial - len(df)
    if dupes:
        log.info("Removed %d duplicate metadata row(s)", dupes)

    # ── Schema check ─────────────────────────────────────────────────
    for col in META_REQUIRED:
        n_null = df[col].isna().sum()
        if n_null > 0:
            log.warning("Dropping %d metadata rows with null %s", n_null, col)
            df = df.dropna(subset=[col])

    # ── Canonicalize parent_asin type ────────────────────────────────
    df["parent_asin"] = df["parent_asin"].astype(str)

    # ── Deserialize list columns (stored as JSON strings by subset_data.py)
    for col in ("features", "description", "categories"):
        if col in df.columns:
            df[col] = df[col].apply(_safe_json_loads)

    log.info(
        "Metadata loaded: %d products",
        len(df),
    )
    return df


def _safe_json_loads(val):
    """Parse a JSON string back into a Python list; pass through non-strings."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


# ── Convenience: load both in one call ───────────────────────────────
def load_data(
    config_path: str | Path | None = None,
    reviews_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load config, reviews, and metadata in one call.

    Returns (reviews_df, metadata_df, config_dict).
    """
    cfg = load_config(config_path)
    reviews = load_reviews(reviews_path)
    metadata = load_metadata(meta_path)
    return reviews, metadata, cfg
