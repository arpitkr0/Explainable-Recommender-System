"""
Tests for Section 6 — Dataset Subsetting & Large-File Ingestion

Validates that the two output Parquet files satisfy the spec's invariants:
  1. Every parent_asin in reviews_subset exists in metadata_subset
  2. Every user in reviews_subset has >= min_ratings_per_user interactions
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_subset.parquet"
META_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_subset.parquet"


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def reviews_df():
    assert REVIEWS_PATH.exists(), f"Reviews subset not found at {REVIEWS_PATH}"
    return pd.read_parquet(REVIEWS_PATH)


@pytest.fixture(scope="module")
def meta_df():
    assert META_PATH.exists(), f"Metadata subset not found at {META_PATH}"
    return pd.read_parquet(META_PATH)


# ── Tests ────────────────────────────────────────────────────────────
class TestSubsetIntegrity:
    """Core invariants from Section 6 of the spec."""

    def test_reviews_file_exists(self):
        assert REVIEWS_PATH.exists(), "reviews_subset.parquet must exist"

    def test_metadata_file_exists(self):
        assert META_PATH.exists(), "metadata_subset.parquet must exist"

    def test_parent_asin_coverage(self, reviews_df, meta_df):
        """Every parent_asin that appears in the reviews must have a
        corresponding row in the metadata."""
        review_asins = set(reviews_df["parent_asin"].unique())
        meta_asins = set(meta_df["parent_asin"].unique())
        missing = review_asins - meta_asins
        assert not missing, (
            f"{len(missing)} parent_asin(s) in reviews are missing from metadata. "
            f"Examples: {list(missing)[:10]}"
        )

    def test_min_ratings_per_user(self, config, reviews_df):
        """Every user in the reviews subset must have at least
        min_ratings_per_user interactions."""
        min_ratings = config["dataset"]["min_ratings_per_user"]
        user_counts = reviews_df.groupby("user_id").size()
        violating = user_counts[user_counts < min_ratings]
        assert violating.empty, (
            f"{len(violating)} user(s) have fewer than {min_ratings} interactions. "
            f"Min observed: {violating.min()}"
        )

    def test_max_users_respected(self, config, reviews_df):
        """The number of distinct users should not exceed max_users."""
        max_users = config["dataset"]["max_users"]
        n_users = reviews_df["user_id"].nunique()
        assert n_users <= max_users, (
            f"Found {n_users} users, but max_users is {max_users}"
        )

    def test_reviews_not_empty(self, reviews_df):
        assert len(reviews_df) > 0, "reviews_subset must contain at least one row"

    def test_metadata_not_empty(self, meta_df):
        assert len(meta_df) > 0, "metadata_subset must contain at least one row"

    def test_no_null_required_fields(self, reviews_df):
        """Required fields (user_id, parent_asin, rating) must not be null."""
        for col in ["user_id", "parent_asin", "rating"]:
            n_null = reviews_df[col].isna().sum()
            assert n_null == 0, f"Found {n_null} null values in reviews_subset.{col}"
