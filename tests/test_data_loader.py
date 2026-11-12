"""
Tests for the configuration loader and data loading pipeline.

Confirms:
  - config.yaml loads and contains all expected top-level keys
  - The data loader runs end-to-end on the subsetted files without errors
  - Loaded DataFrames have the expected columns and types
  - parent_asin is consistently typed as string (Section 5 canonicalization)
  - No nulls survive in required fields after cleaning (Section 7)
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_config, load_data, load_metadata, load_reviews
from src.data.cleaner import (
    META_EXPECTED_COLS,
    META_REQUIRED_COLS,
    REVIEW_EXPECTED_COLS,
    REVIEW_REQUIRED_COLS,
    check_no_nulls,
    remove_duplicates,
    validate_schema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════════════
class TestConfig:
    """config.yaml loads and contains all Section 22 keys."""

    def test_config_loads(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_config_has_dataset_section(self):
        cfg = load_config()
        assert "dataset" in cfg
        ds = cfg["dataset"]
        assert "min_ratings_per_user" in ds
        assert "max_users" in ds
        assert "category" in ds

    def test_config_has_retrieval_section(self):
        cfg = load_config()
        assert "retrieval" in cfg
        ret = cfg["retrieval"]
        assert "candidate_count" in ret
        assert "top_k" in ret
        assert "diversity_cap_per_brand" in ret

    def test_config_has_ranking_section(self):
        cfg = load_config()
        assert "ranking" in cfg
        rk = cfg["ranking"]
        assert "model" in rk
        assert rk["model"] == "xgboost"
        assert "positive_rating_threshold" in rk
        assert "negative_rating_threshold" in rk

    def test_config_has_hybrid_section(self):
        cfg = load_config()
        assert "hybrid" in cfg
        hy = cfg["hybrid"]
        assert "alpha" in hy
        assert "beta" in hy
        assert "gamma" in hy

    def test_config_values_are_sensible(self):
        cfg = load_config()
        assert cfg["dataset"]["min_ratings_per_user"] > 0
        assert cfg["dataset"]["max_users"] > 0
        assert 0 <= cfg["hybrid"]["alpha"] <= 1
        assert 0 <= cfg["hybrid"]["beta"] <= 1
        assert 0 <= cfg["hybrid"]["gamma"] <= 1


# ═══════════════════════════════════════════════════════════════════════
# Data loader tests
# ═══════════════════════════════════════════════════════════════════════
class TestDataLoader:
    """The loader reads subsetted Parquet files without errors."""

    @pytest.fixture(scope="class")
    def reviews(self):
        return load_reviews()

    @pytest.fixture(scope="class")
    def metadata(self):
        return load_metadata()

    # ── Reviews ──────────────────────────────────────────────────────
    def test_reviews_loads(self, reviews):
        assert isinstance(reviews, pd.DataFrame)
        assert len(reviews) > 0

    def test_reviews_has_expected_columns(self, reviews):
        for col in REVIEW_EXPECTED_COLS:
            assert col in reviews.columns, f"Missing column: {col}"

    def test_reviews_no_null_required_fields(self, reviews):
        for col in REVIEW_REQUIRED_COLS:
            assert reviews[col].isna().sum() == 0, f"Null values in {col}"

    def test_reviews_parent_asin_is_string(self, reviews):
        assert reviews["parent_asin"].dtype == object  # pandas string dtype

    def test_reviews_rating_range(self, reviews):
        assert reviews["rating"].min() >= 1.0
        assert reviews["rating"].max() <= 5.0

    # ── Metadata ─────────────────────────────────────────────────────
    def test_metadata_loads(self, metadata):
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) > 0

    def test_metadata_has_expected_columns(self, metadata):
        for col in META_EXPECTED_COLS:
            assert col in metadata.columns, f"Missing column: {col}"

    def test_metadata_no_null_required_fields(self, metadata):
        for col in META_REQUIRED_COLS:
            assert metadata[col].isna().sum() == 0, f"Null values in {col}"

    def test_metadata_parent_asin_is_string(self, metadata):
        assert metadata["parent_asin"].dtype == object

    def test_metadata_no_duplicate_parent_asin(self, metadata):
        assert metadata["parent_asin"].is_unique, "Duplicate parent_asin in metadata"

    def test_metadata_list_columns_deserialized(self, metadata):
        """features, description, categories should be Python lists, not JSON strings."""
        for col in ("features", "description", "categories"):
            if col in metadata.columns:
                non_null = metadata[col].dropna()
                if len(non_null) > 0:
                    sample = non_null.iloc[0]
                    assert isinstance(sample, list), (
                        f"{col} should be deserialized to list, got {type(sample)}"
                    )


# ═══════════════════════════════════════════════════════════════════════
# Combined load_data convenience function
# ═══════════════════════════════════════════════════════════════════════
class TestLoadData:
    """The combined load_data function returns all three objects."""

    def test_load_data_returns_tuple(self):
        reviews, metadata, cfg = load_data()
        assert isinstance(reviews, pd.DataFrame)
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(cfg, dict)
