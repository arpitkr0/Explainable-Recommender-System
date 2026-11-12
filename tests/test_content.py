"""
Tests for Section 9 (Product Content Representation) and Section 10 (User Profile).

Verifies:
  1. Every product gets a TF-IDF vector.
  2. Every user with rating history gets a profile vector.
  3. A user with no history correctly falls back to the popularity-based path instead of erroring.
  4. Non-negative rating weights (5★: 1.0, 4★: 0.75, 1–3★: 0.0).
  5. Popularity fallback diversification across categories.
"""

import numpy as np
import pandas as pd
import pytest

from src.content.text_features import ContentVectorizer, build_combined_text, clean_text
from src.content.user_profile import (
    DEFAULT_RATING_WEIGHTS,
    PopularityRecommender,
    UserProfileBuilder,
    get_rating_weight,
)


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def sample_metadata():
    return pd.DataFrame(
        [
            {
                "parent_asin": "PROD_1",
                "title": "Wireless Bluetooth Headphones Over-Ear",
                "features": ["Noise cancelling", "30 hour battery", "Built-in mic"],
                "description": ["Premium sound quality headphones with deep bass"],
                "categories": ["Electronics", "Headphones", "Over-Ear"],
                "store": "SoundBrand",
                "main_category": "Headphones",
                "average_rating": 4.8,
                "rating_number": 1200,
            },
            {
                "parent_asin": "PROD_2",
                "title": "USB-C Fast Charging Cable 6ft Braided",
                "features": ["Durable nylon", "Fast charging", "USB 3.0"],
                "description": ["Heavy duty charging cable for phones and laptops"],
                "categories": ["Electronics", "Accessories", "Cables"],
                "store": "CableTech",
                "main_category": "Accessories",
                "average_rating": 4.5,
                "rating_number": 5000,
            },
            {
                "parent_asin": "PROD_3",
                "title": "Mechanical Gaming Keyboard RGB Backlit",
                "features": ["Blue switches", "Customizable lighting", "Anti-ghosting"],
                "description": ["Tactile mechanical keyboard for PC gamers"],
                "categories": ["Electronics", "Computers", "Keyboards"],
                "store": "GameGear",
                "main_category": "Computers",
                "average_rating": 4.6,
                "rating_number": 800,
            },
            {
                "parent_asin": "PROD_4",
                "title": "4K Ultra HD Action Camera Waterproof",
                "features": ["Wide angle lens", "Waterproof case", "WiFi control"],
                "description": ["Capture outdoor sports adventures in 4K resolution"],
                "categories": ["Electronics", "Camera & Photo", "Action Cameras"],
                "store": "CamPro",
                "main_category": "Camera & Photo",
                "average_rating": 4.2,
                "rating_number": 350,
            },
            {
                "parent_asin": "PROD_5",
                "title": "Budget Wired Earbuds 3.5mm Jack",
                "features": ["In-ear fit", "Tangle free cable"],
                "description": ["Basic affordable earbuds"],
                "categories": ["Electronics", "Headphones", "Earbuds"],
                "store": "BasicSound",
                "main_category": "Headphones",
                "average_rating": 3.2,
                "rating_number": 40,
            },
        ]
    )


@pytest.fixture
def sample_reviews():
    return pd.DataFrame(
        [
            # User 1: loves headphones and audio
            {"user_id": "USER_1", "parent_asin": "PROD_1", "rating": 5.0, "timestamp": 1600000000},
            {"user_id": "USER_1", "parent_asin": "PROD_2", "rating": 4.0, "timestamp": 1601000000},
            # User 2: loves gaming gear, disliked budget earbuds
            {"user_id": "USER_2", "parent_asin": "PROD_3", "rating": 5.0, "timestamp": 1602000000},
            {"user_id": "USER_2", "parent_asin": "PROD_5", "rating": 1.0, "timestamp": 1603000000},
            # User 3: only rated neutral / negative
            {"user_id": "USER_3", "parent_asin": "PROD_5", "rating": 2.0, "timestamp": 1604000000},
            {"user_id": "USER_3", "parent_asin": "PROD_2", "rating": 3.0, "timestamp": 1605000000},
        ]
    )


# ── Section 9 Tests: Text Features & TF-IDF ──────────────────────────
def test_clean_text():
    raw = "Sony WH-1000XM4 Noise-Cancelling! Headphones (Black)..."
    cleaned = clean_text(raw)
    assert "wh 1000xm4" in cleaned
    assert "!" not in cleaned
    assert "(" not in cleaned
    assert ")" not in cleaned
    assert "." not in cleaned
    # Multiple spaces collapsed
    assert "  " not in cleaned


def test_build_combined_text(sample_metadata):
    combined = build_combined_text(sample_metadata)
    assert len(combined) == len(sample_metadata)

    # Check PROD_1 contains keywords from title, features, desc, categories, store
    text_prod1 = combined["PROD_1"]
    assert "headphones" in text_prod1
    assert "battery" in text_prod1
    assert "bass" in text_prod1
    assert "soundbrand" in text_prod1
    assert "over ear" in text_prod1


def test_every_product_gets_tfidf_vector(sample_metadata):
    """Every product in metadata must receive a valid, non-zero TF-IDF vector."""
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(sample_metadata)

    # Matrix dimensions
    assert matrix.shape[0] == len(sample_metadata)
    assert matrix.shape[1] > 0

    # Verify lookup and non-zero norm for EVERY product
    for asin in sample_metadata["parent_asin"]:
        assert vectorizer.has_item(asin)
        vec = vectorizer.get_item_vector(asin)
        assert vec is not None
        assert vec.shape == (1, matrix.shape[1])
        # Non-zero vector (norm > 0)
        norm = np.linalg.norm(vec.toarray())
        assert norm > 0.0, f"Product {asin} has an all-zero vector!"


# ── Section 10 Tests: Weighting & User Profiles ──────────────────────
def test_rating_weight_mapping():
    """Verify Section 10 non-negative rating weights."""
    assert get_rating_weight(5.0) == 1.0
    assert get_rating_weight(4.0) == 0.75
    assert get_rating_weight(3.0) == 0.0  # neutral excluded
    assert get_rating_weight(2.0) == 0.0  # negative excluded
    assert get_rating_weight(1.0) == 0.0  # negative excluded

    # Test edge values / floats
    assert get_rating_weight(4.8) == 1.0
    assert get_rating_weight(3.7) == 0.75
    assert get_rating_weight(0.0) == 0.0


def test_every_user_with_rating_history_gets_profile_vector(
    sample_metadata, sample_reviews
):
    """Every user with rating history must produce a valid unit-norm profile vector."""
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(sample_metadata)

    builder = UserProfileBuilder(vectorizer=vectorizer)

    for user_id in ["USER_1", "USER_2", "USER_3"]:
        profile = builder.build_user_profile(user_id, sample_reviews)
        assert profile is not None, f"Failed to build profile for {user_id} with rating history"
        assert isinstance(profile, np.ndarray)
        assert profile.shape == (vectorizer.num_features,)
        # Must be unit normalized (L2 norm == 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(profile), 1.0, decimal=5)


def test_positive_ratings_weight_proportions(sample_metadata, sample_reviews):
    """Verify that 5-star items receive 1.0 weight and 4-star items receive 0.75 weight."""
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(sample_metadata)

    builder = UserProfileBuilder(vectorizer=vectorizer)

    # USER_1 rated PROD_1 (5.0, weight 1.0) and PROD_2 (4.0, weight 0.75)
    profile_u1 = builder.build_user_profile("USER_1", sample_reviews)

    v1 = vectorizer.get_item_vector("PROD_1").toarray().flatten()
    v2 = vectorizer.get_item_vector("PROD_2").toarray().flatten()
    expected_unnorm = (1.0 * v1) + (0.75 * v2)
    expected_norm = expected_unnorm / np.linalg.norm(expected_unnorm)

    np.testing.assert_allclose(profile_u1, expected_norm, atol=1e-5)


def test_user_with_no_history_falls_back_to_popularity_instead_of_error(
    sample_metadata, sample_reviews
):
    """A user with no history must correctly fall back to the popularity-based path instead of erroring."""
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(sample_metadata)

    pop_rec = PopularityRecommender(sample_metadata, sample_reviews)
    builder = UserProfileBuilder(
        vectorizer=vectorizer, popularity_recommender=pop_rec
    )

    # Unknown user with zero history in reviews
    profile, is_cold_start, fallback = builder.get_profile_or_fallback(
        "BRAND_NEW_USER_WITH_NO_HISTORY", sample_reviews, top_k=3
    )
    assert profile is None
    assert is_cold_start is True
    assert isinstance(fallback, list)
    assert len(fallback) == 3
    # Check fallback items are valid products from catalog
    for item in fallback:
        assert item in sample_metadata["parent_asin"].values

    # User queried with empty reviews DataFrame
    profile_empty, is_cold_empty, fallback_empty = builder.get_profile_or_fallback(
        "ANY_USER", pd.DataFrame(), top_k=2
    )
    assert profile_empty is None
    assert is_cold_empty is True
    assert len(fallback_empty) == 2


def test_popularity_recommender_category_diversity(sample_metadata):
    """Verify popularity recommendations spread across categories."""
    pop_rec = PopularityRecommender(sample_metadata, category_col="main_category")
    # Request top 3 with max 1 per category
    recs = pop_rec.recommend(top_k=3, max_per_category=1)

    categories_selected = [
        sample_metadata.loc[sample_metadata["parent_asin"] == asin, "main_category"].values[0]
        for asin in recs
    ]
    # Each chosen item must be from a distinct category
    assert len(categories_selected) == len(set(categories_selected))


def test_user_profile_recency_decay(sample_metadata):
    """With recency_decay < 1.0, newer positive interactions have higher weight than older ones."""
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(sample_metadata)

    revs = pd.DataFrame(
        [
            {"user_id": "U_DECAY", "parent_asin": "PROD_1", "rating": 5.0, "timestamp": 1000},  # Older
            {"user_id": "U_DECAY", "parent_asin": "PROD_3", "rating": 5.0, "timestamp": 2000},  # Newer
        ]
    )
    builder = UserProfileBuilder(vectorizer=vectorizer, recency_decay=0.5)
    profile = builder.build_user_profile("U_DECAY", revs)
    assert profile is not None

    v_old = vectorizer.get_item_vector("PROD_1").toarray().flatten()
    v_new = vectorizer.get_item_vector("PROD_3").toarray().flatten()
    assert float(np.dot(profile, v_new)) > float(np.dot(profile, v_old))
