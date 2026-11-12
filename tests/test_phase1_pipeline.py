"""
Unit and Integration Tests for Phase 1 Recommendation Pipeline.

Validates:
  1. Section 8: Chronological train/test split and strict no-temporal-leakage invariant.
  2. Section 11: Candidate generation with already-interacted filtering and brand diversity cap.
  3. Section 12: Ranking model feature extraction and explicit 4/2/3 labeling rule.
  4. Section 13: SHAP TreeExplainer feature contributions.
  5. Section 14: Template-based explanation engine.
  6. Section 23: recommend(user_id, k) API contract compliance for both active and cold-start users.
"""

import numpy as np
import pandas as pd
import pytest

from src.content.text_features import ContentVectorizer
from src.content.user_profile import PopularityRecommender, UserProfileBuilder
from src.data.splitter import assert_no_temporal_leakage, temporal_train_test_split
from src.explainability.explanation_generator import ExplanationEngine
from src.explainability.shap_explainer import ShapExplainer
from src.pipeline.recommender import Phase1Recommender
from src.ranking.features import FEATURE_NAMES, FeatureExtractor
from src.ranking.ranker import CandidateGenerator, XGBoostRanker


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def mock_catalog():
    return pd.DataFrame(
        [
            {
                "parent_asin": "SONY_WH1000",
                "title": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
                "features": ["Noise cancelling", "30 hour battery", "Crystal clear hands-free calling"],
                "description": ["Premium wireless over-ear noise cancelling headphones"],
                "categories": ["Electronics", "Headphones", "Over-Ear"],
                "store": "Sony",
                "main_category": "Headphones",
                "price": 399.99,
                "average_rating": 4.7,
                "rating_number": 8500,
            },
            {
                "parent_asin": "SONY_WF1000",
                "title": "Sony WF-1000XM4 Noise Canceling Wireless Earbuds",
                "features": ["Industry leading noise canceling", "Integrated Processor V1"],
                "description": ["Truly wireless earbuds with high resolution audio"],
                "categories": ["Electronics", "Headphones", "Earbuds"],
                "store": "Sony",
                "main_category": "Headphones",
                "price": 279.99,
                "average_rating": 4.5,
                "rating_number": 6200,
            },
            {
                "parent_asin": "SONY_SPEAKER",
                "title": "Sony SRS-XB13 Extra Bass Wireless Compact Speaker",
                "features": ["Extra bass", "Waterproof and dustproof IP67"],
                "description": ["Portable Bluetooth speaker with deep sound"],
                "categories": ["Electronics", "Audio", "Speakers"],
                "store": "Sony",
                "main_category": "Audio",
                "price": 59.99,
                "average_rating": 4.6,
                "rating_number": 4300,
            },
            {
                "parent_asin": "BOSE_QC45",
                "title": "Bose QuietComfort 45 Bluetooth Wireless Headphones",
                "features": ["High-fidelity audio", "Quiet and Aware modes"],
                "description": ["Iconic quiet comfort and acclaimed noise cancelling"],
                "categories": ["Electronics", "Headphones", "Over-Ear"],
                "store": "Bose",
                "main_category": "Headphones",
                "price": 329.00,
                "average_rating": 4.6,
                "rating_number": 7100,
            },
            {
                "parent_asin": "ANKER_CABLE",
                "title": "Anker USB C Charger Cable Powerline 6ft",
                "features": ["Fast charging", "Heavy duty nylon braid"],
                "description": ["Durable USB-C charging cable"],
                "categories": ["Electronics", "Accessories", "Cables"],
                "store": "Anker",
                "main_category": "Accessories",
                "price": 14.99,
                "average_rating": 4.8,
                "rating_number": 12000,
            },
            {
                "parent_asin": "LOGI_MOUSE",
                "title": "Logitech MX Master 3S Wireless Performance Mouse",
                "features": ["Quiet clicks", "8K DPI sensor", "Ergonomic design"],
                "description": ["High precision productivity mouse for Mac and Windows"],
                "categories": ["Electronics", "Computers", "Mice"],
                "store": "Logitech",
                "main_category": "Computers",
                "price": 99.99,
                "average_rating": 4.7,
                "rating_number": 9500,
            },
        ]
    )


@pytest.fixture
def mock_reviews():
    return pd.DataFrame(
        [
            # User A: Audio enthusiast with chronological history
            {"user_id": "USER_A", "parent_asin": "SONY_WH1000", "rating": 5.0, "timestamp": 1600000000},
            {"user_id": "USER_A", "parent_asin": "BOSE_QC45", "rating": 4.0, "timestamp": 1610000000},
            {"user_id": "USER_A", "parent_asin": "ANKER_CABLE", "rating": 5.0, "timestamp": 1620000000},
            {"user_id": "USER_A", "parent_asin": "SONY_WF1000", "rating": 5.0, "timestamp": 1630000000},  # Test item
            # User B: PC user
            {"user_id": "USER_B", "parent_asin": "LOGI_MOUSE", "rating": 5.0, "timestamp": 1600500000},
            {"user_id": "USER_B", "parent_asin": "ANKER_CABLE", "rating": 2.0, "timestamp": 1610500000},
            {"user_id": "USER_B", "parent_asin": "BOSE_QC45", "rating": 3.0, "timestamp": 1620500000},  # Ambiguous rating
            {"user_id": "USER_B", "parent_asin": "SONY_SPEAKER", "rating": 4.0, "timestamp": 1630500000},  # Test item
        ]
    )


# ── 1. Section 8 Tests: Temporal Split & Leakage Invariant ────────────
def test_temporal_split_order_and_sizes(mock_reviews):
    train_df, test_df = temporal_train_test_split(mock_reviews, holdout_k=1)

    # 1 holdout per user
    assert len(test_df) == 2
    assert set(test_df["user_id"]) == {"USER_A", "USER_B"}

    # USER_A test item must be SONY_WF1000 (most recent)
    user_a_test = test_df[test_df["user_id"] == "USER_A"].iloc[0]
    assert user_a_test["parent_asin"] == "SONY_WF1000"

    # USER_B test item must be SONY_SPEAKER (most recent)
    user_b_test = test_df[test_df["user_id"] == "USER_B"].iloc[0]
    assert user_b_test["parent_asin"] == "SONY_SPEAKER"

    # Invariant: max train timestamp <= min test timestamp
    assert_no_temporal_leakage(train_df, test_df)


def test_strict_temporal_leakage_assertion(mock_reviews, mock_catalog):
    train_df, test_df = temporal_train_test_split(mock_reviews, holdout_k=1)

    # Build profiles from TRAIN ONLY
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(mock_catalog)
    builder = UserProfileBuilder(vectorizer=vectorizer)

    train_profiles_items = {
        "USER_A": train_df[train_df["user_id"] == "USER_A"]["parent_asin"].tolist(),
        "USER_B": train_df[train_df["user_id"] == "USER_B"]["parent_asin"].tolist(),
    }

    # Should pass cleanly without error
    assert_no_temporal_leakage(train_df, test_df, user_profiles_items=train_profiles_items)

    # Intentionally corrupt profile history to include test item
    corrupted_profiles_items = {
        "USER_A": train_profiles_items["USER_A"] + ["SONY_WF1000"],  # Leaked test item!
    }
    with pytest.raises(AssertionError, match="Data leakage detected"):
        assert_no_temporal_leakage(train_df, test_df, user_profiles_items=corrupted_profiles_items)


# ── 2. Section 11 Tests: Candidate Generation & Filter ────────────────
def test_candidate_generator_filters_already_interacted(mock_catalog, mock_reviews):
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(mock_catalog)
    builder = UserProfileBuilder(vectorizer=vectorizer)

    user_a_revs = mock_reviews[mock_reviews["user_id"] == "USER_A"]
    profile_a = builder.build_profile_from_interactions(user_a_revs)
    interacted_a = set(user_a_revs["parent_asin"])

    cand_gen = CandidateGenerator(vectorizer=vectorizer)
    candidates = cand_gen.generate_candidates(
        user_profile=profile_a,
        user_interacted_asins=interacted_a,
        candidate_count=10,
    )

    candidate_asins = [c[0] for c in candidates]

    # Invariant: none of the items USER_A already rated may appear in candidates
    for asin in interacted_a:
        assert asin not in candidate_asins, f"Already interacted item {asin} leaked into candidates!"


# ── 3. Section 12 Tests: Features & Labeling Rule ─────────────────────
def test_section_12_labeling_rule(mock_catalog, mock_reviews):
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(mock_catalog)
    builder = UserProfileBuilder(vectorizer=vectorizer)
    extractor = FeatureExtractor(mock_catalog, vectorizer, builder)

    X, y = extractor.build_training_dataset(mock_reviews)

    # Rating == 3 (BOSE_QC45 rated 3.0 by USER_B) must be ignored / excluded
    assert len(X) == len(mock_reviews) - 1  # 8 total - 1 ambiguous = 7 rows
    assert len(y) == 7

    # Positive examples (ratings >= 4): 5.0, 4.0, 5.0, 5.0, 5.0, 4.0 -> 6 rows
    assert (y == 1).sum() == 6
    # Negative examples (ratings <= 2): 2.0 -> 1 row
    assert (y == 0).sum() == 1

    # Feature columns check
    assert list(X.columns) == FEATURE_NAMES
    for col in FEATURE_NAMES:
        assert not X[col].isna().any()


# ── 3b. Leave-One-Out Training Anti-Leakage Test ──────────────────────
def test_training_dataset_has_no_target_leakage(mock_catalog):
    """Target item must never leak into its own UserSummary during training dataset creation."""
    reviews = pd.DataFrame(
        [
            {"user_id": "U_LEAK", "parent_asin": "SONY_WH1000", "rating": 5.0, "timestamp": 1000},
            {"user_id": "U_LEAK", "parent_asin": "BOSE_QC45", "rating": 5.0, "timestamp": 2000},
            {"user_id": "U_LEAK", "parent_asin": "ANKER_CABLE", "rating": 5.0, "timestamp": 3000},
            {"user_id": "U_LEAK", "parent_asin": "LOGI_MOUSE", "rating": 1.0, "timestamp": 4000},
        ]
    )
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(mock_catalog)
    builder = UserProfileBuilder(vectorizer=vectorizer, recency_decay=1.0)
    extractor = FeatureExtractor(mock_catalog, vectorizer, builder)

    X, y = extractor.build_training_dataset(reviews)
    assert len(X) == 4
    # Every item has a unique brand in mock_catalog (Sony, Bose, Anker, Logitech),
    # so excluding the target item means NO item has another positive item of the same brand.
    assert (X["brand_match"] == 0.0).all(), (
        f"Target leakage detected! brand_match should be 0.0 for all unique-brand rows, got {X['brand_match'].tolist()}"
    )


# ── 4. Section 11 Diversity Cap Test ──────────────────────────────────
def test_brand_diversity_cap(mock_catalog, mock_reviews):
    """Verify XGBoostRanker.rank_candidates enforces diversity_cap_per_brand."""
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(mock_catalog)
    builder = UserProfileBuilder(vectorizer=vectorizer)
    extractor = FeatureExtractor(mock_catalog, vectorizer, builder)

    # Fit a simple ranker on synthetic data
    np.random.seed(42)
    X_syn = pd.DataFrame(np.random.rand(40, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    y_syn = pd.Series([1, 0] * 20)
    ranker = XGBoostRanker(n_estimators=10, max_depth=2)
    ranker.fit(X_syn, y_syn)

    user_summary = extractor.build_user_summary(
        "USER_CAP",
        pd.DataFrame([{"user_id": "USER_CAP", "parent_asin": "SONY_WH1000", "rating": 5.0, "timestamp": 1000}]),
    )
    # Candidates include 3 Sony items + Bose + Anker
    candidates = ["SONY_WH1000", "SONY_WF1000", "SONY_SPEAKER", "BOSE_QC45", "ANKER_CABLE"]
    ranked = ranker.rank_candidates(
        user_summary=user_summary,
        candidate_asins=candidates,
        feature_extractor=extractor,
        top_k=4,
        diversity_cap_per_brand=2,
    )

    assert len(ranked) == 4
    sony_count = sum(1 for r in ranked if r["brand"] == "sony")
    assert sony_count <= 2, f"Expected at most 2 Sony items in top-4, got {sony_count}"


# ── 5. Section 13 & 14 Tests: SHAP & Explanation Engine ───────────────
def test_shap_and_explanation_engine():
    # Synthetic trained XGBoost model on FEATURE_NAMES
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(50, len(FEATURE_NAMES)), columns=FEATURE_NAMES)
    y = pd.Series(np.random.randint(0, 2, size=50))

    ranker = XGBoostRanker(n_estimators=10, max_depth=3)
    ranker.fit(X, y)

    shap_explainer = ShapExplainer(ranker)
    test_instance = {f: 0.8 for f in FEATURE_NAMES}
    shap_vals = shap_explainer.explain_instance(test_instance)

    assert set(shap_vals.keys()) == set(FEATURE_NAMES)
    for v in shap_vals.values():
        assert isinstance(v, float)

    # Explanation Engine test
    engine = ExplanationEngine()
    explanations = engine.explain(
        shap_values={"brand_match": 0.25, "text_similarity": 0.15},
        product_metadata={"brand": "Sony", "main_category": "Headphones"},
        top_n=2,
    )

    assert len(explanations) == 2
    assert explanations[0]["feature"] == "brand_match"
    assert "Sony" in explanations[0]["evidence"]
    assert explanations[1]["feature"] == "text_similarity"


# ── 6. Section 23 Tests: The recommend(user_id, k) Contract ───────────
def test_recommend_api_contract_active_and_cold_start(mock_catalog, mock_reviews):
    # End-to-end recommender initialization
    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(mock_catalog)

    pop_rec = PopularityRecommender(mock_catalog, mock_reviews)
    builder = UserProfileBuilder(vectorizer=vectorizer, popularity_recommender=pop_rec)
    feat_extractor = FeatureExtractor(mock_catalog, vectorizer, builder)
    cand_gen = CandidateGenerator(vectorizer=vectorizer)

    X, y = feat_extractor.build_training_dataset(mock_reviews)
    ranker = XGBoostRanker(n_estimators=10, max_depth=2)
    ranker.fit(X, y)

    shap_explainer = ShapExplainer(ranker)
    engine = ExplanationEngine()

    recommender = Phase1Recommender(
        metadata_df=mock_catalog,
        reviews_df=mock_reviews,
        vectorizer=vectorizer,
        user_profile_builder=builder,
        feature_extractor=feat_extractor,
        candidate_generator=cand_gen,
        ranker=ranker,
        shap_explainer=shap_explainer,
        explanation_engine=engine,
    )

    # 1. Active User Recommendation
    active_recs = recommender.recommend("USER_A", k=2)
    assert len(active_recs) == 2

    for rec in active_recs:
        # Section 23 required fields
        assert "product_id" in rec
        assert "title" in rec
        assert "brand" in rec
        assert "category" in rec
        assert "price" in rec
        assert "score" in rec
        assert "rank" in rec
        assert "explanation" in rec
        assert "shap_values" in rec
        assert "influenced_by" in rec
        assert isinstance(rec["explanation"], list)
        assert len(rec["explanation"]) > 0
        assert rec["influenced_by"] == []  # Phase 1: empty list

    # 2. Cold-Start User Recommendation
    cold_recs = recommender.recommend("UNKNOWN_USER_999", k=3)
    assert len(cold_recs) == 3

    for rec in cold_recs:
        assert "product_id" in rec
        assert "title" in rec
        assert "brand" in rec
        assert "category" in rec
        assert "price" in rec
        assert "score" in rec
        assert "rank" in rec
        assert "explanation" in rec
        assert "shap_values" in rec
        assert rec["influenced_by"] == []
