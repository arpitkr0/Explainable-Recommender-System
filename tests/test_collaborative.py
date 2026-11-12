"""
Tests for Phase 2 — Collaborative Filtering (Section 16) & Hybrid Combination (Section 17).
"""

import numpy as np
import pandas as pd
import pytest

from src.collaborative.cf_model import CollaborativeFilter
from src.content.text_features import ContentVectorizer
from src.content.user_profile import PopularityRecommender, UserProfileBuilder
from src.explainability.explanation_generator import ExplanationEngine
from src.explainability.shap_explainer import ShapExplainer
from src.pipeline.recommender import HybridRecommender, Phase1Recommender
from src.ranking.features import FEATURE_NAMES, FeatureExtractor
from src.ranking.ranker import CandidateGenerator, XGBoostRanker


@pytest.fixture
def cf_catalog():
    return pd.DataFrame(
        [
            {"parent_asin": "ITEM_1", "title": "Sony Headphones", "features": ["ANC"], "description": ["Audio"], "categories": ["Headphones"], "store": "Sony", "main_category": "Headphones", "price": 299.0, "average_rating": 4.7, "rating_number": 500},
            {"parent_asin": "ITEM_2", "title": "Bose Headphones", "features": ["Comfort"], "description": ["Audio"], "categories": ["Headphones"], "store": "Bose", "main_category": "Headphones", "price": 279.0, "average_rating": 4.6, "rating_number": 450},
            {"parent_asin": "ITEM_3", "title": "Anker Cable", "features": ["USB-C"], "description": ["Cable"], "categories": ["Cables"], "store": "Anker", "main_category": "Accessories", "price": 15.0, "average_rating": 4.8, "rating_number": 1000},
            {"parent_asin": "ITEM_4", "title": "Logitech Mouse", "features": ["Wireless"], "description": ["Mouse"], "categories": ["Mice"], "store": "Logitech", "main_category": "Computers", "price": 79.0, "average_rating": 4.7, "rating_number": 800},
            {"parent_asin": "ITEM_5", "title": "Keychron Keyboard", "features": ["Mechanical"], "description": ["Keyboard"], "categories": ["Keyboards"], "store": "Keychron", "main_category": "Computers", "price": 99.0, "average_rating": 4.8, "rating_number": 600},
            {"parent_asin": "ITEM_6", "title": "Unseen New Speaker", "features": ["Bluetooth"], "description": ["Speaker"], "categories": ["Speakers"], "store": "JBL", "main_category": "Audio", "price": 89.0, "average_rating": 4.5, "rating_number": 200},
        ]
    )


@pytest.fixture
def cf_reviews():
    return pd.DataFrame(
        [
            {"user_id": "U1", "parent_asin": "ITEM_1", "rating": 5.0, "timestamp": 1000},
            {"user_id": "U1", "parent_asin": "ITEM_3", "rating": 4.0, "timestamp": 2000},
            {"user_id": "U1", "parent_asin": "ITEM_4", "rating": 2.0, "timestamp": 3000},
            {"user_id": "U2", "parent_asin": "ITEM_1", "rating": 5.0, "timestamp": 1100},
            {"user_id": "U2", "parent_asin": "ITEM_2", "rating": 5.0, "timestamp": 2100},
            {"user_id": "U2", "parent_asin": "ITEM_4", "rating": 1.0, "timestamp": 3100},
            {"user_id": "U3", "parent_asin": "ITEM_4", "rating": 5.0, "timestamp": 1200},
            {"user_id": "U3", "parent_asin": "ITEM_5", "rating": 5.0, "timestamp": 2200},
            {"user_id": "U3", "parent_asin": "ITEM_1", "rating": 2.0, "timestamp": 3200},
        ]
    )


def test_cf_fit_and_predict_bounds(cf_reviews):
    cf = CollaborativeFilter(n_factors=2, random_state=42).fit(cf_reviews)
    assert cf.is_fitted
    assert cf.has_user("U1")
    assert cf.has_item("ITEM_2")
    assert not cf.has_item("ITEM_6")  # Unseen in reviews

    rating = cf.predict_rating("U1", "ITEM_2")
    score = cf.predict_score("U1", "ITEM_2")
    assert rating is not None and 1.0 <= rating <= 5.0
    assert score is not None and 0.0 <= score <= 1.0

    # Cold-start / unseen returns None
    assert cf.predict_score("UNKNOWN_USER", "ITEM_1") is None
    assert cf.predict_score("U1", "ITEM_6") is None


def test_cf_recommend_candidates_excludes_interacted(cf_reviews):
    cf = CollaborativeFilter(n_factors=2, random_state=42).fit(cf_reviews)
    u1_seen = {"ITEM_1", "ITEM_3", "ITEM_4"}
    cands = cf.recommend_candidates("U1", user_interacted_asins=u1_seen, top_n=5)
    cand_ids = [a for a, _ in cands]
    assert len(cand_ids) > 0
    for seen in u1_seen:
        assert seen not in cand_ids


def test_cf_save_and_load(tmp_path, cf_reviews):
    cf = CollaborativeFilter(n_factors=2, random_state=42).fit(cf_reviews)
    path = tmp_path / "cf.pkl"
    cf.save(path)
    loaded = CollaborativeFilter.load(path)
    assert loaded.predict_score("U1", "ITEM_2") == cf.predict_score("U1", "ITEM_2")


def test_hybrid_recommender_contract_and_alpha_blending(cf_catalog, cf_reviews):
    vec = ContentVectorizer(min_df=1, stop_words="english")
    vec.fit_transform(cf_catalog)
    pop = PopularityRecommender(cf_catalog, cf_reviews)
    pb = UserProfileBuilder(vec, popularity_recommender=pop, recency_decay=0.85)
    fe = FeatureExtractor(cf_catalog, vec, pb)
    cg = CandidateGenerator(vec)
    X, y = fe.build_training_dataset(cf_reviews)
    ranker = XGBoostRanker(n_estimators=10, max_depth=2).fit(X, y)
    shap_exp = ShapExplainer(ranker)
    p1 = Phase1Recommender(
        metadata_df=cf_catalog,
        reviews_df=cf_reviews,
        vectorizer=vec,
        user_profile_builder=pb,
        feature_extractor=fe,
        candidate_generator=cg,
        ranker=ranker,
        shap_explainer=shap_exp,
        explanation_engine=ExplanationEngine(),
    )
    cf = CollaborativeFilter(n_factors=2, random_state=42).fit(cf_reviews)
    hybrid = HybridRecommender(phase1_recommender=p1, cf_model=cf, alpha=0.5)

    recs = hybrid.recommend("U1", k=3, alpha=0.5)
    assert len(recs) == 3
    for rec in recs:
        for key in (
            "product_id",
            "title",
            "brand",
            "category",
            "price",
            "score",
            "rank",
            "explanation",
            "shap_values",
            "influenced_by",
            "content_score",
            "cf_score",
        ):
            assert key in rec
        assert rec["influenced_by"] == []
        if rec["cf_score"] is not None:
            expected = 0.5 * rec["content_score"] + 0.5 * rec["cf_score"]
            assert abs(rec["score"] - expected) < 1e-3
        else:
            # Cold-start item unseen in CF falls back cleanly to content_score
            assert abs(rec["score"] - rec["content_score"]) < 1e-3

    # Cold-start user falls back gracefully
    cold_recs = hybrid.recommend("NEW_USER", k=2)
    assert len(cold_recs) == 2

