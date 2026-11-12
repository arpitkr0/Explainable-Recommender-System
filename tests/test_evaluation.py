"""
Tests for Section 20 Evaluation Plan.

Validates the 10 required properties:
  1. Precision@K calculation
  2. Recall@K calculation
  3. NDCG@K calculation
  4. Correct handling of a single held-out ground-truth item
  5. Recommendations already seen in training are not counted as correct (filtered by candidate generator)
  6. Evaluation uses training profiles only
  7. Test item does not leak into profile construction
  8. Phase 1 evaluation runs end-to-end
  9. Metric calculations are deterministic
  10. SHAP worked-example generation succeeds
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.content.text_features import ContentVectorizer
from src.content.user_profile import PopularityRecommender, UserProfileBuilder
from src.data.splitter import assert_no_temporal_leakage, temporal_train_test_split
from src.evaluation.evaluator import Phase1Evaluator
from src.evaluation.ranking_metrics import (
    aggregate_metrics,
    evaluate_user_recommendations,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.pipeline.recommender import Phase1Recommender
from src.ranking.features import FEATURE_NAMES, FeatureExtractor
from src.ranking.ranker import CandidateGenerator, XGBoostRanker


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def eval_catalog():
    return pd.DataFrame(
        [
            {
                "parent_asin": "ITEM_1",
                "title": "Wireless Bluetooth Over-Ear Headphones",
                "features": ["Noise cancelling", "Long battery life"],
                "description": ["High quality sound headphones"],
                "categories": ["Electronics", "Headphones"],
                "store": "AudioPro",
                "main_category": "Headphones",
                "price": 99.99,
                "average_rating": 4.8,
                "rating_number": 500,
            },
            {
                "parent_asin": "ITEM_2",
                "title": "USB-C Fast Charging Cable 6ft",
                "features": ["Durable nylon braid", "Fast sync"],
                "description": ["Heavy duty charging cable"],
                "categories": ["Electronics", "Accessories", "Cables"],
                "store": "CableTech",
                "main_category": "Accessories",
                "price": 12.99,
                "average_rating": 4.5,
                "rating_number": 1200,
            },
            {
                "parent_asin": "ITEM_3",
                "title": "Portable Bluetooth Waterproof Speaker",
                "features": ["Deep bass", "IPX7 waterproof"],
                "description": ["Outdoor portable sound"],
                "categories": ["Electronics", "Audio", "Speakers"],
                "store": "AudioPro",
                "main_category": "Audio",
                "price": 49.99,
                "average_rating": 4.6,
                "rating_number": 800,
            },
            {
                "parent_asin": "ITEM_4",
                "title": "Gaming Mechanical Keyboard RGB",
                "features": ["Clicky switches", "Custom RGB lighting"],
                "description": ["High performance mechanical keyboard"],
                "categories": ["Electronics", "Computers", "Keyboards"],
                "store": "GameMaster",
                "main_category": "Computers",
                "price": 79.99,
                "average_rating": 4.7,
                "rating_number": 950,
            },
            {
                "parent_asin": "ITEM_5",
                "title": "Optical Wireless Productivity Mouse",
                "features": ["Ergonomic grip", "Precision sensor"],
                "description": ["Quiet wireless mouse for office"],
                "categories": ["Electronics", "Computers", "Mice"],
                "store": "GameMaster",
                "main_category": "Computers",
                "price": 29.99,
                "average_rating": 4.4,
                "rating_number": 400,
            },
            {
                "parent_asin": "ITEM_6",
                "title": "Earbud Silicone Tips Replacement Pack",
                "features": ["Soft silicone", "Multiple sizes"],
                "description": ["Replacement ear tips"],
                "categories": ["Electronics", "Accessories"],
                "store": "AudioPro",
                "main_category": "Accessories",
                "price": 6.99,
                "average_rating": 4.2,
                "rating_number": 150,
            },
        ]
    )


@pytest.fixture
def eval_reviews():
    return pd.DataFrame(
        [
            # User 1: Audio fan
            {"user_id": "U1", "parent_asin": "ITEM_1", "rating": 5.0, "timestamp": 1000},
            {"user_id": "U1", "parent_asin": "ITEM_6", "rating": 2.0, "timestamp": 1500},  # Negative example
            {"user_id": "U1", "parent_asin": "ITEM_2", "rating": 4.0, "timestamp": 2000},
            {"user_id": "U1", "parent_asin": "ITEM_3", "rating": 5.0, "timestamp": 3000},  # Held-out test
            # User 2: PC fan
            {"user_id": "U2", "parent_asin": "ITEM_4", "rating": 5.0, "timestamp": 1100},
            {"user_id": "U2", "parent_asin": "ITEM_6", "rating": 1.0, "timestamp": 1600},  # Negative example
            {"user_id": "U2", "parent_asin": "ITEM_2", "rating": 4.0, "timestamp": 2100},
            {"user_id": "U2", "parent_asin": "ITEM_5", "rating": 4.0, "timestamp": 3100},  # Held-out test
        ]
    )


# ── 1. Precision@K Calculation ────────────────────────────────────────
def test_precision_at_k():
    recs = ["A", "B", "C", "D", "E"]
    gt = {"B", "D", "Z"}

    # Top 5: B and D hit -> 2 / 5 = 0.4
    assert precision_at_k(recs, gt, k=5) == 0.4
    # Top 2: only B hits -> 1 / 2 = 0.5
    assert precision_at_k(recs, gt, k=2) == 0.5
    # No hits in top 1 -> 0 / 1 = 0.0
    assert precision_at_k(recs, {"Z"}, k=1) == 0.0
    # Empty ground truth -> 0.0
    assert precision_at_k(recs, set(), k=5) == 0.0


# ── 2. Recall@K Calculation ───────────────────────────────────────────
def test_recall_at_k():
    recs = ["A", "B", "C", "D", "E"]
    gt = {"B", "D"}

    # Top 5 captures both B and D -> 2 / 2 = 1.0
    assert recall_at_k(recs, gt, k=5) == 1.0
    # Top 2 captures only B -> 1 / 2 = 0.5
    assert recall_at_k(recs, gt, k=2) == 0.5
    # Top 1 captures none -> 0.0
    assert recall_at_k(recs, gt, k=1) == 0.0


# ── 3. NDCG@K Calculation ────────────────────────────────────────────
def test_ndcg_at_k():
    gt = {"A"}
    # Relevant item at rank 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
    assert math.isclose(ndcg_at_k(["A", "B", "C"], gt, k=3), 1.0)

    # Relevant item at rank 2: DCG = 1/log2(3) = 0.6309297535714574
    expected_rank2 = 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(["B", "A", "C"], gt, k=3), expected_rank2)

    # Relevant item outside top-K: NDCG = 0.0
    assert ndcg_at_k(["B", "C", "D"], gt, k=3) == 0.0


# ── 4. Correct Handling of Single Held-Out Ground-Truth Item ─────────
def test_single_held_out_ground_truth_item():
    gt = {"TARGET_ITEM"}
    recs = ["OTHER_1", "TARGET_ITEM", "OTHER_2", "OTHER_3", "OTHER_4"]

    metrics = evaluate_user_recommendations(recs, gt, k_values=(5, 10))
    # Recall@5 is 1.0 (hit)
    assert metrics["recall@5"] == 1.0
    # Precision@5 is 1/5 = 0.2
    assert metrics["precision@5"] == 0.2
    # NDCG@5 is 1/log2(2 + 1) = 1/log2(3)
    assert math.isclose(metrics["ndcg@5"], 1.0 / math.log2(3))


# ── 5. Recommendations in Training are Not Counted as Correct ────────
def test_recommendations_already_seen_in_training_filtered(eval_catalog, eval_reviews):
    train_df, test_df = temporal_train_test_split(eval_reviews, holdout_k=1)

    vectorizer = ContentVectorizer(min_df=1, stop_words="english")
    vectorizer.fit_transform(eval_catalog)
    builder = UserProfileBuilder(vectorizer=vectorizer)

    u1_train = train_df[train_df["user_id"] == "U1"]
    profile_u1 = builder.build_profile_from_interactions(u1_train)
    interacted_u1 = set(u1_train["parent_asin"])

    cand_gen = CandidateGenerator(vectorizer=vectorizer)
    candidates = cand_gen.generate_candidates(
        user_profile=profile_u1,
        user_interacted_asins=interacted_u1,
        candidate_count=10,
    )
    candidate_ids = [c[0] for c in candidates]

    # Verify no training item (ITEM_1, ITEM_2) is eligible as a recommendation
    for item in interacted_u1:
        assert item not in candidate_ids


# ── 6. Evaluation Uses Training Profiles Only ─────────────────────────
def test_evaluation_uses_training_profiles_only(eval_catalog, eval_reviews):
    train_df, test_df = temporal_train_test_split(eval_reviews, holdout_k=1)

    evaluator = Phase1Evaluator(
        train_df=train_df,
        test_df=test_df,
        metadata_df=eval_catalog,
    )

    # Verify recommender reviews_df contains exclusively train_df
    assert len(evaluator.recommender.reviews_df) == len(train_df)
    # Check that test interactions are absent
    for _, test_row in test_df.iterrows():
        match = evaluator.recommender.reviews_df[
            (evaluator.recommender.reviews_df["user_id"] == test_row["user_id"])
            & (evaluator.recommender.reviews_df["parent_asin"] == test_row["parent_asin"])
        ]
        assert match.empty, "Test interaction leaked into recommender review history!"


# ── 7. Test Item Does Not Leak Into Profile Construction ──────────────
def test_test_item_does_not_leak_into_profile(eval_reviews):
    train_df, test_df = temporal_train_test_split(eval_reviews, holdout_k=1)

    # Invariant: max train timestamp <= min test timestamp
    assert_no_temporal_leakage(train_df, test_df)

    # Check that profile items dict strictly from train passes
    train_items = {
        "U1": train_df[train_df["user_id"] == "U1"]["parent_asin"].tolist(),
        "U2": train_df[train_df["user_id"] == "U2"]["parent_asin"].tolist(),
    }
    assert_no_temporal_leakage(train_df, test_df, user_profiles_items=train_items)

    # Deliberate corruption should raise AssertionError
    corrupted_items = {
        "U1": train_items["U1"] + ["ITEM_3"],  # ITEM_3 is U1's test item!
    }
    with pytest.raises(AssertionError, match="Data leakage detected"):
        assert_no_temporal_leakage(train_df, test_df, user_profiles_items=corrupted_items)


# ── 8. Phase 1 Evaluation Runs End-to-End ─────────────────────────────
def test_phase1_evaluation_runs_end_to_end(eval_catalog, eval_reviews):
    train_df, test_df = temporal_train_test_split(eval_reviews, holdout_k=1)

    evaluator = Phase1Evaluator(
        train_df=train_df,
        test_df=test_df,
        metadata_df=eval_catalog,
    )

    results = evaluator.run_evaluation(k_values=(5, 10))

    assert results["num_evaluated_users"] == 2
    assert results["num_excluded_users"] == 0
    assert "popularity_baseline_metrics" in results
    assert "global_shap_importance" in results

    overall = results["overall_metrics"]
    for k in (5, 10):
        assert f"precision@{k}" in overall
        assert f"recall@{k}" in overall
        assert f"ndcg@{k}" in overall
        assert 0.0 <= overall[f"precision@{k}"] <= 1.0
        assert 0.0 <= overall[f"recall@{k}"] <= 1.0
        assert 0.0 <= overall[f"ndcg@{k}"] <= 1.0


# ── 9. Metric Calculations Are Deterministic ──────────────────────────
def test_metric_calculations_are_deterministic():
    recs = ["ITEM_A", "ITEM_B", "ITEM_C", "ITEM_D", "ITEM_E"]
    gt = {"ITEM_C"}

    m1 = evaluate_user_recommendations(recs, gt, k_values=(5, 10))
    m2 = evaluate_user_recommendations(recs, gt, k_values=(5, 10))

    for k in (5, 10):
        assert m1[f"precision@{k}"] == m2[f"precision@{k}"]
        assert m1[f"recall@{k}"] == m2[f"recall@{k}"]
        assert m1[f"ndcg@{k}"] == m2[f"ndcg@{k}"]


# ── 10. SHAP Worked-Example Generation Succeeds ───────────────────────
def test_shap_worked_example_generation(eval_catalog, eval_reviews):
    train_df, test_df = temporal_train_test_split(eval_reviews, holdout_k=1)

    evaluator = Phase1Evaluator(
        train_df=train_df,
        test_df=test_df,
        metadata_df=eval_catalog,
    )

    results = evaluator.run_evaluation(k_values=(5, 10))
    examples = results.get("worked_shap_examples", [])

    assert len(examples) > 0
    ex = examples[0]
    assert "user_id" in ex
    assert "recommended_product" in ex
    assert "predicted_score" in ex
    assert "features" in ex
    assert set(ex["features"].keys()) == set(FEATURE_NAMES)
    assert "shap_values" in ex
    assert "final_explanation" in ex
    assert isinstance(ex["final_explanation"], list)
