"""
User Profile & Cold-Start Fallback — Section 10 of the specification.

Builds rating-weighted user profile vectors (P_u) from product TF-IDF vectors (V_i)
using a non-negative weighting scheme (5★: 1.0, 4★: 0.75, 1–3★: 0.0/excluded).
Provides a popularity-based fallback path for cold-start users with no interaction history.
"""

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.content.text_features import ContentVectorizer

log = logging.getLogger(__name__)

# ── Rating-to-Weight Mapping (Section 10) ────────────────────────────
# Deliberately kept non-negative to ensure TF-IDF cosine similarities
# remain stable and interpretable.
DEFAULT_RATING_WEIGHTS: dict[int, float] = {
    5: 1.0,
    4: 0.75,
    3: 0.0,   # Neutral — excluded from positive profile average
    2: 0.0,   # Negative — excluded from positive profile
    1: 0.0,   # Negative — excluded from positive profile
}


def get_rating_weight(
    rating: float | int, weights: dict[int, float] | None = None
) -> float:
    """Return non-negative weight for a rating value (defaulting to 0.0 for <= 3)."""
    if weights is None:
        weights = DEFAULT_RATING_WEIGHTS
    try:
        r_int = int(round(float(rating)))
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, weights.get(r_int, 0.0))


# ── Popularity Fallback Recommender ──────────────────────────────────
class PopularityRecommender:
    """Computes popularity-based fallback recommendations for cold-start users.

    Per Section 10:
    "highest-rated, most-reviewed products, spread across categories for some variety"
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        reviews_df: pd.DataFrame | None = None,
        category_col: str = "main_category",
    ):
        self.category_col = category_col
        self._ranked_items: list[dict[str, Any]] = []
        self._precompute_popularity(metadata_df, reviews_df)

    def _precompute_popularity(
        self, metadata_df: pd.DataFrame, reviews_df: pd.DataFrame | None = None
    ) -> None:
        """Score each catalog item by a combination of rating and review count."""
        df = metadata_df.copy()

        # Parse average_rating and rating_number
        df["avg_rating"] = pd.to_numeric(df.get("average_rating", 0.0), errors="coerce").fillna(0.0)
        df["num_ratings"] = pd.to_numeric(df.get("rating_number", 0), errors="coerce").fillna(0)

        # If reviews_df is provided, add actual observed review count from our dataset
        if reviews_df is not None and not reviews_df.empty:
            obs_counts = reviews_df.groupby("parent_asin").size()
            df["obs_reviews"] = df["parent_asin"].map(obs_counts).fillna(0)
            # Combine catalog rating_number with observed reviews
            total_review_signal = df["num_ratings"] + (df["obs_reviews"] * 10)
        else:
            total_review_signal = df["num_ratings"]

        # Popularity score: Bayesian-damped rating weighted by log volume
        # S_i = avg_rating * log1p(reviews)
        df["popularity_score"] = df["avg_rating"] * np.log1p(total_review_signal)

        # Standardize category
        if self.category_col not in df.columns or df[self.category_col].isna().all():
            df["cat_clean"] = "Electronics"
        else:
            df["cat_clean"] = df[self.category_col].fillna("Unknown").astype(str)

        # Sort descending by score
        sorted_df = df.sort_values(by="popularity_score", ascending=False)
        self._ranked_items = sorted_df[
            ["parent_asin", "title", "cat_clean", "avg_rating", "popularity_score"]
        ].to_dict(orient="records")

        log.info("Precomputed popularity ranking for %d catalog items", len(self._ranked_items))

    def recommend(
        self, top_k: int = 10, max_per_category: int = 2
    ) -> list[str]:
        """Return top_k product IDs spread across categories for variety."""
        if not self._ranked_items:
            return []

        selected: list[str] = []
        category_counts: dict[str, int] = {}
        deferred: list[str] = []

        # Pass 1: Select up to max_per_category to ensure category spread
        for item in self._ranked_items:
            asin = str(item["parent_asin"])
            cat = item.get("cat_clean", "Unknown")
            count = category_counts.get(cat, 0)
            if count < max_per_category:
                selected.append(asin)
                category_counts[cat] = count + 1
                if len(selected) >= top_k:
                    break
            else:
                deferred.append(asin)

        # Pass 2: If diversity constraint prevented reaching top_k, fill with next best
        if len(selected) < top_k:
            for asin in deferred:
                if asin not in selected:
                    selected.append(asin)
                    if len(selected) >= top_k:
                        break

        return selected[:top_k]


# ── User Profile Builder ─────────────────────────────────────────────
class UserProfileBuilder:
    """Builds rating-weighted user profile vectors and handles cold-start fallback."""

    def __init__(
        self,
        vectorizer: ContentVectorizer,
        rating_weights: dict[int, float] | None = None,
        popularity_recommender: PopularityRecommender | None = None,
        recency_decay: float = 1.0,
    ):
        self.vectorizer = vectorizer
        self.rating_weights = rating_weights or DEFAULT_RATING_WEIGHTS
        self.popularity_recommender = popularity_recommender
        self.recency_decay = float(recency_decay)

    def build_profile_from_interactions(
        self, interactions: pd.DataFrame
    ) -> np.ndarray | None:
        """Compute rating-weighted (and optionally recency-decayed) average of product vectors.

        interactions must contain 'parent_asin' and 'rating'.
        If recency_decay < 1.0 and 'timestamp' is present, more recent interactions
        receive exponentially higher weight: w = rating_weight * (recency_decay ** rank).
        Returns 1D L2-normalized numpy array or None if no valid interactions.
        """
        if interactions.empty or self.vectorizer.tfidf_matrix is None:
            return None

        ordered = interactions
        use_decay = self.recency_decay < 1.0
        if use_decay and "timestamp" in interactions.columns:
            ordered = interactions.sort_values(by="timestamp", ascending=False)

        profile = np.zeros(self.vectorizer.num_features, dtype=np.float64)
        total_weight = 0.0

        for rank_idx, (_, row) in enumerate(ordered.iterrows()):
            asin = str(row.get("parent_asin", ""))
            rating = row.get("rating", 0)
            rating_w = get_rating_weight(rating, self.rating_weights)

            # Exclude non-positive interactions (weights == 0.0)
            if rating_w <= 0.0:
                continue

            decay_factor = (self.recency_decay ** rank_idx) if use_decay else 1.0
            weight = rating_w * decay_factor
            if weight < 1e-8:
                break

            item_vec = self.vectorizer.get_item_vector(asin)
            if item_vec is None:
                continue

            # Add sparse vector directly into dense accumulator (avoids dense copies)
            if item_vec.nnz > 0:
                profile[item_vec.indices] += item_vec.data * weight
            total_weight += weight

        # If user has interactions, but all ratings are non-positive (total_weight == 0),
        # fallback to an unweighted average across their rated items so every user with
        # rating history gets a profile vector representing their domain of interest.
        if total_weight == 0.0:
            for _, row in ordered.iterrows():
                asin = str(row.get("parent_asin", ""))
                item_vec = self.vectorizer.get_item_vector(asin)
                if item_vec is not None:
                    if item_vec.nnz > 0:
                        profile[item_vec.indices] += item_vec.data
                    total_weight += 1.0

        if total_weight == 0.0:
            return None

        profile = profile / total_weight

        # Normalize to unit length (L2 norm) for clean cosine similarities
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        return profile

    def build_user_profile(
        self, user_id: str, reviews_df: pd.DataFrame
    ) -> np.ndarray | None:
        """Build profile vector for a user given their history in reviews_df."""
        user_str = str(user_id)
        user_reviews = reviews_df[reviews_df["user_id"].astype(str) == user_str]
        if user_reviews.empty:
            return None
        return self.build_profile_from_interactions(user_reviews)

    def get_profile_or_fallback(
        self,
        user_id: str,
        reviews_df: pd.DataFrame | None = None,
        top_k: int = 10,
    ) -> tuple[np.ndarray | None, bool, list[str]]:
        """Get profile vector or fall back to popularity-based path without erroring.

        Returns:
            (profile_vector, is_cold_start, fallback_recommendations)
            - If user has positive rating history: (profile_vector, False, [])
            - If user is cold-start (no history / no positive ratings): (None, True, [parent_asins])
        """
        user_str = str(user_id)

        profile = None
        if reviews_df is not None and not reviews_df.empty:
            profile = self.build_user_profile(user_str, reviews_df)

        if profile is not None:
            return profile, False, []

        # Cold-start fallback path
        fallback_items: list[str] = []
        if self.popularity_recommender is not None:
            fallback_items = self.popularity_recommender.recommend(top_k=top_k)
        else:
            log.warning("No popularity recommender configured for cold-start fallback.")

        return None, True, fallback_items

    def build_all_profiles(
        self, reviews_df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Build profile vectors for all users with valid positive history."""
        profiles = {}
        grouped = reviews_df.groupby(reviews_df["user_id"].astype(str))
        log.info("Building user profiles for %d unique users", len(grouped))

        for user_id, user_group in grouped:
            prof = self.build_profile_from_interactions(user_group)
            if prof is not None:
                profiles[str(user_id)] = prof

        log.info("Successfully built %d valid user profile vectors", len(profiles))
        return profiles
