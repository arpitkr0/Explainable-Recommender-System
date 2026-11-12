"""
Candidate Generation & XGBoost Ranking Model — Section 11 & Section 12 of the specification.

1. CandidateGenerator (Section 11):
   - Computes cosine similarity between user profile and all product vectors.
   - Filters out products the user has already interacted with.
   - Selects top candidate_count candidates (default 100).

2. XGBoostRanker (Section 12):
   - Trains an XGBoost classifier on the 6 interpretable features.
   - Predicts P(user likes item).
   - Applies the brand diversity cap (max N items from same brand in final top-K).
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix

from src.content.text_features import ContentVectorizer
from src.ranking.features import FEATURE_NAMES, FeatureExtractor, UserSummary

log = logging.getLogger(__name__)


class CandidateGenerator:
    """Generates candidate items via vector cosine similarity (Section 11)."""

    def __init__(self, vectorizer: ContentVectorizer):
        self.vectorizer = vectorizer

    def generate_candidates(
        self,
        user_profile: np.ndarray,
        user_interacted_asins: set[str],
        candidate_count: int = 100,
    ) -> list[tuple[str, float]]:
        """Generate top candidate_count items with already-interacted items removed.

        Args:
            user_profile: 1D L2-normalized profile vector of shape (D,).
            user_interacted_asins: Set of parent_asin strings to exclude.
            candidate_count: Number of candidates to return (default 100).

        Returns:
            List of (parent_asin, cosine_similarity) sorted descending.
        """
        if self.vectorizer.tfidf_matrix is None or user_profile is None:
            return []

        # Cosine similarity is a dot product since both profile and product vectors are unit norm
        # tfidf_matrix is (N, D), user_profile is (D,) -> scores is (N,)
        scores = self.vectorizer.tfidf_matrix.dot(user_profile)
        if isinstance(scores, np.matrix):
            scores = np.asarray(scores).flatten()
        elif hasattr(scores, "toarray"):
            scores = scores.toarray().flatten()

        # Copy to avoid mutating internal structures
        scores = np.copy(scores)

        # Remove products the user has already interacted with (Section 11 explicit filter)
        for asin in user_interacted_asins:
            idx = self.vectorizer.get_index(asin)
            if idx is not None and idx < len(scores):
                scores[idx] = -1.0

        # Select top candidates
        k = min(candidate_count, len(scores))
        if k <= 0:
            return []

        # argpartition for fast top-k selection
        top_indices = np.argpartition(scores, -k)[-k:]
        # Sort top indices descending by score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        candidates: list[tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= -1.0:
                continue  # Filtered out
            asin = self.vectorizer.get_parent_asin(int(idx))
            if asin:
                candidates.append((asin, score))

        return candidates


class XGBoostRanker:
    """XGBoost ranking model predicting P(user likes item) with diversity cap (Section 12)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        monotone_constraints: tuple[int, ...] | None = (1, 1, 1, 1, 1, 1),
        random_state: int = 42,
    ):
        xgb_kwargs: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "eval_metric": "logloss",
        }
        if monotone_constraints is not None and len(monotone_constraints) == len(FEATURE_NAMES):
            xgb_kwargs["monotone_constraints"] = monotone_constraints

        self.model = xgb.XGBClassifier(**xgb_kwargs)
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostRanker":
        """Train the ranking classifier on feature matrix X and labels y."""
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            log.warning(
                "Training dataset contains only one class (%s). XGBoost requires both classes. "
                "Ranker will default to text_similarity scoring.",
                unique_classes,
            )
            self.is_fitted = False
            return self

        log.info("Fitting XGBoost ranking model on %d samples", len(X))
        self.model.fit(X[FEATURE_NAMES], y)
        self.is_fitted = True
        log.info("XGBoost ranking model training complete")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probability of positive class P(user likes item)."""
        if not self.is_fitted:
            raise RuntimeError("XGBoostRanker is not fitted yet.")
        # Column 1 is probability of positive class
        return self.model.predict_proba(X[FEATURE_NAMES])[:, 1]

    def rank_candidates(
        self,
        user_summary: UserSummary,
        candidate_asins: list[str],
        feature_extractor: FeatureExtractor,
        top_k: int = 10,
        diversity_cap_per_brand: int = 2,
    ) -> list[dict[str, Any]]:
        """Extract features, score candidates with XGBoost, and apply diversity cap.

        Args:
            user_summary: Precomputed user profile and preference summary.
            candidate_asins: List of candidate product IDs from CandidateGenerator.
            feature_extractor: FeatureExtractor instance.
            top_k: Number of final recommendations to return (default 10).
            diversity_cap_per_brand: Maximum items from same brand in final top-K (Section 11).

        Returns:
            List of ranked candidate dicts with score, rank, and features.
        """
        if not candidate_asins:
            return []

        # Extract features for each candidate
        feature_list = []
        valid_asins = []
        for asin in candidate_asins:
            feats = feature_extractor.extract_features(user_summary, asin)
            feature_list.append(feats)
            valid_asins.append(asin)

        feature_df = pd.DataFrame(feature_list)[FEATURE_NAMES]

        # Score with XGBoost model if fitted, else fall back to text_similarity
        if self.is_fitted:
            scores = self.predict_proba(feature_df)
        else:
            scores = feature_df["text_similarity"].values

        # Combine candidates with scores and features
        scored_candidates = []
        for asin, score, (_, feat_row) in zip(valid_asins, scores, feature_df.iterrows()):
            cand_meta = feature_extractor.metadata_lookup.get(asin, {})
            brand = cand_meta.get("brand", "")
            scored_candidates.append(
                {
                    "parent_asin": asin,
                    "score": float(score),
                    "brand": brand,
                    "features": feat_row.to_dict(),
                }
            )

        # Sort descending by ranking score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Apply Section 11 brand diversity cap: max N items from same brand in final top-k
        final_recommendations: list[dict[str, Any]] = []
        brand_counts: dict[str, int] = {}
        deferred: list[dict[str, Any]] = []

        for cand in scored_candidates:
            brand = cand["brand"]
            count = brand_counts.get(brand, 0)
            if not brand or count < diversity_cap_per_brand:
                cand["rank"] = len(final_recommendations) + 1
                final_recommendations.append(cand)
                if brand:
                    brand_counts[brand] = count + 1
                if len(final_recommendations) >= top_k:
                    break
            else:
                deferred.append(cand)

        # If diversity cap leaves list short of top_k, fill with remaining highest scoring
        if len(final_recommendations) < top_k:
            for cand in deferred:
                cand["rank"] = len(final_recommendations) + 1
                final_recommendations.append(cand)
                if len(final_recommendations) >= top_k:
                    break

        return final_recommendations

    def save(self, file_path: str | Path) -> None:
        """Serialize fitted model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "is_fitted": self.is_fitted}, f)
        log.info("XGBoostRanker saved to %s", path)

    @classmethod
    def load(cls, file_path: str | Path) -> "XGBoostRanker":
        """Load serialized model from disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.is_fitted = data["is_fitted"]
        log.info("XGBoostRanker loaded from %s", path)
        return obj
