"""
Ranking Features & Training Dataset Construction — Section 12 of the specification.

Computes 6 interpretable user-item features:
  1. text_similarity: Cosine similarity between user profile and candidate product vector.
  2. category_similarity: Recency-weighted affinity between user's preferred categories and candidate.
  3. brand_match: 1 if candidate brand matches a brand the user rated >= positive_threshold, else 0.
  4. feature_similarity: Recency-weighted token affinity between candidate features and user history.
  5. price_similarity: Closeness of candidate price to user's typical historical price range.
  6. rating_similarity: Closeness of candidate average rating to user's typical ratings.

Constructs the training dataset using strictly Leave-One-Out (LOO) user summaries
so the target item never leaks into its own feature vector, following the Section 12
explicit labeling rule:
  - Positive (label=1): rating >= positive_rating_threshold (default 4)
  - Negative (label=0): rating <= negative_rating_threshold (default 2)
  - Ignored: ratings strictly between thresholds (e.g. 3-star, excluded from training)
"""

import logging
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.content.text_features import ContentVectorizer, clean_text
from src.content.user_profile import UserProfileBuilder, get_rating_weight

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    "text_similarity",
    "category_similarity",
    "brand_match",
    "feature_similarity",
    "price_similarity",
    "rating_similarity",
]

GENERIC_ROOT_CATEGORIES = {"electronics", "all electronics"}


def _tokenize_strings(values: Any) -> frozenset[str]:
    """Extract interned lowercase alphanumeric word tokens from string or list of strings."""
    if values is None:
        return frozenset()
    if isinstance(values, (list, tuple, np.ndarray)):
        text = " ".join(str(v) for v in values if v is not None)
    else:
        text = str(values)
    cleaned = clean_text(text)
    return frozenset(sys.intern(tok) for tok in cleaned.split()) if cleaned else frozenset()


def _filter_specific_categories(cat_set: set[str]) -> frozenset[str]:
    """Drop ultra-generic root categories when more specific subcategories exist."""
    specific = cat_set - GENERIC_ROOT_CATEGORIES
    chosen = specific if specific else cat_set
    return frozenset(sys.intern(c) for c in chosen)


class UserSummary:
    """Precomputed summary of a user's positive preferences for fast feature extraction."""

    def __init__(
        self,
        user_id: str,
        profile_vector: np.ndarray | None,
        preferred_categories: set[str],
        preferred_brands: set[str],
        historical_feature_tokens: set[str],
        mean_price: float,
        mean_rating: float,
        interacted_asins: set[str],
        category_weights: Counter | None = None,
        total_category_weight: float = 0.0,
        feature_weights: Counter | None = None,
        total_feature_weight: float = 0.0,
    ):
        self.user_id = user_id
        self.profile_vector = profile_vector
        self.preferred_categories = preferred_categories
        self.preferred_brands = preferred_brands
        self.historical_feature_tokens = historical_feature_tokens
        self.mean_price = mean_price
        self.mean_rating = mean_rating
        self.interacted_asins = interacted_asins
        self.category_weights = category_weights or Counter()
        self.total_category_weight = float(total_category_weight)
        self.feature_weights = feature_weights or Counter()
        self.total_feature_weight = float(total_feature_weight)


class FeatureExtractor:
    """Extracts the 6 interpretable Section 12 features for user-candidate pairs."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        vectorizer: ContentVectorizer,
        user_profile_builder: UserProfileBuilder,
        positive_threshold: float = 4.0,
        negative_threshold: float = 2.0,
    ):
        self.vectorizer = vectorizer
        self.user_profile_builder = user_profile_builder
        self.positive_threshold = float(positive_threshold)
        self.negative_threshold = float(negative_threshold)
        self.metadata_lookup = self._build_metadata_lookup(metadata_df)
        self.catalog_median_price = self._compute_catalog_median_price(metadata_df)

    def _build_metadata_lookup(self, metadata_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Index metadata by parent_asin for fast O(1) attribute lookup."""
        lookup: dict[str, dict[str, Any]] = {}
        for _, row in metadata_df.iterrows():
            asin = str(row["parent_asin"])

            cats = row.get("categories", [])
            cat_set = set()
            if isinstance(cats, (list, tuple)):
                for c in cats:
                    if str(c).strip():
                        cat_set.add(str(c).strip().lower())
            main_cat = row.get("main_category")
            if main_cat and str(main_cat).strip().lower() != "nan":
                cat_set.add(str(main_cat).strip().lower())
            cat_set = _filter_specific_categories(cat_set)

            brand = row.get("store") or row.get("brand") or ""
            brand_str = sys.intern(str(brand).strip().lower()) if brand else ""

            feat_tokens = _tokenize_strings(row.get("features", []))

            try:
                price = float(row.get("price", np.nan))
                if np.isnan(price) or price <= 0:
                    price = np.nan
            except (ValueError, TypeError):
                price = np.nan

            try:
                avg_rating = float(row.get("average_rating", 4.0))
                if np.isnan(avg_rating):
                    avg_rating = 4.0
            except (ValueError, TypeError):
                avg_rating = 4.0

            lookup[asin] = {
                "parent_asin": asin,
                "title": str(row.get("title", "")),
                "categories": cat_set,
                "brand": brand_str,
                "feature_tokens": feat_tokens,
                "price": price,
                "average_rating": avg_rating,
            }
        log.info("Built fast metadata lookup for %d catalog products", len(lookup))
        return lookup

    def _compute_catalog_median_price(self, metadata_df: pd.DataFrame) -> float:
        prices = pd.to_numeric(metadata_df.get("price"), errors="coerce").dropna()
        if len(prices) > 0:
            return float(prices.median())
        return 29.99

    def build_user_summary(
        self, user_id: str, user_reviews: pd.DataFrame
    ) -> UserSummary:
        """Summarize user's history with recency-weighted affinities for fast feature scoring."""
        user_str = str(user_id)
        interacted = set(user_reviews["parent_asin"].astype(str).tolist())

        profile_vec = self.user_profile_builder.build_profile_from_interactions(user_reviews)

        decay = getattr(self.user_profile_builder, "recency_decay", 1.0)
        use_decay = decay < 1.0
        ordered = (
            user_reviews.sort_values(by="timestamp", ascending=False)
            if (use_decay and "timestamp" in user_reviews.columns)
            else user_reviews
        ).reset_index(drop=True)

        ordered_pos = ordered[ordered["rating"] >= self.positive_threshold]
        ref_revs = ordered_pos if not ordered_pos.empty else ordered

        pref_categories: set[str] = set()
        pref_brands: set[str] = set()
        hist_feature_tokens: set[str] = set()
        cat_weights: Counter = Counter()
        feat_weights: Counter = Counter()
        tot_cat_w = 0.0
        tot_feat_w = 0.0
        price_wsum = 0.0
        price_wtot = 0.0

        for rank_idx, row in ref_revs.iterrows():
            asin = str(row["parent_asin"])
            meta = self.metadata_lookup.get(asin)
            if not meta:
                continue

            rating_w = get_rating_weight(row["rating"], self.user_profile_builder.rating_weights)
            base_w = rating_w if rating_w > 0.0 else 1.0
            w = base_w * ((decay ** int(rank_idx)) if use_decay else 1.0)
            if w < 1e-8:
                continue

            pref_categories.update(meta["categories"])
            for c in meta["categories"]:
                cat_weights[c] += w
                tot_cat_w += w

            if meta["brand"]:
                pref_brands.add(meta["brand"])

            hist_feature_tokens.update(meta["feature_tokens"])
            for ft in meta["feature_tokens"]:
                feat_weights[ft] += w
                tot_feat_w += w

            if not np.isnan(meta["price"]):
                price_wsum += meta["price"] * w
                price_wtot += w

        mean_price = (price_wsum / price_wtot) if price_wtot > 0 else self.catalog_median_price

        ratings = pd.to_numeric(user_reviews["rating"], errors="coerce").dropna()
        mean_rating = float(ratings.mean()) if not ratings.empty else 4.0

        return UserSummary(
            user_id=user_str,
            profile_vector=profile_vec,
            preferred_categories=pref_categories,
            preferred_brands=pref_brands,
            historical_feature_tokens=hist_feature_tokens,
            mean_price=mean_price,
            mean_rating=mean_rating,
            interacted_asins=interacted,
            category_weights=cat_weights,
            total_category_weight=tot_cat_w,
            feature_weights=feat_weights,
            total_feature_weight=tot_feat_w,
        )

    def extract_features(
        self, user_summary: UserSummary, candidate_asin: str
    ) -> dict[str, float]:
        """Compute the 6 Section 12 features for a single (user, candidate) pair."""
        cand_meta = self.metadata_lookup.get(str(candidate_asin))
        if not cand_meta:
            return {f: 0.0 for f in FEATURE_NAMES}

        # 1. text_similarity (cosine similarity via sparse vector dot)
        if user_summary.profile_vector is not None:
            item_vec = self.vectorizer.get_item_vector(candidate_asin)
            if item_vec is not None and item_vec.nnz > 0:
                text_sim = float(np.dot(item_vec.data, user_summary.profile_vector[item_vec.indices]))
                text_sim = max(0.0, min(1.0, text_sim))
            else:
                text_sim = 0.0
        else:
            text_sim = 0.0

        # 2. category_similarity (recency/frequency-weighted affinity)
        cand_cats = cand_meta["categories"]
        if cand_cats and user_summary.total_category_weight > 0:
            matched_w = sum(user_summary.category_weights.get(c, 0.0) for c in cand_cats)
            cat_sim = min(1.0, (matched_w / user_summary.total_category_weight) * 3.0)
        elif user_summary.preferred_categories and cand_cats:
            inter = len(user_summary.preferred_categories.intersection(cand_cats))
            cat_sim = inter / len(cand_cats)
        else:
            cat_sim = 0.0

        # 3. brand_match (1 if candidate brand matches user's preferred brands)
        cand_brand = cand_meta["brand"]
        brand_match = 1.0 if (cand_brand and cand_brand in user_summary.preferred_brands) else 0.0

        # 4. feature_similarity (recency/frequency-weighted token affinity)
        cand_tokens = cand_meta["feature_tokens"]
        if cand_tokens and user_summary.total_feature_weight > 0:
            matched_fw = sum(user_summary.feature_weights.get(ft, 0.0) for ft in cand_tokens)
            feat_sim = min(1.0, (matched_fw / user_summary.total_feature_weight) * 3.0)
        elif user_summary.historical_feature_tokens and cand_tokens:
            inter = len(user_summary.historical_feature_tokens.intersection(cand_tokens))
            feat_sim = inter / len(cand_tokens)
        else:
            feat_sim = 0.0

        # 5. price_similarity (log-price proximity)
        cand_price = cand_meta["price"]
        if np.isnan(cand_price):
            price_sim = 0.5
        else:
            log_diff = abs(np.log1p(cand_price) - np.log1p(user_summary.mean_price))
            price_sim = float(1.0 / (1.0 + log_diff))

        # 6. rating_similarity (closeness of average rating to user's standard)
        cand_rating = cand_meta["average_rating"]
        rating_diff = abs(cand_rating - user_summary.mean_rating)
        rating_sim = float(max(0.0, min(1.0, 1.0 - (rating_diff / 4.0))))

        return {
            "text_similarity": float(text_sim),
            "category_similarity": float(cat_sim),
            "brand_match": float(brand_match),
            "feature_similarity": float(feat_sim),
            "price_similarity": float(price_sim),
            "rating_similarity": float(rating_sim),
        }

    def build_training_dataset(
        self,
        reviews_df: pd.DataFrame,
        max_samples: int | None = None,
        max_pos_per_user: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Construct Leave-One-Out (X, y) training dataset following Section 12 labeling rule:

        Positive examples (1) = rating >= positive_threshold (default 4)
        Negative examples (0) = rating <= negative_threshold (default 2)
        Ignored               = ratings strictly between negative and positive thresholds (e.g. 3)

        Each training row (u, i) uses a Leave-One-Out user summary computed strictly from
        user u's OTHER interactions (excluding i), preventing target leakage.
        """
        is_labeled = (reviews_df["rating"] >= self.positive_threshold) | (
            reviews_df["rating"] <= self.negative_threshold
        )
        labeled_df = reviews_df[is_labeled].copy()

        if max_samples and len(labeled_df) > max_samples:
            sampled_indices = set(labeled_df.sample(n=max_samples, random_state=42).index)
        else:
            sampled_indices = None

        decay = getattr(self.user_profile_builder, "recency_decay", 1.0)
        use_decay = decay < 1.0
        rating_weights = self.user_profile_builder.rating_weights
        n_features = self.vectorizer.num_features

        feature_rows: list[list[float]] = []
        labels: list[int] = []

        log.info(
            "Constructing Leave-One-Out ranking training dataset from %d labeled interactions",
            len(labeled_df),
        )

        for _, user_all_revs in reviews_df.groupby("user_id"):
            ordered = (
                user_all_revs.sort_values(by="timestamp", ascending=False)
                if (use_decay and "timestamp" in user_all_revs.columns)
                else user_all_revs
            )

            # Accumulate full-user state once, then subtract per item for O(1) LOO
            full_profile = np.zeros(n_features, dtype=np.float64)
            total_prof_w = 0.0
            brand_w: Counter = Counter()
            cat_w: Counter = Counter()
            feat_w: Counter = Counter()
            tot_cat_w = 0.0
            tot_feat_w = 0.0
            price_wsum = 0.0
            price_wtot = 0.0
            rating_sum = 0.0
            rating_cnt = 0

            # Per-row cached weights aligned to `ordered`
            row_records: list[tuple[Any, str, float, float, float]] = []

            for rank_idx, (idx_label, row) in enumerate(ordered.iterrows()):
                asin = str(row["parent_asin"])
                rating = float(row["rating"])
                rating_sum += rating
                rating_cnt += 1

                is_pos = rating >= self.positive_threshold
                is_neg = rating <= self.negative_threshold

                rw = get_rating_weight(rating, rating_weights)
                prof_w = 0.0
                if rw > 0.0:
                    prof_w = rw * ((decay ** rank_idx) if use_decay else 1.0)
                    iv = self.vectorizer.get_item_vector(asin)
                    if iv is not None and iv.nnz > 0 and prof_w > 1e-8:
                        full_profile[iv.indices] += iv.data * prof_w
                    total_prof_w += prof_w

                attr_w = 0.0
                if is_pos:
                    attr_w = (rw if rw > 0.0 else 1.0) * ((decay ** rank_idx) if use_decay else 1.0)
                    meta = self.metadata_lookup.get(asin)
                    if meta and attr_w > 1e-8:
                        if meta["brand"]:
                            brand_w[meta["brand"]] += attr_w
                        for c in meta["categories"]:
                            cat_w[c] += attr_w
                            tot_cat_w += attr_w
                        for ft in meta["feature_tokens"]:
                            feat_w[ft] += attr_w
                            tot_feat_w += attr_w
                        if not np.isnan(meta["price"]):
                            price_wsum += meta["price"] * attr_w
                            price_wtot += attr_w

                if is_pos or is_neg:
                    if sampled_indices is None or idx_label in sampled_indices:
                        row_records.append((idx_label, asin, rating, prof_w, attr_w))

            if max_pos_per_user is not None:
                neg_recs = [r for r in row_records if r[2] <= self.negative_threshold]
                pos_recs = [r for r in row_records if r[2] >= self.positive_threshold][:max_pos_per_user]
                row_records = neg_recs + pos_recs

            # Precompute full squared norm for fast LOO norm calculation
            full_sq_norm = float(np.dot(full_profile, full_profile))

            for _, asin, rating, prof_w, attr_w in row_records:
                meta = self.metadata_lookup.get(asin)
                if not meta:
                    feature_rows.append([0.0] * len(FEATURE_NAMES))
                    labels.append(1 if rating >= self.positive_threshold else 0)
                    continue

                # 1. LOO text_similarity using algebraic sparse subtraction:
                # dot(full - w*v, v) = dot(full, v) - w * ||v||^2
                # ||full - w*v||^2 = ||full||^2 - 2*w*dot(full, v) + w^2 * ||v||^2
                iv = self.vectorizer.get_item_vector(asin)
                if iv is not None and iv.nnz > 0 and total_prof_w > prof_w:
                    v_data = iv.data
                    v_idx = iv.indices
                    dot_full_v = float(np.dot(full_profile[v_idx], v_data))
                    v_sq = float(np.dot(v_data, v_data))
                    if prof_w > 0.0:
                        loo_dot = dot_full_v - prof_w * v_sq
                        loo_sq = max(0.0, full_sq_norm - 2.0 * prof_w * dot_full_v + (prof_w ** 2) * v_sq)
                    else:
                        loo_dot = dot_full_v
                        loo_sq = full_sq_norm

                    if loo_sq > 1e-12:
                        text_sim = max(0.0, min(1.0, loo_dot / float(np.sqrt(loo_sq))))
                    else:
                        text_sim = 0.0
                else:
                    text_sim = 0.0

                is_pos = rating >= self.positive_threshold
                sub_w = attr_w if is_pos else 0.0

                # 2. LOO category_similarity
                cand_cats = meta["categories"]
                rem_cw = tot_cat_w - (sub_w * len(cand_cats))
                if cand_cats and rem_cw > 1e-8:
                    matched_cw = sum(max(0.0, cat_w.get(c, 0.0) - sub_w) for c in cand_cats)
                    cat_sim = min(1.0, (matched_cw / rem_cw) * 3.0)
                else:
                    cat_sim = 0.0

                # 3. LOO brand_match
                cand_brand = meta["brand"]
                brand_match = (
                    1.0
                    if (cand_brand and (brand_w.get(cand_brand, 0.0) - sub_w) > 1e-8)
                    else 0.0
                )

                # 4. LOO feature_similarity
                cand_tokens = meta["feature_tokens"]
                rem_fw = tot_feat_w - (sub_w * len(cand_tokens))
                if cand_tokens and rem_fw > 1e-8:
                    matched_fw = sum(max(0.0, feat_w.get(ft, 0.0) - sub_w) for ft in cand_tokens)
                    feat_sim = min(1.0, (matched_fw / rem_fw) * 3.0)
                else:
                    feat_sim = 0.0

                # 5. LOO price_similarity
                cand_price = meta["price"]
                if np.isnan(cand_price):
                    price_sim = 0.5
                else:
                    rem_pw = price_wtot - sub_w
                    rem_ps = price_wsum - (cand_price * sub_w)
                    mean_p = (rem_ps / rem_pw) if rem_pw > 1e-8 else self.catalog_median_price
                    price_sim = float(1.0 / (1.0 + abs(np.log1p(cand_price) - np.log1p(mean_p))))

                # 6. LOO rating_similarity
                mean_r = ((rating_sum - rating) / (rating_cnt - 1)) if rating_cnt > 1 else 4.0
                rating_sim = float(max(0.0, min(1.0, 1.0 - abs(meta["average_rating"] - mean_r) / 4.0)))

                feature_rows.append([text_sim, cat_sim, brand_match, feat_sim, price_sim, rating_sim])
                labels.append(1 if is_pos else 0)

        X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        y = pd.Series(labels, name="label")

        pos_count = int((y == 1).sum())
        neg_count = int((y == 0).sum())
        log.info(
            "LOO training dataset built: %d rows (Positive=%d [%.2f%%], Negative=%d [%.2f%%])",
            len(X),
            pos_count,
            pos_count / len(X) * 100 if len(X) > 0 else 0,
            neg_count,
            neg_count / len(X) * 100 if len(X) > 0 else 0,
        )
        return X, y
