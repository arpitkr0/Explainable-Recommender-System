"""
The Recommendation API — Section 23 of the specification.

Provides the core recommend(user_id, k) pipeline for Phase 1:
  1. Profile retrieval / building (Section 10)
  2. Candidate generation & interaction filtering (Section 11)
  3. Feature extraction & XGBoost ranking (Section 12)
  4. Brand diversity cap (Section 11)
  5. SHAP explainability (Section 13)
  6. Natural-language explanation generation (Section 14)
  7. Cold-start popularity fallback (Section 10)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.collaborative.cf_model import CollaborativeFilter
from src.content.text_features import ContentVectorizer
from src.content.user_profile import PopularityRecommender, UserProfileBuilder
from src.data.loader import load_config, load_data
from src.explainability.explanation_generator import ExplanationEngine
from src.explainability.shap_explainer import ShapExplainer
from src.ranking.features import FEATURE_NAMES, FeatureExtractor, UserSummary
from src.ranking.ranker import CandidateGenerator, XGBoostRanker

log = logging.getLogger(__name__)


class Phase1Recommender:
    """End-to-end Phase 1 Explainable Content-Based Recommender."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        vectorizer: ContentVectorizer,
        user_profile_builder: UserProfileBuilder,
        feature_extractor: FeatureExtractor,
        candidate_generator: CandidateGenerator,
        ranker: XGBoostRanker,
        shap_explainer: ShapExplainer | None = None,
        explanation_engine: ExplanationEngine | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.metadata_df = metadata_df
        self.reviews_df = reviews_df
        self.vectorizer = vectorizer
        self.user_profile_builder = user_profile_builder
        self.feature_extractor = feature_extractor
        self.candidate_generator = candidate_generator
        self.ranker = ranker
        self.shap_explainer = shap_explainer
        self.explanation_engine = explanation_engine or ExplanationEngine()
        self.config = config or {}

        # Config parameters
        retrieval_cfg = self.config.get("retrieval", {})
        self.candidate_count = retrieval_cfg.get("candidate_count", 100)
        self.default_top_k = retrieval_cfg.get("top_k", 10)
        self.diversity_cap_per_brand = retrieval_cfg.get("diversity_cap_per_brand", 2)

    def recommend(self, user_id: str, k: int | None = None) -> list[dict[str, Any]]:
        """Return top-k explainable recommendations conforming to Section 23 schema."""
        top_k = k if k is not None else self.default_top_k
        user_str = str(user_id)

        # 1. Look up user history
        user_reviews = self.reviews_df[self.reviews_df["user_id"].astype(str) == user_str]

        # Check for cold start (no history)
        if user_reviews.empty:
            return self._recommend_cold_start(user_str, top_k)

        # 2. Build user summary (profile vector, preferred brands/cats/price)
        user_summary = self.feature_extractor.build_user_summary(user_str, user_reviews)

        if user_summary.profile_vector is None:
            return self._recommend_cold_start(user_str, top_k)

        # 3. Candidate generation (Section 11): Cosine sim + filter already-interacted
        candidates = self.candidate_generator.generate_candidates(
            user_profile=user_summary.profile_vector,
            user_interacted_asins=user_summary.interacted_asins,
            candidate_count=self.candidate_count,
        )

        if not candidates:
            return self._recommend_cold_start(user_str, top_k)

        candidate_asins = [asin for asin, _ in candidates]

        # 4. Rank candidates with XGBoost and apply diversity cap (Sections 11 & 12)
        ranked_candidates = self.ranker.rank_candidates(
            user_summary=user_summary,
            candidate_asins=candidate_asins,
            feature_extractor=self.feature_extractor,
            top_k=top_k,
            diversity_cap_per_brand=self.diversity_cap_per_brand,
        )

        # 5. Populate Section 23 recommendation contract with SHAP and explanations
        results: list[dict[str, Any]] = []
        for cand in ranked_candidates:
            asin = cand["parent_asin"]
            meta = self.feature_extractor.metadata_lookup.get(asin, {})
            score = cand["score"]
            rank = cand["rank"]
            features = cand["features"]

            # Compute SHAP values if explainer is available
            shap_dict: dict[str, float] = {}
            if self.shap_explainer is not None:
                try:
                    shap_dict = self.shap_explainer.explain_instance(features)
                except Exception as e:
                    log.warning("SHAP explanation failed for %s: %s", asin, e)
                    shap_dict = {f: 0.0 for f in FEATURE_NAMES}

            # Generate natural language explanation sentences
            explanations = self.explanation_engine.get_plain_sentences(
                shap_values=shap_dict,
                product_metadata=meta,
                top_n=4,
            )

            results.append(
                {
                    "product_id": asin,
                    "title": meta.get("title", ""),
                    "brand": meta.get("brand", ""),
                    "category": self._extract_primary_category(meta),
                    "price": None if np.isnan(meta.get("price", np.nan)) else meta.get("price"),
                    "score": round(float(score), 4),
                    "rank": int(rank),
                    "explanation": explanations,
                    "shap_values": shap_dict,
                    "influenced_by": [],  # Populated in Phase 4
                    "features": {k: round(float(v), 4) for k, v in features.items()},
                }
            )

        return results

    def _recommend_cold_start(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        """Popularity-based fallback path for cold-start users (Section 10 & 23)."""
        log.info("Generating cold-start popularity recommendations for user %s", user_id)
        pop_recs = self.user_profile_builder.popularity_recommender
        if pop_recs is not None:
            asins = pop_recs.recommend(top_k=top_k, max_per_category=2)
        else:
            asins = []

        results: list[dict[str, Any]] = []
        for rank, asin in enumerate(asins, 1):
            meta = self.feature_extractor.metadata_lookup.get(asin, {})
            results.append(
                {
                    "product_id": asin,
                    "title": meta.get("title", ""),
                    "brand": meta.get("brand", ""),
                    "category": self._extract_primary_category(meta),
                    "price": None if np.isnan(meta.get("price", np.nan)) else meta.get("price"),
                    "score": 0.50,
                    "rank": rank,
                    "explanation": [
                        "Popular item with top customer ratings across the catalog",
                        f"Featured in {self._extract_primary_category(meta)}",
                    ],
                    "shap_values": {f: 0.0 for f in FEATURE_NAMES},
                    "influenced_by": [],
                }
            )
        return results

    def _extract_primary_category(self, meta: dict[str, Any]) -> str:
        cats = meta.get("categories", set())
        if isinstance(cats, (set, list, tuple)) and len(cats) > 0:
            return sorted(list(cats))[0].title()
        return "Electronics"


# ── Factory Helper ───────────────────────────────────────────────────
_GLOBAL_RECOMMENDER: Phase1Recommender | None = None


def get_recommender(
    reviews_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    train_model: bool = True,
) -> Phase1Recommender:
    """Create and train a Phase1Recommender instance."""
    global _GLOBAL_RECOMMENDER
    if _GLOBAL_RECOMMENDER is not None and reviews_df is None:
        return _GLOBAL_RECOMMENDER

    cfg = config or load_config()
    if reviews_df is None or metadata_df is None:
        revs, meta, _ = load_data()
    else:
        revs, meta = reviews_df, metadata_df

    content_cfg = cfg.get("content", {})
    ranking_cfg = cfg.get("ranking", {})

    # 1. Content Vectorizer (Section 9)
    vectorizer = ContentVectorizer(
        max_features=content_cfg.get("max_features", 10000),
        min_df=content_cfg.get("min_df", 2),
    )
    vectorizer.fit_transform(meta)

    # 2. User Profile Builder & Popularity Recommender (Section 10)
    pop_rec = PopularityRecommender(meta, revs)
    profile_builder = UserProfileBuilder(
        vectorizer=vectorizer,
        rating_weights=content_cfg.get("rating_weights"),
        popularity_recommender=pop_rec,
        recency_decay=content_cfg.get("recency_decay", 0.85),
    )

    # 3. Feature Extractor (Section 12)
    feat_extractor = FeatureExtractor(
        metadata_df=meta,
        vectorizer=vectorizer,
        user_profile_builder=profile_builder,
        positive_threshold=ranking_cfg.get("positive_rating_threshold", 4.0),
        negative_threshold=ranking_cfg.get("negative_rating_threshold", 2.0),
    )

    # 4. Candidate Generator (Section 11)
    cand_gen = CandidateGenerator(vectorizer=vectorizer)

    # 5. XGBoost Ranker (Section 12)
    ranker = XGBoostRanker()
    if train_model:
        X, y = feat_extractor.build_training_dataset(revs)
        ranker.fit(X, y)

    # 6. SHAP Explainer (Section 13)
    shap_explainer = ShapExplainer(ranker) if ranker.is_fitted else None

    # 7. Explanation Engine (Section 14)
    explanation_engine = ExplanationEngine()

    recommender = Phase1Recommender(
        metadata_df=meta,
        reviews_df=revs,
        vectorizer=vectorizer,
        user_profile_builder=profile_builder,
        feature_extractor=feat_extractor,
        candidate_generator=cand_gen,
        ranker=ranker,
        shap_explainer=shap_explainer,
        explanation_engine=explanation_engine,
        config=cfg,
    )
    _GLOBAL_RECOMMENDER = recommender
    return recommender


def recommend(user_id: str, k: int = 10) -> list[dict[str, Any]]:
    """The Recommendation API entrypoint (Section 23)."""
    rec = get_recommender()
    return rec.recommend(user_id=user_id, k=k)


# ── Phase 2: Hybrid Recommender (Content + Collaborative Filtering) ──
class HybridRecommender:
    """Section 17 Hybrid Recommender combining Phase 1 Content (C) and Phase 2 CF."""

    def __init__(
        self,
        phase1_recommender: Phase1Recommender,
        cf_model: CollaborativeFilter | None = None,
        alpha: float = 0.5,
        cf_candidate_count: int = 50,
    ):
        self.phase1 = phase1_recommender
        self.cf_model = cf_model
        self.alpha = float(alpha)
        self.cf_candidate_count = int(cf_candidate_count)

    def recommend(
        self,
        user_id: str,
        k: int | None = None,
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k explainable hybrid recommendations: Score = alpha*C + (1-alpha)*CF."""
        top_k = k if k is not None else self.phase1.default_top_k
        blend_alpha = self.alpha if alpha is None else float(alpha)
        user_str = str(user_id)

        user_reviews = self.phase1.reviews_df[
            self.phase1.reviews_df["user_id"].astype(str) == user_str
        ]
        if user_reviews.empty:
            recs = self.phase1._recommend_cold_start(user_str, top_k)
            for r in recs:
                r["content_score"] = r["score"]
                r["cf_score"] = None
                r["cf_available"] = False
                r["alpha"] = blend_alpha
                r["features"] = {f: 0.0 for f in FEATURE_NAMES}
            return recs

        user_summary = self.phase1.feature_extractor.build_user_summary(user_str, user_reviews)
        if user_summary.profile_vector is None:
            recs = self.phase1._recommend_cold_start(user_str, top_k)
            for r in recs:
                r["content_score"] = r["score"]
                r["cf_score"] = None
                r["cf_available"] = False
                r["alpha"] = blend_alpha
                r["features"] = {f: 0.0 for f in FEATURE_NAMES}
            return recs

        # 1. Gather Phase 1 content candidates + Phase 2 CF candidates (strictly excluding seen)
        content_cands = self.phase1.candidate_generator.generate_candidates(
            user_profile=user_summary.profile_vector,
            user_interacted_asins=user_summary.interacted_asins,
            candidate_count=self.phase1.candidate_count,
        )
        merged_asins = [asin for asin, _ in content_cands]
        seen_set = set(merged_asins)

        if self.cf_model is not None and self.cf_model.is_fitted and self.cf_candidate_count > 0:
            cf_cands = self.cf_model.recommend_candidates(
                user_id=user_str,
                user_interacted_asins=user_summary.interacted_asins,
                top_n=self.cf_candidate_count,
            )
            for asin, _ in cf_cands:
                if asin not in seen_set and asin not in user_summary.interacted_asins:
                    merged_asins.append(asin)
                    seen_set.add(asin)

        if not merged_asins:
            return self.phase1._recommend_cold_start(user_str, top_k)

        # 2. Score all merged candidates with Phase 1 XGBoost ranker (C)
        feature_list = [
            self.phase1.feature_extractor.extract_features(user_summary, asin)
            for asin in merged_asins
        ]
        feature_df = pd.DataFrame(feature_list)[FEATURE_NAMES]

        if self.phase1.ranker.is_fitted:
            content_scores = self.phase1.ranker.predict_proba(feature_df)
        else:
            content_scores = feature_df["text_similarity"].values

        # 3. Compute CF scores (None if item unseen in CF training matrix)
        if self.cf_model is not None and self.cf_model.is_fitted:
            cf_scores_map = self.cf_model.predict_batch(user_str, merged_asins)
        else:
            cf_scores_map = {a: None for a in merged_asins}

        # 4. Combine via Section 17 formula: Score = alpha * C + (1 - alpha) * CF
        scored_candidates: list[dict[str, Any]] = []
        for asin, c_score, (_, feat_row) in zip(merged_asins, content_scores, feature_df.iterrows()):
            c_val = float(c_score)
            cf_val = cf_scores_map.get(asin)
            cf_available = cf_val is not None
            effective_cf = float(cf_val) if cf_available else c_val
            hybrid_score = blend_alpha * c_val + (1.0 - blend_alpha) * effective_cf

            cand_meta = self.phase1.feature_extractor.metadata_lookup.get(asin, {})
            scored_candidates.append(
                {
                    "parent_asin": asin,
                    "score": float(hybrid_score),
                    "content_score": c_val,
                    "cf_score": float(cf_val) if cf_available else None,
                    "cf_available": cf_available,
                    "brand": cand_meta.get("brand", ""),
                    "features": feat_row.to_dict(),
                }
            )

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 5. Brand diversity cap (Section 11)
        cap = self.phase1.diversity_cap_per_brand
        final_cands: list[dict[str, Any]] = []
        brand_counts: dict[str, int] = {}
        deferred: list[dict[str, Any]] = []

        for cand in scored_candidates:
            brand = cand["brand"]
            cnt = brand_counts.get(brand, 0)
            if not brand or cnt < cap:
                cand["rank"] = len(final_cands) + 1
                final_cands.append(cand)
                if brand:
                    brand_counts[brand] = cnt + 1
                if len(final_cands) >= top_k:
                    break
            else:
                deferred.append(cand)

        if len(final_cands) < top_k:
            for cand in deferred:
                cand["rank"] = len(final_cands) + 1
                final_cands.append(cand)
                if len(final_cands) >= top_k:
                    break

        # 6. Populate Section 23 contract with SHAP + explanations
        results: list[dict[str, Any]] = []
        for cand in final_cands:
            asin = cand["parent_asin"]
            meta = self.phase1.feature_extractor.metadata_lookup.get(asin, {})
            features = cand["features"]

            shap_dict: dict[str, float] = {}
            if self.phase1.shap_explainer is not None:
                try:
                    shap_dict = self.phase1.shap_explainer.explain_instance(features)
                except Exception as e:
                    log.warning("SHAP explanation failed for %s: %s", asin, e)
                    shap_dict = {f: 0.0 for f in FEATURE_NAMES}

            explanations = self.phase1.explanation_engine.get_plain_sentences(
                shap_values=shap_dict,
                product_metadata=meta,
                top_n=4,
            )
            if cand["cf_available"] and blend_alpha < 1.0:
                explanations.append(
                    f"Collaborative affinity from similar users (CF score: {cand['cf_score']:.2f})"
                )

            results.append(
                {
                    "product_id": asin,
                    "title": meta.get("title", ""),
                    "brand": meta.get("brand", ""),
                    "category": self.phase1._extract_primary_category(meta),
                    "price": None if np.isnan(meta.get("price", np.nan)) else meta.get("price"),
                    "score": round(float(cand["score"]), 4),
                    "rank": int(cand["rank"]),
                    "explanation": explanations,
                    "shap_values": shap_dict,
                    "influenced_by": [],  # Populated in Phase 4
                    "content_score": round(float(cand["content_score"]), 4),
                    "cf_score": round(float(cand["cf_score"]), 4) if cand["cf_available"] else None,
                    "cf_available": cand["cf_available"],
                    "alpha": round(float(blend_alpha), 4),
                    "features": {k: round(float(v), 4) for k, v in features.items()},
                }
            )

        return results


def get_hybrid_recommender(
    reviews_df: pd.DataFrame | None = None,
    metadata_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    models_dir: str | Path | None = None,
    use_cache: bool = True,
) -> HybridRecommender:
    """Build or load cached Phase 1 + Phase 2 HybridRecommender."""
    cfg = config or load_config()
    if reviews_df is None or metadata_df is None:
        revs, meta, _ = load_data()
    else:
        revs, meta = reviews_df, metadata_df

    project_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(models_dir) if models_dir else (project_root / "models")
    vec_path = cache_dir / "content_vectorizer.pkl"
    ranker_path = cache_dir / "xgboost_ranker.pkl"
    cf_path = cache_dir / "cf_model.pkl"

    content_cfg = cfg.get("content", {})
    ranking_cfg = cfg.get("ranking", {})

    if use_cache and vec_path.exists() and ranker_path.exists() and cf_path.exists():
        log.info("Loading cached Phase 1 & 2 models from %s", cache_dir)
        vectorizer = ContentVectorizer.load(vec_path)
        ranker = XGBoostRanker.load(ranker_path)
        cf_model = CollaborativeFilter.load(cf_path)

        pop_rec = PopularityRecommender(meta, revs)
        profile_builder = UserProfileBuilder(
            vectorizer=vectorizer,
            rating_weights=content_cfg.get("rating_weights"),
            popularity_recommender=pop_rec,
            recency_decay=content_cfg.get("recency_decay", 0.85),
        )
        feat_extractor = FeatureExtractor(
            metadata_df=meta,
            vectorizer=vectorizer,
            user_profile_builder=profile_builder,
            positive_threshold=ranking_cfg.get("positive_rating_threshold", 4.0),
            negative_threshold=ranking_cfg.get("negative_rating_threshold", 2.0),
        )
        cand_gen = CandidateGenerator(vectorizer=vectorizer)
        shap_explainer = ShapExplainer(ranker) if ranker.is_fitted else None
        p1 = Phase1Recommender(
            metadata_df=meta,
            reviews_df=revs,
            vectorizer=vectorizer,
            user_profile_builder=profile_builder,
            feature_extractor=feat_extractor,
            candidate_generator=cand_gen,
            ranker=ranker,
            shap_explainer=shap_explainer,
            explanation_engine=ExplanationEngine(),
            config=cfg,
        )
    else:
        p1 = get_recommender(reviews_df=revs, metadata_df=meta, config=cfg, train_model=True)
        cf_model = CollaborativeFilter(n_factors=32, random_state=42).fit(revs)
        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            p1.vectorizer.save(vec_path)
            p1.ranker.save(ranker_path)
            cf_model.save(cf_path)

    alpha = cfg.get("hybrid", {}).get("alpha", 0.5)
    return HybridRecommender(phase1_recommender=p1, cf_model=cf_model, alpha=alpha)

