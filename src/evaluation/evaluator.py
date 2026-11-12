"""
Phase 1 Recommender Evaluation Runner — Section 20 of the specification.

Executes temporal evaluation with strict leakage prevention:
  - Trains recommender components exclusively on train_df.
  - Generates Top-10 recommendations for each user in test_df.
  - Computes Precision@K, Recall@K, and NDCG@K (K=5, 10).
  - Segregates metrics for Seen-in-Train (Group A) vs. Cold-Start (Group B) items.
  - Extracts 3 representative worked SHAP examples.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.content.text_features import ContentVectorizer
from src.content.user_profile import PopularityRecommender, UserProfileBuilder
from src.data.splitter import assert_no_temporal_leakage, temporal_train_test_split
from src.evaluation.ranking_metrics import aggregate_metrics, evaluate_user_recommendations
from src.explainability.explanation_generator import ExplanationEngine
from src.explainability.shap_explainer import ShapExplainer
from src.pipeline.recommender import Phase1Recommender
from src.ranking.features import FEATURE_NAMES, FeatureExtractor
from src.ranking.ranker import CandidateGenerator, XGBoostRanker

log = logging.getLogger(__name__)


class Phase1Evaluator:
    """Executes Section 20 evaluation on Phase 1 Content-Based Recommender."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        config: dict[str, Any] | None = None,
        train_max_samples: int | None = None,
    ):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.metadata_df = metadata_df.copy()
        self.config = config or {}
        self.train_max_samples = train_max_samples

        # Enforce temporal leakage invariant
        assert_no_temporal_leakage(self.train_df, self.test_df)

        self.recommender: Phase1Recommender | None = None
        self._build_recommender()

    def _build_recommender(self) -> None:
        """Initialize and fit Phase1Recommender strictly on training data."""
        log.info("Building Phase 1 Recommender strictly using train_df (Zero Leakage)")

        cfg = self.config
        content_cfg = cfg.get("content", {})
        ranking_cfg = cfg.get("ranking", {})

        # 1. Product Content Vectorizer on catalog metadata
        vectorizer = ContentVectorizer(
            max_features=content_cfg.get("max_features", 10000),
            min_df=content_cfg.get("min_df", 2),
        )
        vectorizer.fit_transform(self.metadata_df)

        # 2. User Profile Builder using TRAIN interactions only
        self.popularity_recommender = PopularityRecommender(self.metadata_df, self.train_df)
        profile_builder = UserProfileBuilder(
            vectorizer=vectorizer,
            rating_weights=content_cfg.get("rating_weights"),
            popularity_recommender=self.popularity_recommender,
            recency_decay=content_cfg.get("recency_decay", 0.85),
        )

        # 3. Feature Extractor using TRAIN interactions only
        feat_extractor = FeatureExtractor(
            metadata_df=self.metadata_df,
            vectorizer=vectorizer,
            user_profile_builder=profile_builder,
            positive_threshold=ranking_cfg.get("positive_rating_threshold", 4.0),
            negative_threshold=ranking_cfg.get("negative_rating_threshold", 2.0),
        )

        # 4. Candidate Generator
        cand_gen = CandidateGenerator(vectorizer=vectorizer)

        # 5. XGBoost Ranker trained strictly on TRAIN interactions (Leave-One-Out)
        X, y = feat_extractor.build_training_dataset(
            self.train_df, max_samples=self.train_max_samples
        )
        ranker = XGBoostRanker(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        )
        ranker.fit(X, y)

        # 6. SHAP Explainer
        shap_explainer = ShapExplainer(ranker) if ranker.is_fitted else None

        # 7. Explanation Engine
        explanation_engine = ExplanationEngine()

        # Build Recommender with train_df as the active review history
        self.recommender = Phase1Recommender(
            metadata_df=self.metadata_df,
            reviews_df=self.train_df,  # Training history only!
            vectorizer=vectorizer,
            user_profile_builder=profile_builder,
            feature_extractor=feat_extractor,
            candidate_generator=cand_gen,
            ranker=ranker,
            shap_explainer=shap_explainer,
            explanation_engine=explanation_engine,
            config=cfg,
        )
        log.info("Phase 1 Recommender successfully trained on train_df")

    def run_evaluation(
        self, k_values: tuple[int, ...] = (5, 10), max_users: int | None = None
    ) -> dict[str, Any]:
        """Evaluate Phase 1 recommender across test users.

        Returns:
            Dictionary with overall metrics, Group A (seen) metrics, Group B (cold-start) metrics,
            and worked SHAP examples.
        """
        if self.recommender is None:
            raise RuntimeError("Recommender is not initialized.")

        test_users = self.test_df["user_id"].unique().tolist()
        if max_users:
            test_users = test_users[:max_users]

        train_item_set = set(self.train_df["parent_asin"].unique())

        all_user_metrics: list[dict[str, float]] = []
        pop_user_metrics: list[dict[str, float]] = []
        group_a_metrics: list[dict[str, float]] = []  # Item seen in training
        group_b_metrics: list[dict[str, float]] = []  # Cold-start item (unseen in training)

        raw_user_results: list[dict[str, Any]] = []
        rec_feature_rows: list[dict[str, float]] = []

        max_k = max(k_values)
        # Pre-fetch enough popular items to filter per-user training history
        pop_pool = (
            self.popularity_recommender.recommend(top_k=max_k + 200, max_per_category=2)
            if getattr(self, "popularity_recommender", None) is not None
            else []
        )
        train_items_by_user = {
            str(uid): set(grp["parent_asin"].astype(str))
            for uid, grp in self.train_df.groupby("user_id")
        }

        log.info("Starting temporal evaluation on %d test users", len(test_users))

        for uid in test_users:
            user_test_revs = self.test_df[self.test_df["user_id"] == uid]
            ground_truth_items = set(user_test_revs["parent_asin"].astype(str))

            # Popularity baseline (excluding user's own training history)
            user_seen = train_items_by_user.get(str(uid), set())
            pop_recs = [asin for asin in pop_pool if asin not in user_seen][:max_k]
            pop_user_metrics.append(
                evaluate_user_recommendations(
                    recommended=pop_recs,
                    ground_truth=ground_truth_items,
                    k_values=k_values,
                )
            )

            # Generate top-K Phase 1 recommendations using training history
            recs = self.recommender.recommend(uid, k=max_k)
            rec_asins = [str(r["product_id"]) for r in recs]

            # Collect top recommendation feature vector for global SHAP importance
            if recs and self.recommender.shap_explainer is not None:
                user_train = self.train_df[self.train_df["user_id"] == uid]
                u_sum = self.recommender.feature_extractor.build_user_summary(uid, user_train)
                rec_feature_rows.append(
                    self.recommender.feature_extractor.extract_features(u_sum, rec_asins[0])
                )

            # Calculate metrics
            u_metrics = evaluate_user_recommendations(
                recommended=rec_asins,
                ground_truth=ground_truth_items,
                k_values=k_values,
            )
            all_user_metrics.append(u_metrics)

            # Check whether ground truth was seen in training
            is_seen = any(asin in train_item_set for asin in ground_truth_items)
            if is_seen:
                group_a_metrics.append(u_metrics)
            else:
                group_b_metrics.append(u_metrics)

            raw_user_results.append(
                {
                    "user_id": uid,
                    "ground_truth": list(ground_truth_items),
                    "is_seen_in_train": is_seen,
                    "recommendations": recs,
                    "metrics": u_metrics,
                }
            )

        # Aggregate metrics
        overall_summary = aggregate_metrics(all_user_metrics)
        popularity_summary = aggregate_metrics(pop_user_metrics)
        group_a_summary = aggregate_metrics(group_a_metrics) if group_a_metrics else {}
        group_b_summary = aggregate_metrics(group_b_metrics) if group_b_metrics else {}

        # Global SHAP feature importance (Section 13)
        global_shap_importance: dict[str, float] = {}
        if rec_feature_rows and self.recommender.shap_explainer is not None:
            feat_df = pd.DataFrame(rec_feature_rows)[FEATURE_NAMES]
            global_shap_importance = self.recommender.shap_explainer.get_global_feature_importance(feat_df)

        # Select 3 representative worked SHAP examples from active recommendations
        worked_examples = self._extract_worked_shap_examples(raw_user_results, count=3)

        return {
            "num_evaluated_users": len(test_users),
            "num_excluded_users": 0,
            "popularity_baseline_metrics": popularity_summary,
            "overall_metrics": overall_summary,
            "global_shap_importance": global_shap_importance,
            "cold_start_breakdown": {
                "group_a_seen_in_train": {
                    "count": len(group_a_metrics),
                    "metrics": group_a_summary,
                },
                "group_b_unseen_in_train": {
                    "count": len(group_b_metrics),
                    "metrics": group_b_summary,
                },
            },
            "worked_shap_examples": worked_examples,
        }

    def _extract_worked_shap_examples(
        self, raw_user_results: list[dict[str, Any]], count: int = 3
    ) -> list[dict[str, Any]]:
        """Extract diverse, representative worked SHAP examples with full details."""
        examples: list[dict[str, Any]] = []

        for res in raw_user_results:
            recs = res["recommendations"]
            if not recs:
                continue

            # Pick top-ranked recommendation
            top_rec = recs[0]
            shap_vals = top_rec.get("shap_values", {})
            # Include recommendation example with SHAP values
            if shap_vals:
                # Sort SHAP contributions
                pos_contribs = sorted(
                    [(k, v) for k, v in shap_vals.items() if v > 0],
                    key=lambda x: x[1],
                    reverse=True,
                )
                neg_contribs = sorted(
                    [(k, v) for k, v in shap_vals.items() if v < 0],
                    key=lambda x: x[1],
                )

                # Get features from lookup
                asin = top_rec["product_id"]
                u_summary = self.recommender.feature_extractor.build_user_summary(
                    res["user_id"],
                    self.train_df[self.train_df["user_id"] == res["user_id"]],
                )
                feat_dict = self.recommender.feature_extractor.extract_features(
                    u_summary, asin
                )

                examples.append(
                    {
                        "user_id": res["user_id"][:12] + "...",
                        "recommended_product": {
                            "product_id": asin,
                            "title": top_rec.get("title", "Unknown"),
                            "brand": top_rec.get("brand", "Unknown"),
                            "category": top_rec.get("category", "Unknown"),
                            "price": top_rec.get("price"),
                        },
                        "predicted_score": top_rec.get("score"),
                        "features": {k: round(v, 4) for k, v in feat_dict.items()},
                        "shap_values": shap_vals,
                        "top_positive_features": pos_contribs[:3],
                        "top_negative_features": neg_contribs[:3],
                        "final_explanation": top_rec.get("explanation", []),
                    }
                )
                if len(examples) >= count:
                    break

        return examples


def _plot_shap_examples(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Render horizontal SHAP contribution bar charts with readable feature labels."""
    if not examples:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping SHAP plot generation.")
        return

    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.8), squeeze=False)
    fig.suptitle(
        "Phase 1 SHAP Feature Contributions — Worked Examples",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, ex in enumerate(examples):
        ax = axes[0, idx]
        shap_vals = ex["shap_values"]
        items = sorted(shap_vals.items(), key=lambda x: abs(x[1]))
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

        ax.barh(labels, values, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_title(
            f"Example {idx + 1}: User {ex['user_id']}\nScore: {ex['predicted_score']:.4f}",
            fontsize=10.5,
        )
        ax.set_xlabel("SHAP Contribution", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved SHAP worked examples chart to %s", output_path)


def save_phase1_reports(results: dict[str, Any], catalog_size: int, reports_dir: Path) -> None:
    """Persist Phase 1 JSON results, Markdown evaluation report, and SHAP plot."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "phase1_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pop = results.get("popularity_baseline_metrics", {})
    ov = results.get("overall_metrics", {})
    ga = results.get("cold_start_breakdown", {}).get("group_a_seen_in_train", {})
    gb = results.get("cold_start_breakdown", {}).get("group_b_unseen_in_train", {})
    ga_m = ga.get("metrics", {})
    gb_m = gb.get("metrics", {})
    global_shap = results.get("global_shap_importance", {})
    examples = results.get("worked_shap_examples", [])

    _plot_shap_examples(examples, reports_dir / "shap_worked_examples.png")

    n_users = results["num_evaluated_users"]
    ga_cnt = ga.get("count", 0)
    gb_cnt = gb.get("count", 0)
    ga_pct = (ga_cnt / n_users * 100.0) if n_users else 0.0
    gb_pct = (gb_cnt / n_users * 100.0) if n_users else 0.0

    lines = [
        "# Section 20: Phase 1 Evaluation Report",
        "**Model**: Content-Based Filtering + Leave-One-Out XGBoost Ranker + SHAP Explainability  ",
        "**Dataset**: Amazon Reviews 2023 (`Electronics` category subset)  ",
        f"**Catalog Size**: {catalog_size:,} items  ",
        "**Split Protocol**: Chronological Leave-One-Out (Section 8)  ",
        "**Leakage Prevention**: Strictly Verified (temporal train/test ordering + LOO ranking training)",
        "",
        "---",
        "",
        "## 1. Recommendation Quality — Popularity Baseline vs. Phase 1",
        "",
        "| Model | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Evaluated Users |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        (
            f"| Popularity Baseline | {pop.get('precision@5', 0.0):.4f} | {pop.get('recall@5', 0.0):.4f} | "
            f"{pop.get('ndcg@5', 0.0):.4f} | {pop.get('precision@10', 0.0):.4f} | "
            f"{pop.get('recall@10', 0.0):.4f} | {pop.get('ndcg@10', 0.0):.4f} | {n_users} |"
        ),
        (
            f"| **Content-Based + XGBoost + SHAP (Phase 1)** | **{ov.get('precision@5', 0.0):.4f}** | "
            f"**{ov.get('recall@5', 0.0):.4f}** | **{ov.get('ndcg@5', 0.0):.4f}** | "
            f"**{ov.get('precision@10', 0.0):.4f}** | **{ov.get('recall@10', 0.0):.4f}** | "
            f"**{ov.get('ndcg@10', 0.0):.4f}** | **{n_users}** |"
        ),
        "",
        "---",
        "",
        "## 2. Cold-Start vs. Seen-in-Training Analysis",
        "",
        "| Group | Description | User Count | % of Test Users | Precision@10 | Recall@10 | NDCG@10 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        (
            f"| **Group A** | Held-out item **SEEN** in training history | {ga_cnt} | {ga_pct:.1f}% | "
            f"{ga_m.get('precision@10', 0.0):.4f} | {ga_m.get('recall@10', 0.0):.4f} | {ga_m.get('ndcg@10', 0.0):.4f} |"
        ),
        (
            f"| **Group B** | Held-out item **UNSEEN** in training (Catalog Cold-Start) | {gb_cnt} | {gb_pct:.1f}% | "
            f"{gb_m.get('precision@10', 0.0):.4f} | {gb_m.get('recall@10', 0.0):.4f} | {gb_m.get('ndcg@10', 0.0):.4f} |"
        ),
        (
            f"| **Overall** | Combined test population | {n_users} | 100.0% | "
            f"{ov.get('precision@10', 0.0):.4f} | {ov.get('recall@10', 0.0):.4f} | {ov.get('ndcg@10', 0.0):.4f} |"
        ),
        "",
        "---",
        "",
        "## 3. Global SHAP Feature Importance (Section 13)",
        "",
        "| Feature | Mean |SHAP Value| |",
        "| :--- | :--- |",
    ]
    for feat, imp in global_shap.items():
        lines.append(f"| `{feat}` | {imp:.4f} |")

    lines.extend(["", "---", "", "## 4. Worked SHAP Examples", "", "![SHAP Contributions](shap_worked_examples.png)", ""])
    for idx, ex in enumerate(examples, 1):
        prod = ex["recommended_product"]
        price_str = f"${prod['price']:.2f}" if prod.get("price") is not None else "$None"
        lines.extend([
            f"### Example {idx}: User `{ex['user_id']}`",
            f"- **Recommended Product**: `[{prod['product_id']}]` {prod['title']}",
            f"- **Brand**: `{prod['brand']}` | **Category**: `{prod['category']}` | **Price**: {price_str}",
            f"- **Predicted Ranking Score $P(\\text{{like}})$**: `{ex['predicted_score']:.4f}`",
            "- **SHAP Contributions**:",
        ])
        for feat, val in sorted(ex["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  - `{feat}`: {val:+.4f}")
        lines.append("- **Natural-Language Explanations**:")
        for sent in ex["final_explanation"]:
            lines.append(f"  - *{sent}*")
        lines.append("")

    md_path = reports_dir / "phase1_evaluation_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved evaluation report to %s and %s", json_path, md_path)


def run_phase2_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    alphas: tuple[float, ...] = (0.7, 0.5, 0.3),
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Run Section 20 comparison across Popularity, Phase 1, and Phase 2 Hybrid alphas."""
    from src.collaborative.cf_model import CollaborativeFilter
    from src.pipeline.recommender import HybridRecommender

    p1_eval = Phase1Evaluator(
        train_df=train_df,
        test_df=test_df,
        metadata_df=metadata_df,
        config=config,
    )
    p1_results = p1_eval.run_evaluation(k_values=k_values)

    cf_model = CollaborativeFilter(n_factors=32, random_state=42).fit(train_df)
    hybrid = HybridRecommender(phase1_recommender=p1_eval.recommender, cf_model=cf_model)

    test_users = test_df["user_id"].unique().tolist()
    train_item_set = set(train_df["parent_asin"].unique())
    max_k = max(k_values)

    alpha_results: dict[str, Any] = {}
    for alpha in alphas:
        all_m, ga_m, gb_m = [], [], []
        for uid in test_users:
            gt = set(test_df[test_df["user_id"] == uid]["parent_asin"].astype(str))
            recs = hybrid.recommend(uid, k=max_k, alpha=alpha)
            rec_asins = [str(r["product_id"]) for r in recs]
            u_m = evaluate_user_recommendations(rec_asins, gt, k_values=k_values)
            all_m.append(u_m)
            if any(a in train_item_set for a in gt):
                ga_m.append(u_m)
            else:
                gb_m.append(u_m)

        alpha_results[f"alpha_{alpha:.1f}"] = {
            "alpha": alpha,
            "overall_metrics": aggregate_metrics(all_m),
            "group_a_seen_metrics": aggregate_metrics(ga_m) if ga_m else {},
            "group_b_unseen_metrics": aggregate_metrics(gb_m) if gb_m else {},
        }

    return {
        "num_evaluated_users": len(test_users),
        "catalog_size": len(metadata_df),
        "popularity_baseline_metrics": p1_results["popularity_baseline_metrics"],
        "phase1_metrics": p1_results["overall_metrics"],
        "phase1_cold_start_breakdown": p1_results["cold_start_breakdown"],
        "global_shap_importance": p1_results["global_shap_importance"],
        "phase2_alpha_sweep": alpha_results,
        "worked_shap_examples": p1_results["worked_shap_examples"],
    }


def save_phase2_reports(results: dict[str, Any], reports_dir: Path) -> None:
    """Persist Phase 1 vs Phase 2 evaluation JSON and Markdown report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "phase2_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pop = results["popularity_baseline_metrics"]
    p1 = results["phase1_metrics"]
    n_users = results["num_evaluated_users"]

    def _row(label: str, m: dict[str, float]) -> str:
        return (
            f"| {label} | {m.get('precision@5', 0.0):.4f} | {m.get('recall@5', 0.0):.4f} | "
            f"{m.get('ndcg@5', 0.0):.4f} | {m.get('precision@10', 0.0):.4f} | "
            f"{m.get('recall@10', 0.0):.4f} | {m.get('ndcg@10', 0.0):.4f} | {n_users} |"
        )

    lines = [
        "# Section 20: Phase 1 vs. Phase 2 Evaluation Report",
        f"**Dataset**: Amazon Reviews 2023 (`Electronics` subset, {results['catalog_size']:,} items)  ",
        "**Split Protocol**: Chronological Leave-One-Out (Section 8, Zero Leakage Verified)",
        "",
        "## Recommendation Quality Comparison",
        "",
        "| Model | Precision@5 | Recall@5 | NDCG@5 | Precision@10 | Recall@10 | NDCG@10 | Users |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        _row("Popularity Baseline", pop),
        _row("Phase 1 (Content + XGBoost + SHAP)", p1),
    ]
    for key in sorted(results["phase2_alpha_sweep"].keys(), reverse=True):
        entry = results["phase2_alpha_sweep"][key]
        lines.append(_row(f"Phase 2 Hybrid (alpha={entry['alpha']:.1f})", entry["overall_metrics"]))

    md_path = reports_dir / "phase2_evaluation_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved Phase 2 evaluation report to %s and %s", json_path, md_path)


if __name__ == "__main__":
    import argparse
    from src.data.loader import load_data

    parser = argparse.ArgumentParser(description="Section 20 Evaluation Runner")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="1 for Phase 1, 2 for Phase 1+2 comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    revs, meta, cfg = load_data()
    train_df, test_df = temporal_train_test_split(revs, holdout_k=1)
    reports_dir = Path(__file__).resolve().parents[2] / "reports"

    if args.phase == 1:
        evaluator = Phase1Evaluator(train_df=train_df, test_df=test_df, metadata_df=meta, config=cfg)
        results = evaluator.run_evaluation(k_values=(5, 10))
        save_phase1_reports(results, catalog_size=len(meta), reports_dir=reports_dir)
    else:
        results = run_phase2_evaluation(train_df=train_df, test_df=test_df, metadata_df=meta, config=cfg)
        save_phase2_reports(results, reports_dir=reports_dir)

