"""
Streamlit Explainable Hybrid Recommender Prototype (Phases 1 & 2)

Showcases:
  - Phase 1: Content-Based Filtering (TF-IDF + Recency Profile) + LOO XGBoost + SHAP Explainability
  - Phase 2: SVD Collaborative Filtering + Interactive Hybrid Blending (Score = α·C + (1-α)·CF)
  - Local per-item SHAP bar charts + deterministic plain-language explanations
  - Global SHAP feature importance & Section 20 offline evaluation metrics

Run:
    streamlit run app.py
"""

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_config
from src.pipeline.recommender import get_hybrid_recommender
from src.ranking.features import FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_LABELS = {
    "text_similarity": "Text Similarity (TF-IDF)",
    "category_similarity": "Category Affinity",
    "brand_match": "Brand Match",
    "feature_similarity": "Feature Token Match",
    "price_similarity": "Price Proximity",
    "rating_similarity": "Rating Alignment",
}


st.set_page_config(
    page_title="Explainable Hybrid Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading dataset and Phase 1 + Phase 2 models...")
def load_engine():
    cfg = load_config()

    # Load only the columns required for runtime UI + inference to stay well under 1 GB RAM
    reviews_df = (
        pd.read_parquet(
            PROCESSED_DIR / "reviews_subset.parquet",
            columns=["user_id", "parent_asin", "rating", "timestamp"],
        )
        .drop_duplicates(subset=["user_id", "parent_asin", "timestamp"])
        .dropna(subset=["user_id", "parent_asin", "rating"])
    )
    reviews_df["parent_asin"] = reviews_df["parent_asin"].astype(str)

    # Check if cached models exist; only load `description` if we need to train from scratch
    models_dir = PROJECT_ROOT / "models"
    has_cache = all(
        (models_dir / fname).exists()
        for fname in ("content_vectorizer.pkl", "xgboost_ranker.pkl", "cf_model.pkl")
    )
    meta_cols = [
        "parent_asin",
        "main_category",
        "title",
        "average_rating",
        "rating_number",
        "features",
        "price",
        "store",
        "categories",
    ]
    if not has_cache:
        meta_cols.append("description")

    metadata_df = (
        pd.read_parquet(PROCESSED_DIR / "metadata_subset.parquet", columns=meta_cols)
        .drop_duplicates(subset=["parent_asin"], keep="first")
        .dropna(subset=["parent_asin", "title"])
    )
    metadata_df["parent_asin"] = metadata_df["parent_asin"].astype(str)
    for col in ("features", "categories", "description"):
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else v
            )

    hybrid = get_hybrid_recommender(
        reviews_df=reviews_df,
        metadata_df=metadata_df,
        config=cfg,
        use_cache=True,
    )
    if hybrid.phase1.vectorizer.tfidf_matrix is not None:
        hybrid.phase1.vectorizer.tfidf_matrix = hybrid.phase1.vectorizer.tfidf_matrix.astype(np.float32)

    catalog_size = len(metadata_df)
    hybrid.phase1.metadata_df = pd.DataFrame()
    del metadata_df
    gc.collect()

    meta_lookup = hybrid.phase1.feature_extractor.metadata_lookup
    user_counts = reviews_df.groupby("user_id").size().sort_values(ascending=False)
    return reviews_df, catalog_size, cfg, hybrid, meta_lookup, user_counts


def load_eval_reports() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    p2_path = REPORTS_DIR / "phase2_evaluation_results.json"
    p1_path = REPORTS_DIR / "phase1_evaluation_results.json"
    p2 = json.loads(p2_path.read_text(encoding="utf-8")) if p2_path.exists() else None
    p1 = json.loads(p1_path.read_text(encoding="utf-8")) if p1_path.exists() else None
    return p1, p2


def make_shap_chart(shap_values: dict[str, float], title: str) -> go.Figure:
    items = sorted(shap_values.items(), key=lambda x: abs(x[1]))
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    vals = [float(v) for _, v in items]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    max_abs = max([abs(v) for v in vals] + [0.1]) * 1.28
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="SHAP Contribution (log-odds)", range=[-max_abs, max_abs], zeroline=True, zerolinewidth=1.5, zerolinecolor="#64748b"),
        yaxis=dict(title=""),
        height=265,
        margin=dict(l=10, r=25, t=35, b=30),
        template="plotly_white",
    )
    return fig


def make_global_shap_chart(global_shap: dict[str, float]) -> go.Figure:
    items = sorted(global_shap.items(), key=lambda x: x[1])
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    vals = [float(v) for _, v in items]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP| Across 500 Users)",
        xaxis=dict(title="Mean |SHAP Value|", range=[0, max(vals + [0.1]) * 1.22]),
        height=320,
        margin=dict(l=10, r=30, t=45, b=35),
        template="plotly_white",
    )
    return fig


def main():
    reviews_df, catalog_size, cfg, hybrid, meta_lookup, user_counts = load_engine()
    p1_report, p2_report = load_eval_reports()

    st.title("Explainable Hybrid Recommendation System")
    st.caption(
        "Amazon Electronics Dataset | Phase 1 (Content + LOO XGBoost + SHAP) & Phase 2 (SVD Collaborative Filtering Hybrid)"
    )

    # ── Sidebar Controls ────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        mode = st.radio(
            "Recommendation Engine",
            [
                "Phase 2: Hybrid (Content + CF)",
                "Phase 1: Content + XGBoost + SHAP",
            ],
            index=0,
        )
        is_phase2 = mode.startswith("Phase 2")

        default_alpha = float(cfg.get("hybrid", {}).get("alpha", 0.5))
        if is_phase2:
            alpha = st.slider(
                "Hybrid Weight α (Content vs. CF)",
                min_value=0.0,
                max_value=1.0,
                value=default_alpha,
                step=0.05,
                help="Score = α · Content + (1 - α) · CF",
            )
            st.caption(f"Formula: `{alpha:.2f} × Content + {1 - alpha:.2f} × CF`")
        else:
            alpha = 1.0
            st.info("Phase 1 uses Content-Based + XGBoost + SHAP only (`α = 1.00`).")

        st.divider()
        st.subheader("User Selection")
        user_mode = st.radio(
            "User Type",
            ["Existing Active User", "Simulate Cold-Start New User"],
            index=0,
        )

        featured_users = [
            u for u in ["AFZUK3MTBIBEDQOPAK3OATUOUKLA", "AE37Q6TCPNWG4RUJMJIKRYFBI4CQ", "AE3LZ5EMSN5BJ5USW4SJRDHI5M4A"]
            if u in user_counts.index
        ]
        all_users = featured_users + [u for u in user_counts.index.tolist() if u not in featured_users]

        if user_mode == "Existing Active User":
            selected_user = st.selectbox(
                "Choose User ID",
                options=all_users,
                format_func=lambda u: f"{u[:16]}... ({int(user_counts[u])} reviews)"
                + (" ★ Demo" if u in featured_users else ""),
            )
        else:
            selected_user = "NEW_COLD_START_USER_DEMO"
            st.warning("Cold-start mode: user has 0 history -> Popularity fallback active.")

        top_k = st.slider("Top-K Recommendations", min_value=3, max_value=15, value=5, step=1)

        st.divider()
        st.markdown(
            f"""
            **Active Safeguards**
            - Canonical ID: `parent_asin`
            - Already-interacted items filtered
            - Brand diversity cap: `{cfg.get('retrieval', {}).get('diversity_cap_per_brand', 2)} / brand`
            - Recency decay: `{cfg.get('content', {}).get('recency_decay', 0.85)}`
            """
        )

    # ── Main Tabs ───────────────────────────────────────────────────
    tab_live, tab_eval = st.tabs(
        [
            "🔍 Live Recommendations & SHAP Proof",
            "📊 Global Explainability & Evaluation",
        ]
    )

    with tab_live:
        user_hist = reviews_df[reviews_df["user_id"] == selected_user].sort_values(
            "timestamp", ascending=False
        )
        is_cold = user_hist.empty

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Mode", "Phase 2 Hybrid" if is_phase2 else "Phase 1 Content")
        col2.metric("User History", "0 (Cold-Start)" if is_cold else f"{len(user_hist)} reviews")
        col3.metric("Catalog Size", f"{catalog_size:,} items")
        col4.metric("Blend α", f"{alpha:.2f}" if is_phase2 else "1.00 (Content)")

        if not is_cold:
            # Collect all brands in user history and brands with positive (>= 4 stars) ratings
            user_brands_counter = {}
            for asin in user_hist["parent_asin"]:
                b = str(meta_lookup.get(str(asin), {}).get("brand", "")).strip()
                if b:
                    user_brands_counter[b.title()] = user_brands_counter.get(b.title(), 0) + 1

            all_user_brands_sorted = sorted(user_brands_counter.keys())

            user_pos_revs = user_hist[user_hist["rating"] >= 4.0]
            preferred_brands_list = []
            for asin in user_pos_revs["parent_asin"]:
                b = str(meta_lookup.get(str(asin), {}).get("brand", "")).strip()
                if b and b.title() not in preferred_brands_list:
                    preferred_brands_list.append(b.title())

            with st.expander(
                f"📜 User Purchase History ({len(user_hist)} total reviews across {len(user_brands_counter)} brands)",
                expanded=True,
            ):
                if preferred_brands_list:
                    st.caption(
                        f"**Brands rated 4–5★ across full history ({len(preferred_brands_list)} total):** "
                        + ", ".join(preferred_brands_list[:12])
                        + (f", and {len(preferred_brands_list) - 12} more..." if len(preferred_brands_list) > 12 else "")
                    )

                f_col1, f_col2, f_col3 = st.columns([1.2, 1.2, 1.6])
                with f_col1:
                    star_filter = st.multiselect(
                        "Filter by Stars",
                        options=[5, 4, 3, 2, 1],
                        default=[5, 4, 3, 2, 1],
                        format_func=lambda s: f"{s} ★",
                        key="hist_stars_select",
                        help="Filter purchase history by star ratings (e.g. select 4★ and 5★)",
                    )
                with f_col2:
                    brand_filter = st.selectbox(
                        "Filter by Brand",
                        options=["All Brands"] + [f"{b} ({user_brands_counter[b]})" for b in all_user_brands_sorted],
                        index=0,
                        key="hist_brand_select",
                    )
                with f_col3:
                    hist_search = st.text_input(
                        "Search by title / keyword / ASIN",
                        placeholder="e.g. cable, buds, mount (press Enter)...",
                        key="hist_search_input",
                    )

                displayed_hist = user_hist

                # 1. Filter by Stars (if user unselects all, show none; otherwise match selected stars)
                if star_filter:
                    allowed_stars = set(star_filter)
                    displayed_hist = displayed_hist[
                        displayed_hist["rating"].apply(lambda r: int(round(float(r))) in allowed_stars)
                    ]
                else:
                    displayed_hist = displayed_hist.iloc[0:0]

                # 2. Filter by Brand Dropdown
                if brand_filter != "All Brands":
                    sel_b = brand_filter.rsplit(" (", 1)[0].lower()
                    displayed_hist = displayed_hist[
                        displayed_hist["parent_asin"].apply(
                            lambda a: str(meta_lookup.get(str(a), {}).get("brand", "")).lower() == sel_b
                        )
                    ]

                # 3. Filter by Text Keyword Search (splits terms and matches all tokens)
                if hist_search.strip():
                    tokens = hist_search.strip().lower().split()
                    displayed_hist = displayed_hist[
                        displayed_hist["parent_asin"].apply(
                            lambda a: all(
                                tok in str(meta_lookup.get(str(a), {}).get("brand", "")).lower()
                                or tok in str(meta_lookup.get(str(a), {}).get("title", "")).lower()
                                or tok in str(a).lower()
                                for tok in tokens
                            )
                        )
                    ]

                st.caption(f"Showing **{len(displayed_hist)}** purchases from this user's history:")

                recent_rows = []
                for _, row in displayed_hist.iterrows():
                    m = meta_lookup.get(str(row["parent_asin"]), {})
                    price_val = m.get("price")
                    cats = m.get("categories", ())
                    cat_str = sorted(cats)[0].title() if cats else "Electronics"
                    b_str = m.get("brand") or "—"
                    recent_rows.append(
                        {
                            "Product ID": row["parent_asin"],
                            "Title": str(m.get("title", "Unknown"))[:80],
                            "Brand": b_str.title() if b_str != "—" else "—",
                            "Category": cat_str,
                            "User Rating": f"{row['rating']:.1f} ★",
                            "Price": f"${price_val:.2f}" if pd.notna(price_val) else "—",
                        }
                    )
                if recent_rows:
                    st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No purchases in this user's history match the selected filter.")

            with st.expander("🔎 Search Entire Catalog (66,320 items) & Test Live Scoring / SHAP", expanded=False):
                st.caption("Search for any brand or product across the full catalog to inspect its metadata, whether this user has bought it, and how the model scores and explains it.")
                cat_col1, cat_col2 = st.columns([3, 1])
                with cat_col1:
                    cat_query = st.text_input(
                        "Search Catalog by Brand, Title, or ASIN",
                        placeholder="e.g. Sony, Apple, Samsung Galaxy, Microphone...",
                        key="cat_search_input",
                    )
                with cat_col2:
                    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                    st.button("Search Catalog", key="cat_search_btn")

                if cat_query.strip():
                    q_toks = cat_query.strip().lower().split()
                    cat_matches = []
                    for asin, m in meta_lookup.items():
                        t_lower = str(m.get("title", "")).lower()
                        b_lower = str(m.get("brand", "")).lower()
                        if all(tok in t_lower or tok in b_lower or tok in asin.lower() for tok in q_toks):
                            cat_matches.append((asin, m))
                        if len(cat_matches) >= 5:
                            break

                    if not cat_matches:
                        st.warning(f"No catalog products matched '{cat_query}'. Try a broader term like 'headphone', 'cable', 'Samsung'.")
                    else:
                        st.markdown(f"**Found {len(cat_matches)} catalog matches:**")
                        user_asins_set = set(user_hist["parent_asin"].astype(str))

                        for asin, m in cat_matches:
                            b_str = m.get("brand", "Unknown")
                            t_str = m.get("title", "Unknown")
                            p_val = m.get("price")
                            p_txt = f"${p_val:.2f}" if pd.notna(p_val) else "Price N/A"
                            has_bought = asin in user_asins_set

                            with st.container(border=True):
                                c_left, c_right = st.columns([3, 1])
                                with c_left:
                                    st.markdown(f"**{t_str}**")
                                    st.caption(f"ID: `{asin}` | Brand: **{b_str.title()}** | {p_txt} | Status: {'✅ In User History' if has_bought else '🆕 Never Purchased by this User'}")
                                with c_right:
                                    if st.button("Score & Explain", key=f"score_cat_{asin}"):
                                        st.session_state[f"inspect_{asin}"] = True

                                if st.session_state.get(f"inspect_{asin}"):
                                    u_summary = hybrid.phase1.feature_extractor.build_user_summary(selected_user, user_hist)
                                    feats = hybrid.phase1.feature_extractor.extract_features(u_summary, asin)
                                    f_df = pd.DataFrame([feats])[FEATURE_NAMES]
                                    c_score = float(hybrid.phase1.ranker.predict_proba(f_df)[0]) if hybrid.phase1.ranker.is_fitted else float(feats.get("text_similarity", 0.0))
                                    cf_s = hybrid.cf_model.predict_score(selected_user, asin) if (hybrid.cf_model and hybrid.cf_model.is_fitted) else None
                                    eff_cf = cf_s if cf_s is not None else c_score
                                    final_sc = alpha * c_score + (1.0 - alpha) * eff_cf

                                    st.markdown(f"**Recommender Score: `{final_sc:.4f}`** (Content: `{c_score:.4f}`, CF: `{cf_s if cf_s is not None else 'N/A'}`)")
                                    ins_shap = hybrid.phase1.shap_explainer.explain_instance(feats) if hybrid.phase1.shap_explainer else {}
                                    ins_expl = hybrid.phase1.explanation_engine.get_plain_sentences(ins_shap, m, top_n=3)
                                    for sent in ins_expl:
                                        st.markdown(f"- *{sent}*")
                                    if ins_shap:
                                        st.plotly_chart(make_shap_chart(ins_shap, f"SHAP Contributions for {asin}"), use_container_width=True, key=f"insp_shap_{asin}")

        # Generate Recommendations
        if is_phase2:
            recs = hybrid.recommend(selected_user, k=top_k, alpha=alpha)
        else:
            recs = hybrid.phase1.recommend(selected_user, k=top_k)

        st.subheader(f"Top-{len(recs)} Explainable Recommendations")

        for rec in recs:
            price_txt = f"${rec['price']:.2f}" if rec.get("price") is not None else "Price N/A"
            brand_txt = rec.get("brand") or "Unknown Brand"
            cat_txt = rec.get("category") or "Electronics"

            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                with head_l:
                    st.markdown(f"### #{rec['rank']} · {rec['title']}")
                    st.caption(
                        f"ID: `{rec['product_id']}`  |  Brand: **{brand_txt}**  |  Category: **{cat_txt}**  |  {price_txt}"
                    )
                with head_r:
                    st.metric("Final Score", f"{rec['score']:.4f}")

                if is_phase2 and not is_cold:
                    c_sc = rec.get("content_score", rec["score"])
                    cf_sc = rec.get("cf_score")
                    if cf_sc is not None:
                        st.info(
                            f"**Hybrid Breakdown:** `Score = {alpha:.2f} × {c_sc:.4f} (Content XGBoost) + "
                            f"{1 - alpha:.2f} × {cf_sc:.4f} (SVD CF) = {rec['score']:.4f}`"
                        )
                    else:
                        st.info(
                            f"**Hybrid Breakdown (CF Cold-Start Fallback):** Item unseen in CF matrix -> "
                            f"using Content score `C = {c_sc:.4f}`"
                        )

                body_l, body_r = st.columns([1.05, 1.35])
                with body_l:
                    st.markdown("**Why this was recommended:**")
                    for sent in rec.get("explanation", []):
                        st.markdown(f"- {sent}")

                    feats = rec.get("features")
                    if feats and not is_cold:
                        with st.expander("Inspect Raw 6 Ranking Features"):
                            feat_df = pd.DataFrame(
                                [
                                    {"Feature": FEATURE_LABELS.get(f, f), "Value": round(float(feats.get(f, 0.0)), 4)}
                                    for f in FEATURE_NAMES
                                ]
                            )
                            st.dataframe(feat_df, use_container_width=True, hide_index=True)

                with body_r:
                    shap_vals = rec.get("shap_values", {})
                    if not is_cold and shap_vals and any(abs(v) > 0 for v in shap_vals.values()):
                        fig = make_shap_chart(shap_vals, f"SHAP Feature Contributions (Rank #{rec['rank']})")
                        st.plotly_chart(fig, use_container_width=True, key=f"shap_{rec['rank']}_{rec['product_id']}")
                    else:
                        st.caption("Cold-start popularity recommendation (no personalized SHAP profile used).")

    with tab_eval:
        st.subheader("Section 20 — Quantitative Evaluation & Global Explainability")
        report_data = p2_report or p1_report

        if report_data is None:
            st.warning("No evaluation report JSON found in `reports/`. Run `python -m src.evaluation.evaluator --phase 2`.")
        else:
            rows = []
            pop_m = report_data.get("popularity_baseline_metrics", {})
            if pop_m:
                rows.append({"Model": "Popularity Baseline", **{k: round(v, 4) for k, v in pop_m.items()}})

            p1_m = report_data.get("phase1_metrics") or report_data.get("overall_metrics", {})
            if p1_m:
                rows.append({"Model": "Phase 1 (Content + LOO XGBoost + SHAP)", **{k: round(v, 4) for k, v in p1_m.items()}})

            if p2_report and "phase2_alpha_sweep" in p2_report:
                for key in sorted(p2_report["phase2_alpha_sweep"].keys(), reverse=True):
                    entry = p2_report["phase2_alpha_sweep"][key]
                    rows.append(
                        {
                            "Model": f"Phase 2 Hybrid (α = {entry['alpha']:.1f})",
                            **{k: round(v, 4) for k, v in entry["overall_metrics"].items()},
                        }
                    )

            st.markdown("#### Offline Ranking Quality (Chronological Leave-One-Out, 500 Users)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            g_col, cs_col = st.columns([1.2, 1.0])
            with g_col:
                g_shap = report_data.get("global_shap_importance", {})
                if g_shap:
                    st.plotly_chart(make_global_shap_chart(g_shap), use_container_width=True)
            with cs_col:
                st.markdown("#### Cold-Start vs. Seen-in-Training (Phase 1)")
                cs = report_data.get("phase1_cold_start_breakdown") or report_data.get("cold_start_breakdown", {})
                if cs:
                    ga = cs.get("group_a_seen_in_train", {})
                    gb = cs.get("group_b_unseen_in_train", {})
                    cs_df = pd.DataFrame(
                        [
                            {
                                "Cohort": "Group A (Seen in Train)",
                                "Users": ga.get("count", 0),
                                "Recall@10": round(ga.get("metrics", {}).get("recall@10", 0.0), 4),
                                "NDCG@10": round(ga.get("metrics", {}).get("ndcg@10", 0.0), 4),
                            },
                            {
                                "Cohort": "Group B (Unseen Cold-Start)",
                                "Users": gb.get("count", 0),
                                "Recall@10": round(gb.get("metrics", {}).get("recall@10", 0.0), 4),
                                "NDCG@10": round(gb.get("metrics", {}).get("ndcg@10", 0.0), 4),
                            },
                        ]
                    )
                    st.dataframe(cs_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "55.6% of test items had zero prior training interactions. Content-Based Phase 1 still retrieves and explains these cold-start items, while Collaborative Filtering adds signal for seen items."
                    )


if __name__ == "__main__":
    main()

