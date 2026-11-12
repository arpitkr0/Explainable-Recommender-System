"""
Explanation Engine & Data Structure — Section 14 of the specification.

Converts numeric SHAP contribution values into deterministic, template-based
plain-language sentences and structured explanation records.
"""

import logging
from typing import Any

log = logging.getLogger(__name__)

# Template mapping for positive SHAP contributions (Section 14 & 15)
POSITIVE_TEMPLATES = {
    "text_similarity": "Similar to products you previously liked",
    "category_similarity": "Matches your preferred category{category_detail}",
    "brand_match": "Strong match with {brand_detail} products in your history",
    "feature_similarity": "Similar product features",
    "price_similarity": "Fits within your typical price range",
    "rating_similarity": "Highly rated by users with similar standards",
}

NEGATIVE_TEMPLATES = {
    "text_similarity": "Different textual style from your past purchases",
    "category_similarity": "Explores a category outside your usual preferences",
    "brand_match": "From a brand new to your purchase history",
    "feature_similarity": "Features differ from your typical selections",
    "price_similarity": "Price differs from your usual spending range",
    "rating_similarity": "Rating variance compared to your standard preferences",
}


class ExplanationEngine:
    """Deterministic, template-based explanation generator (Section 14)."""

    def __init__(self, min_shap_threshold: float = 0.001):
        self.min_shap_threshold = min_shap_threshold

    def explain(
        self,
        shap_values: dict[str, float],
        product_metadata: dict[str, Any] | None = None,
        top_n: int = 4,
    ) -> list[dict[str, Any]]:
        """Generate structured explanation records from SHAP values and metadata.

        Returns a list of structured records matching Section 14:
        [
            {
                "feature": "brand_match",
                "shap_value": 0.24,
                "direction": "positive",
                "evidence": "Strong match with Sony products in your history"
            },
            ...
        ]
        """
        meta = product_metadata or {}
        brand_raw = meta.get("brand") or meta.get("store") or ""
        brand_detail = str(brand_raw).strip() if str(brand_raw).strip() else "brands"

        cat_raw = meta.get("main_category")
        if not cat_raw and "categories" in meta:
            cats = meta["categories"]
            if isinstance(cats, (list, tuple, set)) and len(cats) > 0:
                cat_raw = list(cats)[0]
        category_detail = f" ({cat_raw})" if cat_raw and str(cat_raw).strip() else ""

        # Format context parameters
        format_kwargs = {
            "brand_detail": brand_detail,
            "category_detail": category_detail,
        }

        # Sort features by SHAP value descending (top positive drivers first)
        sorted_features = sorted(
            shap_values.items(), key=lambda item: abs(item[1]), reverse=True
        )

        records: list[dict[str, Any]] = []
        for feat, val in sorted_features:
            if abs(val) < self.min_shap_threshold:
                continue

            direction = "positive" if val >= 0 else "negative"
            if direction == "positive":
                template = POSITIVE_TEMPLATES.get(
                    feat, f"Positive contribution from {feat}"
                )
            else:
                template = NEGATIVE_TEMPLATES.get(
                    feat, f"Lower contribution from {feat}"
                )

            try:
                evidence = template.format(**format_kwargs)
            except Exception:
                evidence = template

            records.append(
                {
                    "feature": feat,
                    "shap_value": round(float(val), 4),
                    "direction": direction,
                    "evidence": evidence,
                }
            )
            if len(records) >= top_n:
                break

        # Fallback if all SHAP values are near zero
        if not records:
            records.append(
                {
                    "feature": "content_match",
                    "shap_value": 0.0,
                    "direction": "positive",
                    "evidence": "Recommended based on overall catalog popularity and relevance",
                }
            )

        return records

    def get_plain_sentences(
        self,
        shap_values: dict[str, float],
        product_metadata: dict[str, Any] | None = None,
        top_n: int = 4,
    ) -> list[str]:
        """Convenience method returning just the plain-language explanation strings."""
        records = self.explain(shap_values, product_metadata, top_n=top_n)
        return [r["evidence"] for r in records]
