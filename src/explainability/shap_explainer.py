"""
SHAP Explainability — Section 13 of the specification.

Computes local feature contributions (SHAP values) for individual recommendations
using shap.TreeExplainer on the XGBoost ranking model, as well as global feature
importance across the recommendation set.
"""

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap

from src.ranking.features import FEATURE_NAMES

log = logging.getLogger(__name__)


class ShapExplainer:
    """Explains XGBoost ranking predictions via SHAP TreeExplainer."""

    def __init__(self, xgb_model: Any):
        """Initialize TreeExplainer from fitted XGBoost model or XGBoostRanker."""
        if hasattr(xgb_model, "model"):
            self.model = xgb_model.model
        else:
            self.model = xgb_model

        log.info("Initializing SHAP TreeExplainer on XGBoost model")
        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, features_dict: dict[str, float]) -> dict[str, float]:
        """Compute SHAP contribution values for a single candidate's feature dictionary."""
        df = pd.DataFrame([features_dict])[FEATURE_NAMES]
        shap_values = self.explainer.shap_values(df)

        # Handle different SHAP output formats (array, list for multiclass, etc.)
        if isinstance(shap_values, list):
            # Binary classification: index 1 corresponds to positive class
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif hasattr(shap_values, "values"):
            vals = shap_values.values[0]
        else:
            vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        contributions: dict[str, float] = {}
        for feat, val in zip(FEATURE_NAMES, vals):
            contributions[feat] = round(float(val), 4)

        return contributions

    def explain_batch(self, feature_df: pd.DataFrame) -> list[dict[str, float]]:
        """Compute SHAP contributions for a DataFrame of candidate feature vectors."""
        if feature_df.empty:
            return []

        df = feature_df[FEATURE_NAMES]
        shap_values = self.explainer.shap_values(df)

        if isinstance(shap_values, list):
            vals_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            vals_matrix = shap_values.values
        else:
            vals_matrix = shap_values

        results: list[dict[str, float]] = []
        for row_vals in vals_matrix:
            row_dict = {}
            for feat, val in zip(FEATURE_NAMES, row_vals):
                row_dict[feat] = round(float(val), 4)
            results.append(row_dict)

        return results

    def get_global_feature_importance(self, feature_df: pd.DataFrame) -> dict[str, float]:
        """Compute mean absolute SHAP values across recommendations for global reporting."""
        if feature_df.empty:
            return {f: 0.0 for f in FEATURE_NAMES}

        df = feature_df[FEATURE_NAMES]
        shap_values = self.explainer.shap_values(df)

        if isinstance(shap_values, list):
            vals_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            vals_matrix = shap_values.values
        else:
            vals_matrix = shap_values

        mean_abs_shap = np.mean(np.abs(vals_matrix), axis=0)
        importance = {
            feat: round(float(val), 4)
            for feat, val in sorted(zip(FEATURE_NAMES, mean_abs_shap), key=lambda x: x[1], reverse=True)
        }
        return importance
