"""
Information Retrieval Ranking Metrics — Section 20 of the specification.

Provides deterministic calculations for:
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
"""

import math
from typing import Any, Sequence, Set


def precision_at_k(
    recommended: Sequence[str],
    ground_truth: Sequence[str] | Set[str],
    k: int = 10,
) -> float:
    """Calculate Precision@K: fraction of top-K recommendations that are relevant.

    Precision@K = |recommended[:K] ∩ ground_truth| / K
    """
    if k <= 0:
        return 0.0
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in gt_set)
    return float(hits / k)


def recall_at_k(
    recommended: Sequence[str],
    ground_truth: Sequence[str] | Set[str],
    k: int = 10,
) -> float:
    """Calculate Recall@K: fraction of ground-truth relevant items captured in top-K.

    Recall@K = |recommended[:K] ∩ ground_truth| / |ground_truth|

    For leave-one-out evaluation (|ground_truth| == 1), this equals the Hit-Rate@K:
    1.0 if the held-out item appears in recommended[:K], else 0.0.
    """
    if k <= 0:
        return 0.0
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in gt_set)
    return float(hits / len(gt_set))


def ndcg_at_k(
    recommended: Sequence[str],
    ground_truth: Sequence[str] | Set[str],
    k: int = 10,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank K.

    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, |ground_truth|)} 1 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K

    For leave-one-out evaluation (|ground_truth| == 1):
    - IDCG@K = 1 / log2(1 + 1) = 1.0
    - If relevant item is at rank r (1-indexed, r <= K): NDCG@K = 1 / log2(r + 1)
    - If not in top-K: NDCG@K = 0.0
    """
    if k <= 0:
        return 0.0
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    top_k = recommended[:k]

    # Compute DCG
    dcg = 0.0
    for idx, item in enumerate(top_k, 1):
        if item in gt_set:
            dcg += 1.0 / math.log2(idx + 1)

    # Compute Ideal DCG (all relevant items ranked at the very top)
    n_ideal = min(k, len(gt_set))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, n_ideal + 1))

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def evaluate_user_recommendations(
    recommended: Sequence[str],
    ground_truth: Sequence[str] | Set[str],
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Calculate all standard metrics for a single user across specified K values."""
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(recommended, ground_truth, k=k)
        metrics[f"recall@{k}"] = recall_at_k(recommended, ground_truth, k=k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(recommended, ground_truth, k=k)
    return metrics


def aggregate_metrics(
    user_metrics: Sequence[dict[str, float]]
) -> dict[str, float]:
    """Calculate mean metric values across a collection of evaluated users."""
    if not user_metrics:
        return {}

    keys = user_metrics[0].keys()
    n = len(user_metrics)
    return {
        key: float(sum(u[key] for u in user_metrics) / n)
        for key in keys
    }
