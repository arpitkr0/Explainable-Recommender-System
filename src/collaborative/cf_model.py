"""
Collaborative Filtering Model — Section 16 of the specification.

Implements SVD-based Matrix Factorization over the user-item ratings matrix
built strictly from training interactions (zero test leakage).

Predicts normalized collaborative scores in [0.0, 1.0] for (user, item) pairs
and supports top-N collaborative candidate retrieval for hybrid blending.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

log = logging.getLogger(__name__)


class CollaborativeFilter:
    """SVD-based Matrix Factorization Collaborative Filtering model (Section 16)."""

    def __init__(self, n_factors: int = 32, random_state: int = 42):
        self.n_factors = int(n_factors)
        self.random_state = int(random_state)

        self.user_to_idx: dict[str, int] = {}
        self.idx_to_user: dict[int, str] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_item: dict[int, str] = {}

        self.global_mean: float = 4.0
        self.user_means: np.ndarray = np.array([], dtype=np.float32)
        self.item_biases: np.ndarray = np.array([], dtype=np.float32)
        self.user_factors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.item_factors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.user_interacted_indices: dict[int, set[int]] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "CollaborativeFilter":
        """Fit SVD matrix factorization on user-item training ratings."""
        if train_df.empty:
            log.warning("Empty train_df passed to CollaborativeFilter.fit")
            self.is_fitted = False
            return self

        df = train_df[["user_id", "parent_asin", "rating"]].dropna().copy()
        df["user_id"] = df["user_id"].astype(str)
        df["parent_asin"] = df["parent_asin"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])

        # Deduplicate multiple ratings for same (user, item) by taking the latest/mean
        df = df.groupby(["user_id", "parent_asin"], as_index=False)["rating"].mean()

        unique_users = sorted(df["user_id"].unique().tolist())
        unique_items = sorted(df["parent_asin"].unique().tolist())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        self.item_to_idx = {it: j for j, it in enumerate(unique_items)}
        self.idx_to_item = {j: it for j, it in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)
        self.global_mean = float(df["rating"].mean())

        u_idx = df["user_id"].map(self.user_to_idx).values
        i_idx = df["parent_asin"].map(self.item_to_idx).values
        ratings = df["rating"].values.astype(np.float64)

        # Build sparse rating matrix & compute user means
        R = csr_matrix((ratings, (u_idx, i_idx)), shape=(n_users, n_items))

        user_counts = np.diff(R.indptr)
        user_sums = np.asarray(R.sum(axis=1)).ravel()
        self.user_means = np.where(
            user_counts > 0, user_sums / np.maximum(user_counts, 1), self.global_mean
        ).astype(np.float32)

        # Track seen items per user for fast filtering
        self.user_interacted_indices = {}
        for u_i in range(n_users):
            start, end = R.indptr[u_i], R.indptr[u_i + 1]
            self.user_interacted_indices[u_i] = set(R.indices[start:end].tolist())

        # Compute damped item biases from rating residuals
        residuals = ratings - self.user_means[u_idx]
        item_res_sum = np.bincount(i_idx, weights=residuals, minlength=n_items)
        item_counts = np.bincount(i_idx, minlength=n_items)
        # Shrinkage damping (regularization=5.0) prevents 1-review items from extreme biases
        self.item_biases = (item_res_sum / (item_counts + 5.0)).astype(np.float32)

        # Residual matrix centered by user mean + item bias
        svd_residuals = residuals - self.item_biases[i_idx]
        R_centered = csr_matrix((svd_residuals, (u_idx, i_idx)), shape=(n_users, n_items))

        # Truncated SVD (svds requires k < min(n_users, n_items))
        max_k = min(n_users, n_items) - 1
        k = min(self.n_factors, max_k)
        if k >= 1:
            U, sigma, Vt = svds(R_centered, k=k, random_state=self.random_state)
            # svds returns singular values in ascending order; reverse to descending
            order = np.argsort(-sigma)
            U = U[:, order]
            sigma = sigma[order]
            Vt = Vt[order, :]

            # Absorb sqrt(sigma) symmetrically into user and item latent factors
            sqrt_s = np.sqrt(np.maximum(sigma, 0.0))
            self.user_factors = (U * sqrt_s[np.newaxis, :]).astype(np.float32)
            self.item_factors = (Vt.T * sqrt_s[np.newaxis, :]).astype(np.float32)
        else:
            self.user_factors = np.zeros((n_users, 0), dtype=np.float32)
            self.item_factors = np.zeros((n_items, 0), dtype=np.float32)

        self.is_fitted = True
        log.info(
            "CollaborativeFilter fitted: %d users, %d items, %d latent factors (density=%.4f%%)",
            n_users,
            n_items,
            k if k >= 1 else 0,
            len(df) / max(1, n_users * n_items) * 100.0,
        )
        return self

    def has_user(self, user_id: str) -> bool:
        return str(user_id) in self.user_to_idx

    def has_item(self, parent_asin: str) -> bool:
        return str(parent_asin) in self.item_to_idx

    def predict_rating(self, user_id: str, parent_asin: str) -> float | None:
        """Predict raw rating in [1.0, 5.0], or None if user/item unseen in CF."""
        if not self.is_fitted:
            return None
        u_i = self.user_to_idx.get(str(user_id))
        i_j = self.item_to_idx.get(str(parent_asin))
        if u_i is None or i_j is None:
            return None

        pred = float(self.user_means[u_i] + self.item_biases[i_j])
        if self.user_factors.shape[1] > 0:
            pred += float(np.dot(self.user_factors[u_i], self.item_factors[i_j]))
        return float(np.clip(pred, 1.0, 5.0))

    def predict_score(self, user_id: str, parent_asin: str) -> float | None:
        """Predict normalized CF score in [0.0, 1.0], or None if user/item unseen."""
        rating = self.predict_rating(user_id, parent_asin)
        if rating is None:
            return None
        return float((rating - 1.0) / 4.0)

    def predict_batch(self, user_id: str, parent_asins: list[str]) -> dict[str, float | None]:
        """Predict normalized CF scores in [0.0, 1.0] for a list of candidate items."""
        if not self.is_fitted:
            return {str(a): None for a in parent_asins}
        u_i = self.user_to_idx.get(str(user_id))
        if u_i is None:
            return {str(a): None for a in parent_asins}

        u_mean = float(self.user_means[u_i])
        u_vec = self.user_factors[u_i] if self.user_factors.shape[1] > 0 else None

        results: dict[str, float | None] = {}
        for asin in parent_asins:
            a_str = str(asin)
            i_j = self.item_to_idx.get(a_str)
            if i_j is None:
                results[a_str] = None
                continue
            pred = u_mean + float(self.item_biases[i_j])
            if u_vec is not None:
                pred += float(np.dot(u_vec, self.item_factors[i_j]))
            pred = float(np.clip(pred, 1.0, 5.0))
            results[a_str] = float((pred - 1.0) / 4.0)
        return results

    def recommend_candidates(
        self,
        user_id: str,
        user_interacted_asins: set[str] | None = None,
        top_n: int = 50,
    ) -> list[tuple[str, float]]:
        """Retrieve top-N unseen items for a user ordered by predicted CF score."""
        if not self.is_fitted:
            return []
        u_i = self.user_to_idx.get(str(user_id))
        if u_i is None:
            return []

        preds = self.user_means[u_i] + self.item_biases.copy()
        if self.user_factors.shape[1] > 0:
            preds = preds + self.item_factors.dot(self.user_factors[u_i])

        # Mask out items user interacted with
        seen_indices = set(self.user_interacted_indices.get(u_i, set()))
        if user_interacted_asins:
            for a in user_interacted_asins:
                j = self.item_to_idx.get(str(a))
                if j is not None:
                    seen_indices.add(j)

        if seen_indices:
            preds[list(seen_indices)] = -np.inf

        k = min(top_n, len(preds) - len(seen_indices))
        if k <= 0:
            return []

        top_idx = np.argpartition(preds, -k)[-k:]
        top_idx = top_idx[np.argsort(-preds[top_idx])]

        out: list[tuple[str, float]] = []
        for idx in top_idx:
            val = float(preds[idx])
            if not np.isfinite(val):
                continue
            clipped = float(np.clip(val, 1.0, 5.0))
            norm_score = float((clipped - 1.0) / 4.0)
            out.append((self.idx_to_item[int(idx)], norm_score))
        return out

    def save(self, file_path: str | Path) -> None:
        """Serialize fitted CF model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        log.info("CollaborativeFilter saved to %s", path)

    @classmethod
    def load(cls, file_path: str | Path) -> "CollaborativeFilter":
        """Load serialized CollaborativeFilter from disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.__dict__.update(state)
        log.info("CollaborativeFilter loaded from %s", path)
        return obj
