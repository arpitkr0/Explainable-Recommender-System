"""
Product Content Representation — Section 9 of the specification.

Concatenates title, features, description, categories, and brand (store)
into a single combined text representation per product, and vectorizes
via TF-IDF into product vectors (V_i).
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# ── Text Cleaning ────────────────────────────────────────────────────
# Standard regex to strip punctuation while preserving word boundaries
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalize whitespace."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Replace punctuation with a space so words don't merge
    text = PUNCT_REGEX.sub(" ", text)
    # Collapse multiple whitespaces
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def _extract_field_text(val: Any) -> str:
    """Extract and join text whether val is a list, string, or null."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple, np.ndarray)):
        # Join non-empty elements
        elements = [str(item).strip() for item in val if item is not None and str(item).strip()]
        return " ".join(elements)
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("none", "nan", "[]", "null"):
        return ""
    return s


def build_combined_text(metadata_df: pd.DataFrame) -> pd.Series:
    """Concatenate text fields for each product according to Section 9:

    combined_text = title + " " + features + " " + description + " " + categories + " " + brand
    """
    # Identify brand column ('store', 'brand', or empty)
    brand_col = None
    for candidate in ("store", "brand"):
        if candidate in metadata_df.columns:
            brand_col = candidate
            break

    def _combine_row(row: pd.Series) -> str:
        title = _extract_field_text(row.get("title", ""))
        features = _extract_field_text(row.get("features", ""))
        description = _extract_field_text(row.get("description", ""))
        categories = _extract_field_text(row.get("categories", ""))
        brand = _extract_field_text(row.get(brand_col, "")) if brand_col else ""

        parts = [p for p in [title, features, description, categories, brand] if p]
        combined = " ".join(parts)
        return clean_text(combined)

    combined_series = metadata_df.apply(_combine_row, axis=1)
    if "parent_asin" in metadata_df.columns:
        combined_series.index = metadata_df["parent_asin"].astype(str)
    return combined_series


# ── TF-IDF Vectorizer Wrapper ─────────────────────────────────────────
class ContentVectorizer:
    """Fits and manages TF-IDF product content vectors (V_i)."""

    def __init__(
        self,
        max_features: int | None = 10000,
        min_df: int | float = 2,
        ngram_range: tuple[int, int] = (1, 1),
        stop_words: str | None = "english",
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
        )
        self.tfidf_matrix: csr_matrix | None = None
        self.parent_asins: list[str] = []
        self._asin_to_idx: dict[str, int] = {}
        self._idx_to_asin: dict[int, str] = {}

    def fit_transform(
        self, metadata_df: pd.DataFrame, combined_texts: pd.Series | None = None
    ) -> csr_matrix:
        """Fit vectorizer on metadata combined text and return TF-IDF sparse matrix."""
        if combined_texts is None:
            log.info("Building combined text representation from %d metadata rows", len(metadata_df))
            combined_texts = build_combined_text(metadata_df)

        log.info("Fitting TF-IDF vectorizer (max_features=%s, min_df=%s)", self.max_features, self.min_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_texts).astype(np.float32)

        # Store IDs and mappings
        if "parent_asin" in metadata_df.columns:
            self.parent_asins = [str(x) for x in metadata_df["parent_asin"].tolist()]
        else:
            self.parent_asins = [str(x) for x in combined_texts.index.tolist()]

        self._asin_to_idx = {asin: idx for idx, asin in enumerate(self.parent_asins)}
        self._idx_to_asin = {idx: asin for idx, asin in enumerate(self.parent_asins)}

        log.info(
            "TF-IDF matrix built: shape=%s, vocab_size=%d",
            self.tfidf_matrix.shape,
            len(self.vectorizer.vocabulary_),
        )
        return self.tfidf_matrix

    def transform(self, texts: Sequence[str]) -> csr_matrix:
        """Transform arbitrary raw or cleaned texts into TF-IDF vectors."""
        if self.tfidf_matrix is None:
            raise RuntimeError("ContentVectorizer has not been fitted yet.")
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def get_item_vector(self, parent_asin: str) -> csr_matrix | None:
        """Retrieve the TF-IDF row vector for a specific product ID."""
        asin_str = str(parent_asin)
        idx = self._asin_to_idx.get(asin_str)
        if idx is None or self.tfidf_matrix is None:
            return None
        return self.tfidf_matrix[idx]

    def get_item_vectors(self, parent_asins: Sequence[str]) -> tuple[csr_matrix, list[str]]:
        """Retrieve vectors for multiple products; returns (matrix, found_asins)."""
        if self.tfidf_matrix is None:
            raise RuntimeError("ContentVectorizer has not been fitted yet.")
        indices = []
        found = []
        for asin in parent_asins:
            asin_str = str(asin)
            if asin_str in self._asin_to_idx:
                indices.append(self._asin_to_idx[asin_str])
                found.append(asin_str)
        if not indices:
            # Return empty matrix with same number of features
            empty = csr_matrix((0, self.tfidf_matrix.shape[1]), dtype=np.float32)
            return empty, []
        return self.tfidf_matrix[indices], found

    @property
    def num_items(self) -> int:
        return len(self.parent_asins)

    @property
    def num_features(self) -> int:
        if self.tfidf_matrix is None:
            return 0
        return self.tfidf_matrix.shape[1]

    def has_item(self, parent_asin: str) -> bool:
        return str(parent_asin) in self._asin_to_idx

    def get_index(self, parent_asin: str) -> int | None:
        return self._asin_to_idx.get(str(parent_asin))

    def get_parent_asin(self, index: int) -> str | None:
        return self._idx_to_asin.get(index)

    def save(self, file_path: str | Path) -> None:
        """Serialize fitted vectorizer and index state to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "parent_asins": self.parent_asins,
                    "asin_to_idx": self._asin_to_idx,
                    "idx_to_asin": self._idx_to_asin,
                },
                f,
            )
        log.info("ContentVectorizer saved to %s", path)

    @classmethod
    def load(cls, file_path: str | Path) -> "ContentVectorizer":
        """Load a serialized ContentVectorizer from disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.tfidf_matrix = data["tfidf_matrix"]
        obj.parent_asins = data["parent_asins"]
        obj._asin_to_idx = data["asin_to_idx"]
        obj._idx_to_asin = data["idx_to_asin"]
        log.info("ContentVectorizer loaded from %s (%d items)", path, len(obj.parent_asins))
        return obj
