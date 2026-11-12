"""
Section 6 — Dataset Subsetting & Large-File Ingestion

Streams the raw Electronics JSONL files line-by-line (never loading
the whole file into memory) and produces two small, self-contained
Parquet files that every later phase reads from.

Usage:
    python -m src.data.subset_data          # uses configs/config.yaml
    python -m src.data.subset_data --config path/to/config.yaml
"""

import argparse
import gzip
import json
import logging
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REVIEWS_GZ = RAW_DIR / "Electronics.jsonl.gz"
META_GZ = RAW_DIR / "meta_Electronics.jsonl.gz"

REVIEWS_OUT = PROCESSED_DIR / "reviews_subset.parquet"
META_OUT = PROCESSED_DIR / "metadata_subset.parquet"

# ── Review fields to keep (Section 4) ────────────────────────────────
REVIEW_COLS = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "verified_purchase",
]

# ── Metadata fields to keep (Section 4) ─────────────────────────────
META_COLS = [
    "parent_asin",
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "store",
    "categories",
]


# ── Helpers ──────────────────────────────────────────────────────────
def _stream_jsonl_gz(path: Path):
    """Yield parsed JSON objects from a gzipped JSONL file, one line at a time."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSON at line %d in %s", line_no, path.name)


def _is_valid_review(rec: dict) -> bool:
    """Check that the three required fields exist and are non-empty."""
    return bool(rec.get("user_id") and rec.get("parent_asin") and rec.get("rating") is not None)


def _pick_review_fields(rec: dict) -> dict:
    """Return only the columns we want, with safe defaults."""
    return {col: rec.get(col) for col in REVIEW_COLS}


def _pick_meta_fields(rec: dict) -> dict:
    """Return only the metadata columns we want."""
    return {col: rec.get(col) for col in META_COLS}


# ── Main pipeline ────────────────────────────────────────────────────
def run_subsetting(config_path: str | Path | None = None) -> tuple[Path, Path]:
    """Execute the full subsetting pipeline.  Returns paths to the two output files."""

    # ── Load config ──────────────────────────────────────────────────
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    min_ratings = cfg["dataset"]["min_ratings_per_user"]
    max_users = cfg["dataset"]["max_users"]

    log.info("Config: min_ratings_per_user=%d, max_users=%d", min_ratings, max_users)

    # ── Pass 1 — count interactions per user ─────────────────────────
    log.info("Pass 1: counting interactions per user from %s …", REVIEWS_GZ.name)
    user_counts: Counter = Counter()
    total_lines = 0
    skipped = 0

    for rec in _stream_jsonl_gz(REVIEWS_GZ):
        total_lines += 1
        if _is_valid_review(rec):
            user_counts[rec["user_id"]] += 1
        else:
            skipped += 1

    log.info(
        "Pass 1 done: %s total lines, %d valid users, %d skipped rows",
        f"{total_lines:,}",
        len(user_counts),
        skipped,
    )

    # ── Select qualifying users ──────────────────────────────────────
    qualifying = {
        uid: cnt
        for uid, cnt in user_counts.items()
        if cnt >= min_ratings
    }
    log.info(
        "%d users have >= %d interactions",
        len(qualifying),
        min_ratings,
    )

    # Sort by count descending (most active first) and take up to max_users
    selected_users = set(
        uid
        for uid, _ in sorted(qualifying.items(), key=lambda x: x[1], reverse=True)[:max_users]
    )
    log.info("Selected %d users (max_users=%d)", len(selected_users), max_users)

    # ── Pass 2 — collect reviews for selected users ──────────────────
    log.info("Pass 2: collecting reviews for selected users from %s …", REVIEWS_GZ.name)
    reviews: list[dict] = []
    parent_asins: set[str] = set()

    for rec in _stream_jsonl_gz(REVIEWS_GZ):
        if not _is_valid_review(rec):
            continue
        if rec["user_id"] in selected_users:
            reviews.append(_pick_review_fields(rec))
            parent_asins.add(rec["parent_asin"])

    log.info(
        "Pass 2 done: %s reviews retained, %s unique parent_asin values",
        f"{len(reviews):,}",
        f"{len(parent_asins):,}",
    )

    # ── Write reviews_subset.parquet ─────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reviews_df = pd.DataFrame(reviews)
    reviews_df.to_parquet(REVIEWS_OUT, index=False)
    log.info("Wrote %s (%s rows)", REVIEWS_OUT.relative_to(PROJECT_ROOT), f"{len(reviews_df):,}")

    # Free memory before the metadata pass
    del reviews, reviews_df

    # ── Pass 3 — collect metadata for those parent_asins ─────────────
    log.info("Pass 3: filtering metadata from %s …", META_GZ.name)
    meta_rows: list[dict] = []

    for rec in _stream_jsonl_gz(META_GZ):
        pasin = rec.get("parent_asin")
        if pasin and pasin in parent_asins:
            meta_rows.append(_pick_meta_fields(rec))

    meta_df = pd.DataFrame(meta_rows)

    # ── Clean mixed-type columns before Parquet serialization ────────
    # price: coerce non-numeric values (currency symbols, "N/A", etc.) to NaN
    meta_df["price"] = pd.to_numeric(meta_df["price"], errors="coerce")

    # List columns: serialize to JSON strings so Parquet doesn't choke
    # on ragged / mixed-type lists
    for col in ("features", "description", "categories"):
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].apply(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
            )

    meta_df.to_parquet(META_OUT, index=False)
    log.info("Wrote %s (%s rows)", META_OUT.relative_to(PROJECT_ROOT), f"{len(meta_df):,}")

    # ── Quick sanity summary ─────────────────────────────────────────
    log.info("── Sanity check ──")
    log.info("  Reviews:   %s rows, %d unique users", f"{len(meta_rows):,}", len(selected_users))
    log.info("  Metadata:  %s rows", f"{len(meta_rows):,}")
    log.info("  parent_asin coverage: %d in reviews, %d in metadata",
             len(parent_asins), len(meta_df["parent_asin"].unique()))

    return REVIEWS_OUT, META_OUT


# ── CLI entry point ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Section 6 — Dataset subsetting")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    args = parser.parse_args()
    run_subsetting(args.config)


if __name__ == "__main__":
    main()
