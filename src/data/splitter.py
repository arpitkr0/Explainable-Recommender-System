"""
Train / Test Split — Section 8 of the specification.

Per-user chronological split holding out the most recent interaction(s) as the test set.
Enforces the hard implementation constraint: user profiles for test evaluation must only
be built from historical interactions prior to the held-out test interactions.
"""

import logging
from typing import Sequence

import pandas as pd

log = logging.getLogger(__name__)


def temporal_train_test_split(
    reviews_df: pd.DataFrame, holdout_k: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions per user chronologically.

    For each user, interactions are sorted by timestamp. The most recent
    `holdout_k` interactions are held out as the test set; all preceding
    interactions form the training set.

    Args:
        reviews_df: DataFrame with ['user_id', 'parent_asin', 'timestamp', ...]
        holdout_k: Number of most recent interactions per user to hold out (default 1).

    Returns:
        (train_df, test_df)
    """
    if reviews_df.empty:
        return reviews_df.copy(), reviews_df.copy()

    # Sort chronologically per user
    sorted_df = reviews_df.sort_values(by=["user_id", "timestamp"], ascending=[True, True]).copy()

    # Rank from the end: 0 is the most recent interaction for that user
    sorted_df["_rank_from_end"] = sorted_df.groupby("user_id").cumcount(ascending=False)

    is_test = sorted_df["_rank_from_end"] < holdout_k
    test_df = sorted_df[is_test].drop(columns=["_rank_from_end"]).reset_index(drop=True)
    train_df = sorted_df[~is_test].drop(columns=["_rank_from_end"]).reset_index(drop=True)

    log.info(
        "Temporal split complete (holdout_k=%d): Train=%d reviews, Test=%d reviews across %d users",
        holdout_k,
        len(train_df),
        len(test_df),
        train_df["user_id"].nunique(),
    )
    return train_df, test_df


def assert_no_temporal_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_profiles_items: dict[str, Sequence[str]] | None = None,
) -> None:
    """Validate that temporal leakage invariants are strictly respected.

    Invariants:
      1. For each user, max(train_timestamp) <= min(test_timestamp).
      2. If user_profiles_items is supplied, asserts that for every user,
         no test item parent_asin was used to build their profile.
    """
    # Check 1: Temporal ordering
    common_users = set(train_df["user_id"].unique()).intersection(set(test_df["user_id"].unique()))
    for uid in common_users:
        u_train = train_df[train_df["user_id"] == uid]
        u_test = test_df[test_df["user_id"] == uid]

        max_train_ts = u_train["timestamp"].max()
        min_test_ts = u_test["timestamp"].min()

        assert max_train_ts <= min_test_ts, (
            f"Temporal leakage detected for user {uid}: "
            f"max train timestamp ({max_train_ts}) > min test timestamp ({min_test_ts})"
        )

    # Check 2: Profile items exclusion
    if user_profiles_items is not None:
        for uid in common_users:
            profile_items = set(user_profiles_items.get(str(uid), []))
            test_items = set(test_df[test_df["user_id"] == uid]["parent_asin"].astype(str))
            leaked_items = profile_items.intersection(test_items)
            assert not leaked_items, (
                f"Data leakage detected for user {uid}: test items {leaked_items} "
                f"were found in the interactions used to build the user's profile!"
            )
