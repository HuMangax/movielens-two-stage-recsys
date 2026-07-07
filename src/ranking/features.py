"""Feature assembly for the XGBoost ranker.

One row per (user, candidate item). Everything derives from the train
split (via the precomputed feature tables) plus the stage-1 retrieval
output, so the same code path serves training, offline eval, and the API.
"""

import numpy as np
import pandas as pd

from src.config import ITEM_FEATURES_PATH, USER_FEATURES_PATH

FEATURE_COLUMNS = [
    "stage1_score",   # two-tower dot product — the retrieval model's opinion
    "stage1_rank",    # position in the candidate list (0 = best)
    "log_pop",        # item popularity in train (log1p of interaction count)
    "item_mean_rating",
    "item_days_since_last",  # days since the item's last train interaction
    "year",           # release year
    "user_count",     # user's train interaction count
    "user_mean_rating",
    "genre_match",    # dot(user genre preference distribution, item genre one-hot)
    "year_diff_abs",  # |item year - mean year of user's liked movies|
]


class RankerFeatureBuilder:
    def __init__(self) -> None:
        item_df = pd.read_parquet(ITEM_FEATURES_PATH).sort_values("item_idx")
        user_df = pd.read_parquet(USER_FEATURES_PATH).sort_values("user_idx")
        genre_cols = [c for c in item_df.columns if c.startswith("genre_")]

        # Dense per-item arrays indexed by item_idx.
        self.item_log_pop = item_df["log_pop"].to_numpy(np.float32)
        self.item_mean_rating = item_df["item_mean_rating"].to_numpy(np.float32)
        self.item_year = item_df["year"].to_numpy(np.float32)
        self.item_genres = item_df[genre_cols].to_numpy(np.float32)
        item_last_ts = item_df["item_last_ts"].to_numpy(np.float64)
        ref_ts = item_last_ts.max()
        self.item_days_since = ((ref_ts - item_last_ts) / 86400.0).astype(np.float32)

        # Per-user arrays indexed by user_idx. user_features covers every
        # user in interactions (all have train rows), so a dense array works.
        n_users = int(user_df["user_idx"].max()) + 1
        self.user_count = np.zeros(n_users, np.float32)
        self.user_mean_rating = np.zeros(n_users, np.float32)
        self.user_mean_year = np.zeros(n_users, np.float32)
        self.user_genres = np.zeros((n_users, len(genre_cols)), np.float32)
        idx = user_df["user_idx"].to_numpy()
        self.user_count[idx] = user_df["user_count"].to_numpy(np.float32)
        self.user_mean_rating[idx] = user_df["user_mean_rating"].to_numpy(np.float32)
        self.user_mean_year[idx] = user_df["user_mean_year"].to_numpy(np.float32)
        self.user_genres[idx] = user_df[[c for c in user_df.columns if c.startswith("genre_")]].to_numpy(np.float32)

    def build(
        self, users: np.ndarray, candidates: np.ndarray, stage1_scores: np.ndarray
    ) -> np.ndarray:
        """Feature matrix for flattened (user, candidate) pairs.

        ``users``: [n] user_idx per row; ``candidates``: [n] item_idx per row;
        ``stage1_scores``: [n] retrieval scores. ``stage1_rank`` is inferred
        from the order rows arrive in per user block (callers pass candidates
        already sorted by descending stage-1 score, as retrieve_candidates
        returns them).
        """
        n = len(users)
        # Rank within each contiguous user block.
        rank = np.zeros(n, np.float32)
        if n:
            new_user = np.r_[True, users[1:] != users[:-1]]
            block_starts = np.flatnonzero(new_user)
            rank = (np.arange(n) - np.repeat(block_starts, np.diff(np.r_[block_starts, n]))).astype(np.float32)

        genre_match = np.einsum(
            "ij,ij->i", self.user_genres[users], self.item_genres[candidates]
        )
        year_diff = np.abs(self.item_year[candidates] - self.user_mean_year[users])

        return np.column_stack(
            [
                stage1_scores.astype(np.float32),
                rank,
                self.item_log_pop[candidates],
                self.item_mean_rating[candidates],
                self.item_days_since[candidates],
                self.item_year[candidates],
                self.user_count[users],
                self.user_mean_rating[users],
                genre_match,
                year_diff,
            ]
        )
