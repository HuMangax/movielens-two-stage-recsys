"""Shared helpers for loading preprocessed artifacts."""

import numpy as np
import pandas as pd

from src.config import INTERACTIONS_PATH, ITEMS_PATH, USERS_PATH


def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(INTERACTIONS_PATH)


def load_items() -> pd.DataFrame:
    return pd.read_parquet(ITEMS_PATH)


def load_users() -> pd.DataFrame:
    return pd.read_parquet(USERS_PATH)


def seen_items_by_user(
    interactions: pd.DataFrame, splits: tuple[str, ...] = ("train", "val")
) -> dict[int, np.ndarray]:
    """Items each user has already interacted with in the given splits.

    Used to exclude already-seen items from recommendations — recommending
    something the user has already watched is a wasted slot.
    """
    subset = interactions[interactions["split"].isin(splits)]
    return {
        user: group.to_numpy()
        for user, group in subset.groupby("user_idx")["item_idx"]
    }


def ground_truth_by_user(
    interactions: pd.DataFrame, split: str = "test"
) -> dict[int, np.ndarray]:
    """Held-out positive items per user for the given split."""
    subset = interactions[interactions["split"] == split]
    return {
        user: group.to_numpy()
        for user, group in subset.groupby("user_idx")["item_idx"]
    }
