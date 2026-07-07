"""Preprocess MovieLens 1M into model-ready artifacts.

Steps:
1. Parse the raw ``.dat`` files (``::``-separated, latin-1 encoded).
2. Convert explicit ratings to implicit feedback: rating >= 4 is a positive.
3. Re-index users and items to contiguous integer ids.
4. Split each user's positives chronologically: oldest 80% train,
   next 10% validation, newest 10% test (minimum one interaction each).
   A temporal split avoids the leakage a random split would introduce —
   the model never trains on interactions that happened after the ones
   it is evaluated on for a given user.

Outputs (under artifacts/):
- interactions.parquet: user_idx, item_idx, rating, timestamp, split
- items.parquet: item_idx, movie_id, title, year, genres
- users.parquet: user_idx, user_id, gender, age, occupation

Usage: python data/preprocess.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    ARTIFACTS_DIR,
    INTERACTIONS_PATH,
    ITEMS_PATH,
    MIN_POSITIVES_PER_USER,
    POSITIVE_THRESHOLD,
    RAW_DIR,
    TEST_FRACTION,
    USERS_PATH,
    VAL_FRACTION,
)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        RAW_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        RAW_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        RAW_DIR / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    return ratings, movies, users


def temporal_split(positives: pd.DataFrame) -> pd.DataFrame:
    """Assign each positive interaction to train/val/test, per user, by time."""
    # Sort by (user, time); tie-break on movie_id for determinism.
    positives = positives.sort_values(
        ["user_idx", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)

    counts = positives.groupby("user_idx", sort=True)["item_idx"].transform("size")
    # Rank of each interaction within its user's history (0 = oldest).
    rank = positives.groupby("user_idx", sort=True).cumcount()

    n_test = np.maximum(1, (counts * TEST_FRACTION).astype(int))
    n_val = np.maximum(1, (counts * VAL_FRACTION).astype(int))
    split = np.where(
        rank >= counts - n_test,
        "test",
        np.where(rank >= counts - n_test - n_val, "val", "train"),
    )
    positives["split"] = split
    return positives


def main() -> None:
    ratings, movies, users = load_raw()
    print(f"Loaded {len(ratings):,} ratings, {len(movies):,} movies, {len(users):,} users")

    positives = ratings[ratings["rating"] >= POSITIVE_THRESHOLD].copy()
    print(f"Positives (rating >= {POSITIVE_THRESHOLD}): {len(positives):,}")

    # Keep users with enough positives to give each split at least one item.
    per_user = positives.groupby("user_id")["movie_id"].size()
    keep_users = per_user[per_user >= MIN_POSITIVES_PER_USER].index
    dropped = positives["user_id"].nunique() - len(keep_users)
    positives = positives[positives["user_id"].isin(keep_users)]
    print(f"Kept {len(keep_users):,} users (dropped {dropped} with < {MIN_POSITIVES_PER_USER} positives)")

    # Contiguous indices for embedding tables.
    user_ids = np.sort(positives["user_id"].unique())
    movie_ids = np.sort(positives["movie_id"].unique())
    user_to_idx = pd.Series(np.arange(len(user_ids)), index=user_ids)
    movie_to_idx = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    positives["user_idx"] = positives["user_id"].map(user_to_idx)
    positives["item_idx"] = positives["movie_id"].map(movie_to_idx)

    positives = temporal_split(positives)
    print(positives["split"].value_counts().to_string())

    # --- Item table (only movies that appear in the positives) -------------
    items = movies[movies["movie_id"].isin(movie_ids)].copy()
    items["item_idx"] = items["movie_id"].map(movie_to_idx)
    # Titles end with "(YYYY)" — extract the release year as a feature.
    items["year"] = (
        items["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float").astype("Int64")
    )
    items = items.sort_values("item_idx")[
        ["item_idx", "movie_id", "title", "year", "genres"]
    ]

    # --- User table ---------------------------------------------------------
    users = users[users["user_id"].isin(user_ids)].copy()
    users["user_idx"] = users["user_id"].map(user_to_idx)
    users = users.sort_values("user_idx")[
        ["user_idx", "user_id", "gender", "age", "occupation"]
    ]

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out = positives[
        ["user_idx", "item_idx", "user_id", "movie_id", "rating", "timestamp", "split"]
    ].reset_index(drop=True)
    out.to_parquet(INTERACTIONS_PATH, index=False)
    items.to_parquet(ITEMS_PATH, index=False)
    users.to_parquet(USERS_PATH, index=False)
    print(
        f"Wrote {INTERACTIONS_PATH.name} ({len(out):,} rows), "
        f"{ITEMS_PATH.name} ({len(items):,}), {USERS_PATH.name} ({len(users):,})"
    )


if __name__ == "__main__":
    main()
