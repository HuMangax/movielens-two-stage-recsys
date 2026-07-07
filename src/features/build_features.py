"""Build user and item feature tables from the training split.

All statistics are computed from the *train* split only, so the ranker
never sees information from the validation or test periods (no leakage).

Item features:
- pop_count / log_pop: training interaction count (raw and log1p)
- item_mean_rating: mean explicit rating among training positives (4-5 range)
- item_last_ts: timestamp of the item's most recent training interaction
- year: release year parsed from the title
- genre_*: one-hot of the 18 MovieLens genres

User features:
- user_count / user_mean_rating / user_last_ts: activity stats
- user_mean_year: mean release year of movies the user liked
- genre_*: the user's genre preference distribution (fractions summing to 1)

Run: python -m src.features.build_features
"""

import numpy as np
import pandas as pd

from src.config import ITEM_FEATURES_PATH, USER_FEATURES_PATH
from src.dataio import load_interactions, load_items

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def genre_matrix(items: pd.DataFrame) -> pd.DataFrame:
    """One-hot genre columns aligned to items' row order."""
    cols = {}
    for genre in GENRES:
        col = "genre_" + genre.lower().replace("'", "").replace("-", "_")
        cols[col] = items["genres"].str.contains(genre, regex=False).astype(np.int8)
    return pd.DataFrame(cols, index=items.index)


def build_item_features(train: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    stats = train.groupby("item_idx").agg(
        pop_count=("user_idx", "size"),
        item_mean_rating=("rating", "mean"),
        item_last_ts=("timestamp", "max"),
    )
    feats = items.set_index("item_idx").join(stats)
    # Items with no training interactions exist (val/test-only): zero counts.
    feats["pop_count"] = feats["pop_count"].fillna(0).astype(np.int32)
    feats["log_pop"] = np.log1p(feats["pop_count"])
    feats["item_mean_rating"] = feats["item_mean_rating"].fillna(0.0)
    feats["item_last_ts"] = feats["item_last_ts"].fillna(0).astype(np.int64)
    feats["year"] = feats["year"].astype("float").fillna(feats["year"].astype("float").median())

    genre_cols = genre_matrix(feats)
    feats = pd.concat([feats.drop(columns=["movie_id", "title", "genres"]), genre_cols], axis=1)
    return feats.reset_index()


def build_user_features(train: pd.DataFrame, item_feats: pd.DataFrame) -> pd.DataFrame:
    stats = train.groupby("user_idx").agg(
        user_count=("item_idx", "size"),
        user_mean_rating=("rating", "mean"),
        user_last_ts=("timestamp", "max"),
    )

    genre_cols = [c for c in item_feats.columns if c.startswith("genre_")]
    per_item = item_feats.set_index("item_idx")[genre_cols + ["year"]]
    joined = train[["user_idx", "item_idx"]].join(per_item, on="item_idx")

    genre_sums = joined.groupby("user_idx")[genre_cols].sum()
    # Normalize to a preference distribution so heavy and light users compare.
    totals = genre_sums.sum(axis=1).replace(0, 1)
    genre_prefs = genre_sums.div(totals, axis=0)
    mean_year = joined.groupby("user_idx")["year"].mean().rename("user_mean_year")

    feats = stats.join(genre_prefs).join(mean_year)
    return feats.reset_index()


def main() -> None:
    interactions = load_interactions()
    items = load_items()
    train = interactions[interactions["split"] == "train"]

    item_feats = build_item_features(train, items)
    user_feats = build_user_features(train, item_feats)

    item_feats.to_parquet(ITEM_FEATURES_PATH, index=False)
    user_feats.to_parquet(USER_FEATURES_PATH, index=False)
    print(f"Wrote {ITEM_FEATURES_PATH.name}: {item_feats.shape}")
    print(f"Wrote {USER_FEATURES_PATH.name}: {user_feats.shape}")


if __name__ == "__main__":
    main()
