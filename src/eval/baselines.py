"""Popularity baseline: recommend the most-interacted-with training items.

This is the floor every learned model has to beat. Popularity is computed
from the *train* split only, and each user's already-seen items are
excluded from their recommendations.

Run standalone to print test-set metrics and a few sample recommendations:
    python -m src.eval.baselines
"""

import numpy as np
import pandas as pd


class PopularityRecommender:
    def __init__(self, interactions: pd.DataFrame):
        train = interactions[interactions["split"] == "train"]
        counts = train.groupby("item_idx")["user_idx"].size()
        # Items sorted by training interaction count, most popular first.
        self.ranked_items = counts.sort_values(ascending=False).index.to_numpy()

    def recommend(
        self, user_idx: int, k: int, exclude: np.ndarray | None = None
    ) -> np.ndarray:
        if exclude is None or len(exclude) == 0:
            return self.ranked_items[:k]
        mask = ~np.isin(self.ranked_items, exclude)
        return self.ranked_items[mask][:k]


def main() -> None:
    from src.config import EVAL_KS, TOP_K
    from src.dataio import (
        ground_truth_by_user,
        load_interactions,
        load_items,
        seen_items_by_user,
    )
    from src.eval.metrics import evaluate_rankings

    interactions = load_interactions()
    model = PopularityRecommender(interactions)
    seen = seen_items_by_user(interactions, splits=("train", "val"))
    truth = ground_truth_by_user(interactions, split="test")

    recs = {
        user: model.recommend(user, max(EVAL_KS), exclude=seen.get(user))
        for user in truth
    }
    results = evaluate_rankings(recs, truth, ks=EVAL_KS)
    print("Popularity baseline (test split):")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    # Show what the baseline actually recommends for a couple of users.
    items = load_items().set_index("item_idx")
    for user in list(truth)[:2]:
        titles = items.loc[recs[user][:TOP_K], "title"].tolist()
        print(f"\nUser {user} top-{TOP_K}: ")
        for t in titles:
            print(f"    {t}")


if __name__ == "__main__":
    main()
