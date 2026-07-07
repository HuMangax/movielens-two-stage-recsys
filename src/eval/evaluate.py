"""Final three-way comparison on the held-out test split.

Configurations:
1. popularity  — most popular training items the user hasn't seen
2. retrieval   — stage 1 alone: top-K by two-tower dot product
3. two_stage   — stage 1 top-500 re-scored by the XGBoost ranker

All three exclude items the user interacted with in train or val, and are
scored against the user's test-split positives with the from-scratch
NumPy metrics. Results are printed as a markdown table and written to
artifacts/eval_results.json.

Run: python -m src.eval.evaluate
"""

import json

import numpy as np
import xgboost as xgb

from src.config import EVAL_KS, N_CANDIDATES, RANKER_PATH, RESULTS_PATH
from src.dataio import ground_truth_by_user, load_interactions, seen_items_by_user
from src.eval.baselines import PopularityRecommender
from src.eval.metrics import evaluate_rankings
from src.ranking.features import FEATURE_COLUMNS, RankerFeatureBuilder
from src.retrieval.retrieve import load_embeddings, retrieve_candidates


def popularity_recs(interactions, seen, users, k) -> dict[int, np.ndarray]:
    model = PopularityRecommender(interactions)
    return {u: model.recommend(u, k, exclude=seen.get(u)) for u in users}


def retrieval_recs(user_vecs, item_vecs, seen, users, k) -> dict[int, np.ndarray]:
    cands, _ = retrieve_candidates(user_vecs, item_vecs, users, exclude=seen, n=k)
    return {u: cands[row] for row, u in enumerate(users)}


def two_stage_recs(user_vecs, item_vecs, seen, users, k) -> dict[int, np.ndarray]:
    cands, scores = retrieve_candidates(
        user_vecs, item_vecs, users, exclude=seen, n=N_CANDIDATES
    )
    builder = RankerFeatureBuilder()
    flat_users = np.repeat(users, N_CANDIDATES)
    X = builder.build(flat_users, cands.ravel(), scores.ravel())
    booster = xgb.Booster()
    booster.load_model(RANKER_PATH)
    ranker_scores = booster.predict(
        xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    ).reshape(len(users), N_CANDIDATES)

    # Re-order each user's candidates by ranker score, keep top-k.
    order = np.argsort(-ranker_scores, axis=1)[:, :k]
    reranked = np.take_along_axis(cands, order, axis=1)
    return {u: reranked[row] for row, u in enumerate(users)}


def main() -> None:
    interactions = load_interactions()
    truth = ground_truth_by_user(interactions, split="test")
    seen = seen_items_by_user(interactions, splits=("train", "val"))
    users = np.array(sorted(truth.keys()))
    user_vecs, item_vecs = load_embeddings()
    k_max = max(EVAL_KS)

    configs = {
        "popularity": popularity_recs(interactions, seen, users, k_max),
        "retrieval_only": retrieval_recs(user_vecs, item_vecs, seen, users, k_max),
        "two_stage": two_stage_recs(user_vecs, item_vecs, seen, users, k_max),
    }

    results = {
        name: evaluate_rankings(recs, truth, ks=EVAL_KS)
        for name, recs in configs.items()
    }

    metric_names = [f"{m}@{k}" for k in EVAL_KS for m in ("precision", "recall", "ndcg")]
    header = "| config | " + " | ".join(metric_names) + " |"
    sep = "|---" * (len(metric_names) + 1) + "|"
    print(f"\nTest split, {len(users):,} users:\n")
    print(header)
    print(sep)
    for name, res in results.items():
        cells = " | ".join(f"{res[m]:.4f}" for m in metric_names)
        print(f"| {name} | {cells} |")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
