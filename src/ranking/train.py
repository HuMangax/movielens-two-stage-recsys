"""Train the stage-2 XGBoost ranker on stage-1 candidates.

Setup (leakage-safe):
- The two-tower model was trained on the *train* split only.
- For every user we retrieve the top-500 candidates, excluding items the
  user saw in train (validation items stay in — they're the labels).
- Label: 1 if the candidate is one of the user's *validation* positives.
- The *test* split is never touched here; it's reserved for the final
  three-way comparison in src/eval/evaluate.py.

Users are split 90/10 into ranker-train / ranker-early-stopping groups.
Objective: rank:ndcg (listwise LambdaMART), which directly optimizes the
metric we report.

Run: python -m src.ranking.train
"""

import numpy as np
import xgboost as xgb

from src.config import N_CANDIDATES, RANKER_PATH, SEED
from src.dataio import ground_truth_by_user, load_interactions, seen_items_by_user
from src.ranking.features import FEATURE_COLUMNS, RankerFeatureBuilder
from src.retrieval.retrieve import load_embeddings, retrieve_candidates


def build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features X, labels y, qid) over all users with >=1 val hit.

    Groups where retrieval missed every val positive carry no ranking signal
    (all labels 0), so they're dropped — this also speeds up training.
    """
    interactions = load_interactions()
    user_vecs, item_vecs = load_embeddings()
    seen_train = seen_items_by_user(interactions, splits=("train",))
    val_truth = ground_truth_by_user(interactions, split="val")

    users = np.array(sorted(val_truth.keys()))
    cands, scores = retrieve_candidates(
        user_vecs, item_vecs, users, exclude=seen_train, n=N_CANDIDATES
    )

    builder = RankerFeatureBuilder()
    keep_rows_users, keep_rows_cands, keep_rows_scores, labels, qids = [], [], [], [], []
    for row, user in enumerate(users):
        y = np.isin(cands[row], val_truth[user]).astype(np.float32)
        if y.sum() == 0:
            continue
        keep_rows_users.append(np.full(N_CANDIDATES, user))
        keep_rows_cands.append(cands[row])
        keep_rows_scores.append(scores[row])
        labels.append(y)
        qids.append(np.full(N_CANDIDATES, user))

    flat_users = np.concatenate(keep_rows_users)
    X = builder.build(
        flat_users, np.concatenate(keep_rows_cands), np.concatenate(keep_rows_scores)
    )
    y = np.concatenate(labels)
    qid = np.concatenate(qids)
    print(
        f"Ranker data: {len(np.unique(qid)):,} users kept "
        f"(of {len(users):,}), {len(y):,} rows, {int(y.sum()):,} positives"
    )
    return X, y, qid


def main() -> None:
    X, y, qid = build_training_data()

    # Hold out 10% of users for early stopping.
    rng = np.random.default_rng(SEED)
    unique_users = np.unique(qid)
    val_users = rng.choice(unique_users, size=len(unique_users) // 10, replace=False)
    is_val = np.isin(qid, val_users)

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=SEED,
    )
    model.fit(
        X[~is_val],
        y[~is_val],
        qid=qid[~is_val],
        eval_set=[(X[is_val], y[is_val])],
        eval_qid=[qid[is_val]],
        verbose=25,
    )
    print(
        f"Best iteration {model.best_iteration}, "
        f"early-stop users ndcg@10 = {model.best_score:.4f}"
    )

    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1]
    )
    print("Feature importances (gain):")
    for name, imp in importances:
        print(f"  {name:22s} {imp:.3f}")

    model.save_model(RANKER_PATH)
    print(f"Saved ranker to {RANKER_PATH}")


if __name__ == "__main__":
    main()
