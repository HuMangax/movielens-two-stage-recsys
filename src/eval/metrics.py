"""Ranking metrics implemented from scratch in NumPy.

No sklearn / surprise / torchmetrics — these are deliberately hand-rolled.

Conventions:
- ``recommended``: 1-D array of item indices, best first (length >= k).
- ``relevant``: 1-D array (or set-like) of ground-truth positive item indices.
- All metrics use binary relevance.
"""

import numpy as np


def precision_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = np.asarray(recommended)[:k]
    hits = np.isin(top_k, relevant).sum()
    return float(hits) / k


def recall_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    relevant = np.asarray(relevant)
    if relevant.size == 0:
        return 0.0
    top_k = np.asarray(recommended)[:k]
    hits = np.isin(top_k, relevant).sum()
    return float(hits) / relevant.size


def ndcg_at_k(recommended: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    DCG  = sum over positions p (1-indexed) of rel_p / log2(p + 1)
    IDCG = DCG of the ideal ranking: all relevant items first, i.e. the
           first min(k, |relevant|) positions are hits.
    """
    relevant = np.asarray(relevant)
    if relevant.size == 0:
        return 0.0
    top_k = np.asarray(recommended)[:k]
    gains = np.isin(top_k, relevant).astype(np.float64)
    positions = np.arange(1, top_k.size + 1, dtype=np.float64)
    dcg = float(np.sum(gains / np.log2(positions + 1)))

    n_ideal = min(k, relevant.size)
    ideal_positions = np.arange(1, n_ideal + 1, dtype=np.float64)
    idcg = float(np.sum(1.0 / np.log2(ideal_positions + 1)))
    return dcg / idcg


def evaluate_rankings(
    recommendations: dict[int, np.ndarray],
    ground_truth: dict[int, np.ndarray],
    ks: tuple[int, ...] = (10,),
) -> dict[str, float]:
    """Average the three metrics over all users present in ``ground_truth``.

    Users with no recommendations score 0 on every metric (they still count
    in the denominator, so a model can't win by skipping hard users).
    """
    results: dict[str, float] = {}
    users = sorted(ground_truth.keys())
    for k in ks:
        precisions, recalls, ndcgs = [], [], []
        for user in users:
            relevant = ground_truth[user]
            recs = recommendations.get(user)
            if recs is None or len(recs) == 0:
                precisions.append(0.0)
                recalls.append(0.0)
                ndcgs.append(0.0)
                continue
            precisions.append(precision_at_k(recs, relevant, k))
            recalls.append(recall_at_k(recs, relevant, k))
            ndcgs.append(ndcg_at_k(recs, relevant, k))
        results[f"precision@{k}"] = float(np.mean(precisions))
        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))
    results["n_users"] = len(users)
    return results
