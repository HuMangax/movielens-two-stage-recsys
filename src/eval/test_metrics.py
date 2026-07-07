"""Hand-computed sanity checks for the from-scratch metrics.

Run: python -m src.eval.test_metrics  (or pytest src/eval/test_metrics.py)
"""

import math

import numpy as np

from src.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision() -> None:
    recs = np.array([1, 2, 3, 4, 5])
    rel = np.array([2, 5, 9])
    # hits in top-5: items 2 and 5 -> 2/5
    assert precision_at_k(recs, rel, 5) == 2 / 5
    # top-1 = [1], no hit
    assert precision_at_k(recs, rel, 1) == 0.0


def test_recall() -> None:
    recs = np.array([1, 2, 3, 4, 5])
    rel = np.array([2, 5, 9])
    # 2 of 3 relevant items retrieved
    assert recall_at_k(recs, rel, 5) == 2 / 3
    assert recall_at_k(recs, np.array([]), 5) == 0.0


def test_ndcg_perfect_ranking_is_1() -> None:
    recs = np.array([7, 8, 9, 1, 2])
    rel = np.array([7, 8, 9])
    assert math.isclose(ndcg_at_k(recs, rel, 5), 1.0)


def test_ndcg_hand_computed() -> None:
    # Hits at positions 1 and 3 of the top-3; one relevant item missed.
    recs = np.array([10, 11, 12])
    rel = np.array([10, 12, 99])
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # positions 1 and 3
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 ideal hits
    assert math.isclose(ndcg_at_k(recs, rel, 3), dcg / idcg)


def test_ndcg_single_relevant_at_position_2() -> None:
    recs = np.array([5, 6])
    rel = np.array([6])
    # DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert math.isclose(ndcg_at_k(recs, rel, 2), 1 / math.log2(3))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("All metric tests passed.")
