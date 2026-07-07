"""Candidate retrieval: top-N items per user by embedding dot product.

Brute-force scoring over the full catalog — at 3.5K items this is a single
small matrix multiply, so an ANN index would be pure overhead.

Run standalone for a qualitative sanity check of a few users:
    python -m src.retrieval.retrieve --user 42
"""

import argparse

import numpy as np

from src.config import ITEM_EMBEDDINGS_PATH, N_CANDIDATES, USER_EMBEDDINGS_PATH


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    return np.load(USER_EMBEDDINGS_PATH), np.load(ITEM_EMBEDDINGS_PATH)


def retrieve_candidates(
    user_vecs: np.ndarray,
    item_vecs: np.ndarray,
    user_indices: np.ndarray,
    exclude: dict[int, np.ndarray] | None = None,
    n: int = N_CANDIDATES,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-n candidates for each user in ``user_indices``.

    Returns (candidates [len(users), n], scores [len(users), n]),
    both sorted by descending score. Items in ``exclude[user]`` are
    filtered out before the top-n cut.
    """
    scores = user_vecs[user_indices] @ item_vecs.T
    if exclude:
        for row, user in enumerate(user_indices):
            seen = exclude.get(int(user))
            if seen is not None and len(seen):
                scores[row, seen] = -np.inf

    # argpartition for the top-n, then sort just those n.
    top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    row_scores = np.take_along_axis(scores, top, axis=1)
    order = np.argsort(-row_scores, axis=1)
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(
        row_scores, order, axis=1
    )


def main() -> None:
    from src.dataio import load_interactions, load_items, seen_items_by_user

    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=int, default=42, help="user_idx to inspect")
    parser.add_argument("--n", type=int, default=15, help="candidates to show")
    args = parser.parse_args()

    user_vecs, item_vecs = load_embeddings()
    interactions = load_interactions()
    items = load_items().set_index("item_idx")
    seen = seen_items_by_user(interactions, splits=("train",))

    history = interactions[
        (interactions["user_idx"] == args.user) & (interactions["split"] == "train")
    ].sort_values("rating", ascending=False)
    print(f"User {args.user} — sample of liked training movies:")
    for _, row in history.head(10).iterrows():
        print(f"  [{row.rating}] {items.loc[row.item_idx, 'title']}")

    cands, scores = retrieve_candidates(
        user_vecs, item_vecs, np.array([args.user]), exclude=seen, n=N_CANDIDATES
    )
    print(f"\nTop {args.n} of {N_CANDIDATES} retrieved candidates:")
    for item, score in zip(cands[0][: args.n], scores[0][: args.n]):
        print(f"  {score:6.2f}  {items.loc[item, 'title']}")


if __name__ == "__main__":
    main()
