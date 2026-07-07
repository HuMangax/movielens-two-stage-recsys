"""Train the two-tower retrieval model with BPR loss.

For each observed (user, positive item) pair we sample one item the user
has not interacted with in training and push the positive's score above
the negative's. Early stopping monitors validation Recall@100 — the metric
that matters for a candidate generator (did the good items make the pool?),
as opposed to NDCG, which rewards fine-grained ordering the ranker will
redo anyway.

Run: python -m src.retrieval.train
"""

import time

import numpy as np
import pandas as pd
import torch

from src.config import (
    EMBEDDING_DIM,
    ITEM_EMBEDDINGS_PATH,
    ITEM_FEATURES_PATH,
    RETRIEVAL_CKPT_PATH,
    SEED,
    USER_EMBEDDINGS_PATH,
)
from src.dataio import load_interactions
from src.retrieval.model import TwoTowerModel, bpr_loss

BATCH_SIZE = 8192
LR = 1e-3
L2 = 1e-6
MAX_EPOCHS = 120
PATIENCE = 5
EVAL_RECALL_K = 100


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_negatives(
    users: np.ndarray, n_items: int, user_pos: list[set], rng: np.random.Generator
) -> np.ndarray:
    """Uniform negatives, resampled where we accidentally hit a positive."""
    negs = rng.integers(0, n_items, size=len(users))
    for i, (u, j) in enumerate(zip(users, negs)):
        while j in user_pos[u]:
            j = rng.integers(0, n_items)
        negs[i] = j
    return negs


@torch.no_grad()
def val_recall(
    model: TwoTowerModel,
    item_feats: torch.Tensor,
    train_mask_rows: np.ndarray,
    train_mask_cols: np.ndarray,
    val_truth: dict[int, np.ndarray],
    k: int,
) -> float:
    """Mean Recall@k on the validation split, excluding train items."""
    user_mat, item_mat = model.export_embeddings(item_feats)
    scores = (user_mat @ item_mat.T).cpu().numpy()
    scores[train_mask_rows, train_mask_cols] = -np.inf  # never re-recommend seen

    # argpartition: top-k per user without a full sort.
    top_k = np.argpartition(-scores, k, axis=1)[:, :k]
    recalls = [
        len(np.intersect1d(top_k[u], truth, assume_unique=False)) / len(truth)
        for u, truth in val_truth.items()
    ]
    return float(np.mean(recalls))


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = pick_device()
    print(f"Device: {device}")

    interactions = load_interactions()
    n_users = int(interactions["user_idx"].max()) + 1
    n_items = int(interactions["item_idx"].max()) + 1

    train = interactions[interactions["split"] == "train"]
    users_arr = train["user_idx"].to_numpy()
    items_arr = train["item_idx"].to_numpy()
    print(f"{n_users} users, {n_items} items, {len(train):,} training pairs")

    user_pos: list[set] = [set() for _ in range(n_users)]
    for u, i in zip(users_arr, items_arr):
        user_pos[u].add(i)

    val = interactions[interactions["split"] == "val"]
    val_truth = {
        u: g.to_numpy() for u, g in val.groupby("user_idx")["item_idx"]
    }

    item_feat_df = pd.read_parquet(ITEM_FEATURES_PATH).sort_values("item_idx")
    genre_cols = [c for c in item_feat_df.columns if c.startswith("genre_")]
    item_feats = torch.tensor(
        item_feat_df[genre_cols].to_numpy(dtype=np.float32), device=device
    )

    model = TwoTowerModel(n_users, n_items, len(genre_cols), EMBEDDING_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

    best_recall, best_state, patience_left = -1.0, None, PATIENCE
    n_batches = int(np.ceil(len(users_arr) / BATCH_SIZE))

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        perm = rng.permutation(len(users_arr))
        negs = sample_negatives(users_arr, n_items, user_pos, rng)

        model.train()
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            u = torch.from_numpy(users_arr[idx]).to(device)
            pos = torch.from_numpy(items_arr[idx]).to(device)
            neg = torch.from_numpy(negs[idx]).to(device)

            loss = bpr_loss(
                model.score(u, pos, item_feats), model.score(u, neg, item_feats)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)

        model.eval()
        recall = val_recall(
            model, item_feats, users_arr, items_arr, val_truth, EVAL_RECALL_K
        )
        print(
            f"epoch {epoch:2d}  loss {epoch_loss / len(users_arr):.4f}  "
            f"val recall@{EVAL_RECALL_K} {recall:.4f}  ({time.time() - t0:.1f}s)"
        )

        if recall > best_recall:
            best_recall, patience_left = recall, PATIENCE
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "state_dict": best_state,
            "n_users": n_users,
            "n_items": n_items,
            "n_item_feats": len(genre_cols),
            "dim": EMBEDDING_DIM,
            "val_recall": best_recall,
        },
        RETRIEVAL_CKPT_PATH,
    )
    user_mat, item_mat = model.export_embeddings(item_feats)
    np.save(USER_EMBEDDINGS_PATH, user_mat.cpu().numpy().astype(np.float32))
    np.save(ITEM_EMBEDDINGS_PATH, item_mat.cpu().numpy().astype(np.float32))
    print(
        f"Saved checkpoint (best val recall@{EVAL_RECALL_K} {best_recall:.4f}) "
        f"and embeddings: users {tuple(user_mat.shape)}, items {tuple(item_mat.shape)}"
    )


if __name__ == "__main__":
    main()
