"""FastAPI serving layer for the two-stage recommender.

Endpoints:
- GET  /health
- GET  /recommend/{user_id}?k=10   — full two-stage pipeline for a known user
- POST /recommend/cold-start       — recommendations from a list of liked
                                     movie_ids for a user we've never seen

Serving needs no PyTorch: the two-tower model's output is frozen into
user/item embedding matrices at training time, so stage 1 is a NumPy
matrix multiply and stage 2 is an XGBoost predict.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    ITEM_EMBEDDINGS_PATH,
    N_CANDIDATES,
    RANKER_PATH,
    TOP_K,
    USER_EMBEDDINGS_PATH,
)
from src.dataio import load_interactions, load_items, seen_items_by_user
from src.ranking.features import FEATURE_COLUMNS, RankerFeatureBuilder
from src.retrieval.retrieve import retrieve_candidates

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_: FastAPI):
    global state
    state = ServingState()
    yield


app = FastAPI(
    title="MovieLens two-stage recommender",
    description="PyTorch two-tower retrieval + XGBoost ranking, MovieLens 1M.",
    lifespan=lifespan,
)


class Recommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int | None
    k: int
    recommendations: list[Recommendation]


class ColdStartRequest(BaseModel):
    liked_movie_ids: list[int] = Field(min_length=1, max_length=50)
    k: int = Field(default=TOP_K, ge=1, le=100)


class ServingState:
    """Everything loaded once at startup."""

    def __init__(self) -> None:
        self.user_vecs = np.load(USER_EMBEDDINGS_PATH)
        self.item_vecs = np.load(ITEM_EMBEDDINGS_PATH)
        self.booster = xgb.Booster()
        self.booster.load_model(str(RANKER_PATH))
        self.features = RankerFeatureBuilder()

        interactions = load_interactions()
        self.seen = seen_items_by_user(interactions, splits=("train", "val", "test"))
        # raw MovieLens user_id -> contiguous user_idx
        pairs = interactions[["user_id", "user_idx"]].drop_duplicates()
        self.user_id_to_idx = dict(
            zip(pairs["user_id"].tolist(), pairs["user_idx"].tolist())
        )

        items = load_items()
        self.item_titles = items.set_index("item_idx")["title"]
        self.item_movie_ids = items.set_index("item_idx")["movie_id"]
        self.movie_id_to_idx = dict(
            zip(items["movie_id"].tolist(), items["item_idx"].tolist())
        )
        # Fallback user stats for cold-start requests.
        self.global_mean_rating = float(
            interactions.loc[interactions["split"] == "train", "rating"].mean()
        )


state: ServingState | None = None


def run_two_stage(
    user_vec: np.ndarray,
    exclude: np.ndarray | None,
    k: int,
    user_idx_for_features: int | None,
    cold_user_features: dict | None = None,
) -> list[Recommendation]:
    """Retrieve top-500 with the given user vector, re-rank, return top-k."""
    s = state
    scores = user_vec @ s.item_vecs.T
    if exclude is not None and len(exclude):
        scores[exclude] = -np.inf
    n = min(N_CANDIDATES, len(scores))
    cand = np.argpartition(-scores, n - 1)[:n]
    order = np.argsort(-scores[cand])
    cand = cand[order]
    cand_scores = scores[cand]

    if user_idx_for_features is not None:
        users = np.full(len(cand), user_idx_for_features)
        X = s.features.build(users, cand, cand_scores)
    else:
        # Cold user: patch the user-side columns with synthetic stats.
        users = np.zeros(len(cand), dtype=np.int64)
        X = s.features.build(users, cand, cand_scores)
        cols = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
        X[:, cols["user_count"]] = cold_user_features["user_count"]
        X[:, cols["user_mean_rating"]] = cold_user_features["user_mean_rating"]
        X[:, cols["genre_match"]] = (
            s.features.item_genres[cand] @ cold_user_features["genre_prefs"]
        )
        X[:, cols["year_diff_abs"]] = np.abs(
            s.features.item_year[cand] - cold_user_features["mean_year"]
        )

    ranker_scores = s.booster.predict(xgb.DMatrix(X, feature_names=FEATURE_COLUMNS))
    top = np.argsort(-ranker_scores)[:k]
    return [
        Recommendation(
            rank=i + 1,
            movie_id=int(s.item_movie_ids.loc[item]),
            title=str(s.item_titles.loc[item]),
            score=round(float(ranker_scores[j]), 4),
        )
        for i, (j, item) in enumerate(zip(top, cand[top]))
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_users": len(state.user_vecs), "n_items": len(state.item_vecs)}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, k: int = TOP_K) -> RecommendResponse:
    if not 1 <= k <= 100:
        raise HTTPException(status_code=422, detail="k must be in [1, 100]")
    user_idx = state.user_id_to_idx.get(user_id)
    if user_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown user_id {user_id}. Use POST /recommend/cold-start "
            "with liked_movie_ids for new users.",
        )
    recs = run_two_stage(
        state.user_vecs[user_idx],
        exclude=state.seen.get(user_idx),
        k=k,
        user_idx_for_features=user_idx,
    )
    return RecommendResponse(user_id=user_id, k=k, recommendations=recs)


@app.post("/recommend/cold-start", response_model=RecommendResponse)
def cold_start(req: ColdStartRequest) -> RecommendResponse:
    liked = [
        state.movie_id_to_idx[m] for m in req.liked_movie_ids
        if m in state.movie_id_to_idx
    ]
    if not liked:
        raise HTTPException(
            status_code=422, detail="None of the liked_movie_ids are in the catalog."
        )
    liked_arr = np.array(liked)

    # Pseudo user embedding: mean of liked item vectors (dropping the bias
    # column), with the constant-1 slot appended so item biases still apply.
    # This scores items by similarity to the user's liked set.
    item_part = state.item_vecs[liked_arr, :-1].mean(axis=0)
    user_vec = np.concatenate([item_part, [1.0]]).astype(np.float32)

    genre_prefs = state.features.item_genres[liked_arr].mean(axis=0)
    total = genre_prefs.sum()
    if total > 0:
        genre_prefs = genre_prefs / total
    cold_feats = {
        "user_count": float(len(liked)),
        "user_mean_rating": state.global_mean_rating,
        "genre_prefs": genre_prefs,
        "mean_year": float(state.features.item_year[liked_arr].mean()),
    }
    recs = run_two_stage(
        user_vec, exclude=liked_arr, k=req.k,
        user_idx_for_features=None, cold_user_features=cold_feats,
    )
    return RecommendResponse(user_id=None, k=req.k, recommendations=recs)
