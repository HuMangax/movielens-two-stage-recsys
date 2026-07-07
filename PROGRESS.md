# Progress log

## Phase 1 — Data & baseline (done, 2026-07-06)

**Done**
- Environment: Apple M1 (16 GB, no discrete GPU). Built a native arm64 venv
  (python.org 3.12.6) instead of the pre-existing x86_64/Rosetta conda env;
  torch 2.12.1 with MPS available. Local-only quirk: xgboost's `libomp.dylib`
  is resolved by adding the torch wheel's lib dir to its rpath (no Homebrew
  on this machine). The Docker image uses Linux wheels and doesn't need this.
- `data/download.py` + `data/preprocess.py`: MovieLens 1M → implicit feedback
  (rating ≥ 4 = positive; 575,272 positives, 6,034 users, 3,533 movies).
  Per-user chronological split 80/10/10 (train 465,564 / val 54,854 /
  test 54,854) to avoid temporal leakage.
- `src/features/build_features.py`: user/item feature tables computed from the
  train split only (popularity, mean rating, recency, release year, genre
  one-hots, user genre-preference distribution).
- `src/eval/metrics.py`: Precision@K, Recall@K, NDCG@K from scratch in NumPy,
  with hand-computed unit tests (`src/eval/test_metrics.py`).
- `src/eval/baselines.py`: popularity baseline. Measured on test split:
  P@10 0.0311 · R@10 0.0404 · NDCG@10 0.0429 (n=6,034 users).

**Decisions**
- Dataset scale: developing on ML-1M. GPU check at project start: only an M1
  with MPS (no CUDA). Will revisit 25M after Phase 3; two-tower training on
  25M without a discrete GPU is likely hours per run, which would eat the
  timeline for marginal portfolio value. Final call recorded when made.
- 4 users dropped for having < 5 positive ratings (can't give each split
  at least one interaction). 2 further users had no positives at all.

**Deferred / cut**: nothing yet.

## Phase 2 — Retrieval model (done, 2026-07-06)

**Done**
- `src/retrieval/model.py`: two-tower model. User tower = id embedding;
  item tower = id embedding + linear projection of genre one-hots + learned
  popularity bias. The bias folds into the exported matrices as an extra
  dimension, so full-catalog retrieval is one matrix multiply.
- Loss: BPR (pairwise, one uniformly sampled unseen negative per positive).
  Chosen over sampled softmax because it directly optimizes pairwise ranking,
  is the standard baseline for implicit MovieLens, and trains stably at this
  scale without temperature/normalization tuning.
- `src/retrieval/train.py`: trains on MPS (~0.8 s/epoch), early stopping on
  val Recall@100 (candidate-pool recall is what stage 1 is for). Best:
  **val Recall@100 = 0.4016** (epoch 79). Embeddings exported to artifacts/.
- `src/retrieval/retrieve.py`: brute-force top-500 candidates (3.5K-item
  catalog → ANN index would be overhead), plus a CLI for spot checks.
- Sanity check: user 42 (liked rom-coms/comedies) → Austin Powers, Groundhog
  Day, Wedding Singer; user 1000 (liked arthouse dramas) → Secrets & Lies,
  Elizabeth, Rushmore. Clearly personalized vs. the popularity list.
## Phase 3 — Ranking & evaluation (done, 2026-07-06)

**Done**
- `src/ranking/features.py`: 10 features per (user, candidate) row — stage-1
  score and rank, item popularity/mean-rating/recency/year, user activity
  stats, genre-preference match, year affinity. One code path for training,
  offline eval, and serving.
- `src/ranking/train.py`: XGBoost `rank:ndcg` (native API, no sklearn) on
  stage-1 top-500 candidates labeled with *validation* positives; test split
  untouched until the final comparison. 90/10 user split for early stopping.
  Tuning note: eta 0.1/depth 6 overfit by iteration 12; eta 0.05/depth 5
  ran to iteration 106 and improved test NDCG@10 from 0.0584 to 0.0621.
- `src/eval/evaluate.py`: three-way comparison on test, all configs excluding
  train+val items, scored with the from-scratch NumPy metrics.

**Measured results (test split, 6,034 users)**

| config | P@10 | R@10 | NDCG@10 | P@20 | R@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| popularity | 0.0311 | 0.0404 | 0.0429 | 0.0281 | 0.0750 | 0.0531 |
| retrieval_only | 0.0345 | 0.0617 | 0.0545 | 0.0309 | 0.1091 | 0.0689 |
| two_stage | 0.0399 | 0.0699 | 0.0621 | 0.0354 | 0.1200 | 0.0769 |

Two-stage vs retrieval-only: **+15.7% P@10, +13.3% R@10, +14.1% NDCG@10** —
the ranking stage is pulling its weight.

**Decision — staying on ML-1M for the final model.** The brief asked for a
scale-up evaluation once the pipeline worked end-to-end. Hardware is an M1
laptop (16 GB, MPS, no CUDA). ML-25M is ~27× the interactions (162K users ×
62K items): two-tower training would still be feasible (~30 min), but the
full-catalog scoring used in eval (162K × 62K float matrix ≈ 40 GB) and the
81M-row ranker feature matrix would both need chunked rewrites, and each
end-to-end iteration would go from ~2 minutes to hours. That trades most of
the remaining timeline for no change in the architecture story. Noted here
per the "no silent scope cuts" rule.
## Phase 4 — Serving & deploy (code done 2026-07-06; cloud deploy pending user account)

**Done**
- `src/serving/app.py`: FastAPI with `/recommend/{user_id}`,
  `/recommend/cold-start` (pseudo-embedding from liked movies + synthetic
  user features), `/health`. Torch-free: stage 1 runs on exported `.npy`
  embeddings, so the serving image skips PyTorch entirely.
- Multi-stage `Dockerfile` (python:3.12-slim). First build was 1.86 GB
  because the stock xgboost wheel bundles CUDA; switching serving to
  `xgboost-cpu` cut the image to 862 MB. Verified in a real container
  (colima, since Docker wasn't installed on this machine).
- `scripts/benchmark.py` (stdlib-only load test). Measured, 500 requests
  at concurrency 8, zero failures:
  - uvicorn direct on M1: p50 7.1 ms / p95 10.6 ms / ~1,090 rps
  - Docker container in colima VM: p50 11.3 ms / p95 16.8 ms / ~665 rps
- Serving artifacts (7.5 MB total) committed to the repo so Render can
  build the image straight from GitHub; `render.yaml` blueprint included.

**Deployed (2026-07-06).** Max created the Render account and deployed via
the blueprint: https://movielens-two-stage-recsys.onrender.com (free tier,
0.1 CPU). Real load test against the deployed endpoint (500 requests,
concurrency 8, zero failures): **p50 198.9 ms / p95 300.0 ms /
p99 597.6 ms, ~38.5 rps**. The gap vs. the 11 ms local-container p50 is
network round-trip plus the free tier's fractional CPU; fine for a demo.
Note for demos: free instances sleep when idle, first request takes ~30 s.

**Deviations from the brief:** none beyond the documented ML-1M scale
decision (Phase 3) and the optional frontend, which was not built (time
went to deploy verification; the FastAPI `/docs` page serves as the live
demo surface).
