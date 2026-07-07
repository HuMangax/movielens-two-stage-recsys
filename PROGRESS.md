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

## Phase 2 — Retrieval model (pending)
## Phase 3 — Ranking & evaluation (pending)
## Phase 4 — Serving & deploy (pending)
