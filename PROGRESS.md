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
## Phase 3 — Ranking & evaluation (pending)
## Phase 4 — Serving & deploy (pending)
