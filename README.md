# MovieLens two-stage recommender

A production-style recommendation engine on MovieLens 1M: a **PyTorch
two-tower model** retrieves ~500 candidates from the catalog, and an
**XGBoost ranker** re-scores them into a final top-10 — the same
retrieval-then-ranking shape used at YouTube, Netflix, and most large-scale
recommenders. Served with FastAPI from a multi-stage Docker image.

```
                        ┌─────────────────────┐      ┌────────────────────┐
 user id ──────────────▶│  Stage 1: retrieval │─────▶│  Stage 2: ranking  │──▶ top-10
                        │  two-tower (PyTorch)│ ~500 │  XGBoost rank:ndcg │
                        │  user·item dot prod │ cands│  10 features/pair  │
                        └─────────────────────┘      └────────────────────┘
                          embeddings precomputed        popularity, recency,
                          at train time → serving       genre match, stage-1
                          is a NumPy matmul, no torch   score, activity stats
```

## Results

Held-out test split (newest 10% of each user's likes), 6,034 users,
metrics implemented **from scratch in NumPy** ([src/eval/metrics.py](src/eval/metrics.py)):

| config | Precision@10 | Recall@10 | NDCG@10 | Precision@20 | Recall@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| popularity baseline | 0.0311 | 0.0404 | 0.0429 | 0.0281 | 0.0750 | 0.0531 |
| retrieval only (stage 1) | 0.0345 | 0.0617 | 0.0545 | 0.0309 | 0.1091 | 0.0689 |
| **two-stage (full)** | **0.0399** | **0.0699** | **0.0621** | **0.0354** | **0.1200** | **0.0769** |

The ranking stage adds **+15.7% Precision@10, +13.3% Recall@10, +14.1%
NDCG@10** over retrieval alone — the second stage pays for itself.
(Reproduce with `python -m src.eval.evaluate`; raw numbers in
[artifacts/eval_results.json](artifacts/eval_results.json).)

Absolute numbers look small because the protocol is strict: ranking the
full 3.5K-item catalog against only the handful of movies each user went
on to like in their final 10% of history, with everything they already
watched excluded.

## Serving latency

Measured with [scripts/benchmark.py](scripts/benchmark.py) (500 requests,
concurrency 8, full two-stage pipeline per request):

| target | p50 | p95 | p99 | throughput |
|---|---|---|---|---|
| local (uvicorn, M1) | 7.1 ms | 10.6 ms | 12.2 ms | ~1,090 req/s |
| Docker container (colima VM, M1) | 11.3 ms | 16.8 ms | 32.2 ms | ~665 req/s |
| **deployed** ([Render free tier](https://movielens-two-stage-recsys.onrender.com)) | **198.9 ms** | **300.0 ms** | 597.6 ms | ~38.5 req/s |

Deployed numbers include real network round-trip to Render's servers and the
free tier's 0.1-CPU instance; 500 requests, zero failures. Live at
`https://movielens-two-stage-recsys.onrender.com` (free instances sleep when
idle — the first request after a while takes ~30 s to wake).

## Design decisions

- **BPR loss** for the two-tower model: directly optimizes pairwise ranking
  on implicit feedback, is the standard baseline for this setup, and trains
  stably without the temperature/normalization tuning sampled softmax needs.
  One uniformly-sampled unseen negative per positive.
- **Temporal split, not random.** Each user's likes are split 80/10/10 by
  timestamp (train/val/test). A random split would leak future taste into
  training and inflate every number.
- **Leakage discipline.** Retrieval trains on the train split; the ranker
  trains on retrieval's candidates labeled with *validation* positives; the
  test split is touched exactly once, by the final comparison. All engineered
  features derive from train-split statistics only.
- **Ranker features** (10 per user-item pair): stage-1 score and rank, item
  popularity / mean rating / recency / release year, user activity stats,
  genre-preference match, year affinity.
- **No ANN index.** At 3.5K items, brute-force retrieval is one small matrix
  multiply (~sub-ms). FAISS would be pure overhead at this scale.
- **Torch-free serving.** The two-tower model is frozen into `.npy` embedding
  matrices at export time, so the serving image needs only NumPy + XGBoost +
  FastAPI (no 800 MB of PyTorch).
- **Dataset scale: ML-1M, deliberately.** Hardware for this build was an M1
  laptop (no CUDA). ML-25M would not change the architecture story but would
  turn the 2-minute end-to-end iteration loop into hours (full-catalog eval
  scoring alone is a 162K×62K matrix). Documented in
  [PROGRESS.md](PROGRESS.md).

## Live demo

**https://movielens-two-stage-recsys.onrender.com** — pick an existing user
id or search for a few movies you like (cold start) and get top-10
recommendations from the full two-stage pipeline. Plain-HTML/vanilla-JS page
served by the API itself ([src/serving/static/index.html](src/serving/static/index.html));
no build step, no CORS. Free-tier host: first request after idle takes ~30 s.

## API

```bash
# Known user → personalized top-10
curl "$HOST/recommend/1?k=10"

# Cold start: no account, just a few liked movies (MovieLens movie ids)
curl -X POST "$HOST/recommend/cold-start" \
  -H 'Content-Type: application/json' \
  -d '{"liked_movie_ids": [260, 1196, 480], "k": 10}'

curl "$HOST/health"
```

Interactive docs at `$HOST/docs` (FastAPI/OpenAPI).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/download.py            # fetch MovieLens 1M (~6 MB)
python data/preprocess.py          # implicit feedback + temporal split
python -m src.features.build_features
python -m src.retrieval.train      # two-tower, ~90 s on an M1 (MPS)
python -m src.ranking.train        # XGBoost ranker, ~1 min
python -m src.eval.evaluate        # three-way comparison table
python -m src.eval.test_metrics    # unit tests for the NumPy metrics

# Serve locally
uvicorn src.serving.app:app --port 8000

# Or in Docker (multi-stage build; artifacts baked in)
docker build -t movielens-recsys .
docker run -p 8000:8000 movielens-recsys

# Load test whatever is running
python scripts/benchmark.py --base-url http://localhost:8000
```

Every number in this README comes from an executed run of the code in this
repo, on the artifacts committed here.

## Repo layout

```
data/            download + preprocessing scripts (raw data not committed)
src/features/    train-split-only feature engineering
src/retrieval/   PyTorch two-tower model, training, embedding export
src/ranking/     XGBoost ranker + shared feature assembly
src/eval/        from-scratch NumPy metrics, baselines, comparison harness
src/serving/     FastAPI app (torch-free)
scripts/         latency benchmark
artifacts/       committed: trained serving artifacts + eval results
Dockerfile       multi-stage build
render.yaml      one-click Render deploy blueprint
PROGRESS.md      phase-by-phase build log, decisions, deviations
```
