# Multi-stage build: wheels are built/downloaded in a throwaway stage so the
# runtime image ships no compilers or pip caches.
#
# The image serves the *trained* model: artifacts/ (embeddings, ranker,
# feature tables) is baked in at build time. Run the training pipeline
# before building — see README "Reproduce" section.

FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt

FROM python:3.12-slim
# libgomp: OpenMP runtime for xgboost's native library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY artifacts/interactions.parquet artifacts/items.parquet \
     artifacts/item_features.parquet artifacts/user_features.parquet \
     artifacts/user_embeddings.npy artifacts/item_embeddings.npy \
     artifacts/xgb_ranker.json artifacts/

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}"]
