"""Latency load test for the recommendation endpoint. Pure stdlib.

Fires N requests at the API from C concurrent threads, cycling through a
pool of real user ids, and reports p50/p95/p99 wall-clock latency plus
throughput. Failures are counted, not silently dropped.

Usage:
    python scripts/benchmark.py --base-url http://localhost:8000 \
        --requests 500 --concurrency 8
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Real MovieLens user ids (raw ids, not internal indices).
USER_IDS = [1, 42, 100, 555, 1024, 2000, 3000, 4169, 5000, 6040]


def one_request(base_url: str, user_id: int, timeout: float) -> tuple[float, bool]:
    url = f"{base_url}/recommend/{user_id}?k=10"
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            ok = resp.status == 200 and b"recommendations" in body
    except (urllib.error.URLError, TimeoutError):
        ok = False
    return time.perf_counter() - start, ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    for i in range(args.warmup):
        one_request(base, USER_IDS[i % len(USER_IDS)], args.timeout)

    jobs = [USER_IDS[i % len(USER_IDS)] for i in range(args.requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda u: one_request(base, u, args.timeout), jobs))
    wall = time.perf_counter() - t0

    latencies = sorted(lat for lat, ok in results if ok)
    failures = sum(1 for _, ok in results if not ok)
    if not latencies:
        raise SystemExit("All requests failed — is the server up?")

    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))]

    report = {
        "base_url": base,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "failures": failures,
        "throughput_rps": round(args.requests / wall, 1),
        "p50_ms": round(pct(50) * 1000, 1),
        "p95_ms": round(pct(95) * 1000, 1),
        "p99_ms": round(pct(99) * 1000, 1),
        "mean_ms": round(statistics.mean(latencies) * 1000, 1),
        "max_ms": round(latencies[-1] * 1000, 1),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
