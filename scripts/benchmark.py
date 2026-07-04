#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from urllib import request as urlrequest
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the inference service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--features", type=int, default=4)
    args = parser.parse_args()

    latencies_ms: list[float] = []
    total_items = 0
    started = time.perf_counter()

    for _ in range(args.requests):
        payload = {
            "inputs": np.random.default_rng().normal(
                size=(args.batch_size, args.features)
            ).astype(float).tolist()
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            args.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urlrequest.urlopen(req, timeout=10) as response:
            response.read()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        total_items += args.batch_size

    elapsed = time.perf_counter() - started
    print(f"requests: {args.requests}")
    print(f"items: {total_items}")
    print(f"throughput_items_s: {total_items / elapsed:.2f}")
    print(f"latency_ms_p50: {statistics.median(latencies_ms):.2f}")
    print(f"latency_ms_p95: {percentile(latencies_ms, 95):.2f}")
    print(f"latency_ms_max: {max(latencies_ms):.2f}")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


if __name__ == "__main__":
    main()
