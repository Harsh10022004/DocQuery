import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import DocuQueryEngine

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "load_test_results.json")

SAMPLE_QUERIES = [
    "How to map ports in docker?",
    "How to map ports in docker?", # repeated query -> cache hit
    "How to define path parameters in FastAPI?",
    "How to define path parameters in FastAPI?", # repeated query -> cache hit
    "What is the difference between git merge and git rebase?",
    "What does git cherry-pick do?",
    "When to use GIN index in Postgres?",
    "How to run background tasks in FastAPI?",
    "How to run background tasks in FastAPI?", # repeated query -> cache hit
    "What is git stash pop?",
    "How to build docker image from Dockerfile?",
    "How to build docker image from Dockerfile?", # repeated query -> cache hit
    "Hello! What can you do?", # small talk router
    "Hi there!", # small talk router
    "How to return HTTPException in FastAPI?"
]


def run_load_test(iterations=3):
    print("Starting load and latency benchmark...")
    engine = DocuQueryEngine()
    
    latencies = []
    costs = []
    cache_hits = 0
    total_queries = len(SAMPLE_QUERIES) * iterations

    start_wall_time = time.time()

    for it in range(iterations):
        for q in SAMPLE_QUERIES:
            res = engine.query(q)
            lat = res["latency_seconds"]
            cost = res["cost_usd"]
            
            latencies.append(lat)
            costs.append(cost)
            if res.get("cache_hit", False):
                cache_hits += 1

    total_time = time.time() - start_wall_time
    throughput_rps = round(total_queries / total_time, 2)

    # calculate latency percentiles
    p50 = round(float(np.percentile(latencies, 50)), 3)
    p90 = round(float(np.percentile(latencies, 90)), 3)
    p99 = round(float(np.percentile(latencies, 99)), 3)
    avg_lat = round(float(np.mean(latencies)), 3)
    
    cache_hit_rate = round((cache_hits / total_queries) * 100, 1)
    avg_cost = round(float(np.mean(costs)), 6)
    cost_per_1k = round(avg_cost * 1000, 4)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_requests": total_queries,
        "throughput_rps": throughput_rps,
        "latency_p50_seconds": p50,
        "latency_p90_seconds": p90,
        "latency_p99_seconds": p99,
        "latency_mean_seconds": avg_lat,
        "cache_hit_rate_pct": cache_hit_rate,
        "avg_cost_per_request_usd": avg_cost,
        "cost_per_1000_requests_usd": cost_per_1k
    }

    print("\n" + "="*50)
    print("LOAD TEST & LATENCY BENCHMARK RESULTS:")
    print(f"Total Requests:     {total_queries}")
    print(f"Throughput:         {throughput_rps} req/sec")
    print(f"P50 Latency:        {p50}s")
    print(f"P90 Latency:        {p90}s")
    print(f"P99 Latency:        {p99}s")
    print(f"Cache Hit Rate:     {cache_hit_rate}%")
    print(f"Avg Cost / Request: ${avg_cost:.6f}")
    print(f"Cost / 1000 Reqs:   ${cost_per_1k:.4f}")
    print("="*50)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved load test metrics to: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_load_test()
