"""
Runs the benchmark questions through naive RAG mode (no semantic cache, no hybrid
retrieval or RRF, no early-exit, just dense top-5 chunks dumped into the prompt) to
get the baseline numbers for the comparison table in the README.

Basically what DocuQuery looked like before we added caching, hybrid search and the
early-exit gate.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import DocuQueryEngine
from eval.run_eval import evaluate_faithfulness_and_relevance

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "benchmark.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "baseline_naive_scores.json")


def run_naive_baseline():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # only the answerable questions - naive mode has no abstention logic so
    # comparing it against unanswerable/adversarial cases isn't meaningful
    cases = [c for c in cases if c["category"] not in ("unanswerable", "adversarial")]

    engine = DocuQueryEngine()

    latencies, costs, faithfulness_scores = [], [], []

    print(f"Running naive RAG baseline across {len(cases)} cases (no cache, no hybrid, no early-exit)...")
    for idx, item in enumerate(cases, 1):
        res = engine.query(item["question"], naive_mode=True)
        time.sleep(1)  # stay under groq's per-minute rate limit
        f_score, _ = evaluate_faithfulness_and_relevance(res["answer"], item["ground_truth"], False, False)
        latencies.append(res["latency_seconds"])
        costs.append(res["cost_usd"])
        faithfulness_scores.append(f_score)
        print(f"[{idx:02d}/{len(cases)}] latency={res['latency_seconds']:.2f}s cost=${res['cost_usd']:.6f} f={f_score:.2f}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "naive_rag_no_cache_no_hybrid",
        "total_cases": len(cases),
        "avg_faithfulness": round(float(np.mean(faithfulness_scores)), 3),
        "latency_p50_seconds": round(float(np.percentile(latencies, 50)), 3),
        "latency_p90_seconds": round(float(np.percentile(latencies, 90)), 3),
        "avg_cost_per_request_usd": round(float(np.mean(costs)), 6),
    }

    print("\n" + "=" * 50)
    print("NAIVE RAG BASELINE RESULTS:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")
    return summary


if __name__ == "__main__":
    run_naive_baseline()
