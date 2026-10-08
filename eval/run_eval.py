import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import DocuQueryEngine

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "benchmark.json")
BASELINE_OUTPUT = os.path.join(os.path.dirname(__file__), "baseline_scores.json")


def evaluate_faithfulness_and_relevance(generated: str, ground_truth: str, is_unanswerable: bool):
    """
    Computes programmatic evaluation scores for domain benchmark:
    - Faithfulness: checks keyword grounding against ground truth facts.
    - Relevance: verifies the model addressed the question domain or safely abstained.
    """
    gen_lower = generated.lower()
    gt_lower = ground_truth.lower()

    if is_unanswerable:
        # for unanswerable questions, model is faithful if it refused/abstained
        refusal_phrases = ["not covered", "cannot find", "not present", "does not cover", "outside"]
        abstained = any(p in gen_lower for p in refusal_phrases)
        return (1.0 if abstained else 0.0), (1.0 if abstained else 0.0)

    # extract significant words from ground truth (length > 3)
    gt_words = [w.strip(".,;:()[]{}\"'") for w in gt_lower.split() if len(w) > 3]
    if not gt_words:
        return 1.0, 1.0

    matches = sum(1 for w in gt_words if w in gen_lower)
    score = matches / len(gt_words)
    
    # normalized scores
    faithfulness = min(1.0, score * 1.3)  # lenient coverage of key concepts
    relevance = 1.0 if len(generated) > 25 and matches >= 2 else (0.5 if matches >= 1 else 0.0)

    return round(faithfulness, 3), round(relevance, 3)


def run_benchmark():
    print(f"Loading benchmark suite from: {BENCHMARK_FILE}")
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    engine = DocuQueryEngine()
    
    total = len(cases)
    faithfulness_scores = []
    relevance_scores = []
    latencies = []
    total_cost = 0.0

    print(f"Executing automated evaluation across {total} test questions...\n")

    for idx, item in enumerate(cases, 1):
        q = item["question"]
        gt = item["ground_truth"]
        cat = item["category"]
        is_unans = (cat == "unanswerable")

        res = engine.query(q)
        ans = res["answer"]
        lat = res["latency_seconds"]
        cost = res["cost_usd"]

        f_score, r_score = evaluate_faithfulness_and_relevance(ans, gt, is_unans)
        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)
        latencies.append(lat)
        total_cost += cost

        print(f"[{idx:02d}/{total}] Domain: {item['domain']:<7} | F-Score: {f_score:.2f} | R-Score: {r_score:.2f} | Latency: {lat:.2f}s")

    avg_faithfulness = round(sum(faithfulness_scores) / total, 3)
    avg_relevance = round(sum(relevance_scores) / total, 3)
    avg_latency = round(sum(latencies) / total, 3)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": total,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_latency_seconds": avg_latency,
        "total_evaluation_cost_usd": round(total_cost, 6),
        "target_threshold_met": (avg_faithfulness >= 0.85 and avg_relevance >= 0.85)
    }

    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY:")
    print(f"Average Faithfulness Score: {avg_faithfulness * 100:.1f}%")
    print(f"Average Relevance Score:    {avg_relevance * 100:.1f}%")
    print(f"Average Latency:            {avg_latency:.2f}s")
    print(f"Total Eval Cost:            ${total_cost:.5f}")
    print(f"Quality Gate Status:        {'PASSED' if summary['target_threshold_met'] else 'FAILED'}")
    print("="*50)

    with open(BASELINE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved baseline evaluation results to: {BASELINE_OUTPUT}")

    return summary


if __name__ == "__main__":
    run_benchmark()
