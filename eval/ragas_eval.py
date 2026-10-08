"""
Real RAGAS evaluation, separate from eval/run_eval.py.

run_eval.py is our fast keyword-overlap check that runs in CI on every push -
it's cheap and doesn't need an API key. This script does an actual RAGAS pass
(faithfulness, answer relevancy, context precision) using Gemini Flash as the
judge model, which costs real API credits and takes longer, so we run it
manually before submission instead of on every CI run.

Only scores a subset of the benchmark (15 cases) to keep this inside the
~$5-20 total API budget mentioned in the project handout.

Needs GEMINI_API_KEY set and `pip install -r requirements-ragas.txt`.
"""
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import DocuQueryEngine

BENCHMARK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "benchmark.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "ragas_scores.json")

SUBSET_SIZE = 15  # keep this small, each RAGAS metric = another LLM call per row


def build_eval_rows(engine, cases):
    rows = []
    for item in cases:
        res = engine.query(item["question"])
        contexts = [c["excerpt"] for c in res.get("citations", [])] or [res["answer"]]
        rows.append({
            "question": item["question"],
            "answer": res["answer"],
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
    return rows


def main():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("GEMINI_API_KEY not set - RAGAS eval needs a real LLM judge, skipping.")
        print("Set GEMINI_API_KEY in .env and re-run: python eval/ragas_eval.py")
        return

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError:
        print("ragas / langchain-google-genai / datasets not installed.")
        print("Run: pip install ragas langchain-google-genai datasets")
        return

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    # only use the normal Q&A cases (skip unanswerable/adversarial - RAGAS's
    # metrics assume a grounded answer exists, they don't fit abstention cases)
    answerable_cases = [c for c in all_cases if c["category"] not in ("unanswerable", "adversarial")]
    subset = answerable_cases[:SUBSET_SIZE]

    print(f"Running RAGAS on {len(subset)} cases (out of {len(all_cases)} total benchmark items)...")
    engine = DocuQueryEngine()
    rows = build_eval_rows(engine, subset)
    dataset = Dataset.from_list(rows)

    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_key, temperature=0)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores = result.to_pandas()[["faithfulness", "answer_relevancy", "context_precision"]].mean().to_dict()
    summary = {
        "cases_scored": len(subset),
        "ragas_faithfulness": round(float(scores["faithfulness"]), 3),
        "ragas_answer_relevancy": round(float(scores["answer_relevancy"]), 3),
        "ragas_context_precision": round(float(scores["context_precision"]), 3),
        "judge_model": "gemini-2.0-flash",
    }

    print("\nRAGAS results:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
