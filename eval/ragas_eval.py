"""
RAGAS evaluation (faithfulness, answer relevancy, context precision) using the same
Groq model we serve answers with as the judge.

Separate from run_eval.py because this takes a while and burns through a lot more
requests, so we run it by hand before submission rather than on every CI push. Only
scores a 15 case subset.

Needs GROQ_API_KEY and `pip install -r requirements-ragas.txt`.
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
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("GROQ_API_KEY not set - RAGAS needs a real LLM judge, skipping.")
        print("Set GROQ_API_KEY in .env and re-run: python eval/ragas_eval.py")
        return

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("ragas deps not installed.")
        print("Run: pip install -r requirements-ragas.txt")
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

    judge_model = engine.config["models"]["primary_llm"]
    judge_llm = ChatGroq(model=judge_model, api_key=groq_key, temperature=0)

    # groq has no embeddings endpoint, so answer_relevancy uses the same local
    # MiniLM model the vector store already uses. keeps it free too.
    judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
        "judge_model": judge_model,
    }

    print("\nRAGAS results:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
