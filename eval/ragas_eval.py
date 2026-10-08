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

SUBSET_SIZE = 10  # keep this small, each RAGAS metric = another LLM call per row


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
        from langchain_core.embeddings import Embeddings
    except ImportError as e:
        print(f"ragas deps not installed ({e}).")
        print("Run: pip install -r requirements-ragas.txt")
        return

    class ChromaEmbeddings(Embeddings):
        """Reuses the MiniLM model chroma already has loaded, so we don't pull in
        sentence-transformers (and torch with it) just to score answer_relevancy."""

        def __init__(self, fn):
            self.fn = fn

        def embed_documents(self, texts):
            return [list(v) for v in self.fn(list(texts))]

        def embed_query(self, text):
            return list(self.fn([text])[0])

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
    judge_embeddings = ChromaEmbeddings(engine.retriever.collection._embedding_function)

    # answer_relevancy defaults to strictness=3, which asks the model for 3 completions
    # in one call. groq rejects that with "'n' : number must be at most 1", so every
    # answer_relevancy job errored on our first run and the score came out meaningless.
    answer_relevancy.strictness = 1

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    cols = ["faithfulness", "answer_relevancy", "context_precision"]
    scores = df[cols].mean().to_dict()

    # a job that hits a rate limit or times out comes back as NaN and pandas just
    # skips it in mean(), so without this you can get a confident looking 1.0 that
    # was actually averaged over 3 surviving rows out of 15
    valid = {c: int(df[c].notna().sum()) for c in cols}

    summary = {
        "cases_attempted": len(subset),
        "cases_with_valid_scores": valid,
        "ragas_faithfulness": round(float(scores["faithfulness"]), 3),
        "ragas_answer_relevancy": round(float(scores["answer_relevancy"]), 3),
        "ragas_context_precision": round(float(scores["context_precision"]), 3),
        "judge_model": judge_model,
    }

    print("\nRAGAS results:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    incomplete = [c for c in cols if valid[c] < len(subset)]
    if incomplete:
        print("\nWARNING: these metrics did not score every case, so the averages above")
        print("are over a partial set and shouldn't be quoted as final:")
        for c in incomplete:
            print(f"  {c}: {valid[c]}/{len(subset)} cases scored")
        print("Usually means the API rate limit was hit. Re-run with fresh quota.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
