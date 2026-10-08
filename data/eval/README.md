# Evaluation Set

File: `benchmark.json` - 43 test cases, all written by us by going through the four
docs we indexed and picking out the things people actually look up (exact flags,
syntax, and the usual "difference between X and Y" questions). Ground truth was
written from the docs themselves.

## Breakdown

| Category | Count | What it tests |
| :--- | :--- | :--- |
| `syntax` | 18 | Exact commands/flags like `--reload`, `Depends()`, `git stash pop` |
| `concept` | 13 | Explanations - merge vs rebase, when to use a GIN index |
| `code` | 1 | Answer should contain a working code snippet |
| `unanswerable` | 3 | Topics we deliberately did not index (Kubernetes, Django, React). System should abstain |
| `adversarial` | 8 | Prompt injection attempts - "ignore all previous instructions", fake SYSTEM OVERRIDE headers, roleplay framing |

By domain: FastAPI 11, Docker 9, Git 8, PostgreSQL 5, general/out-of-domain 10.

Fields per case: `id`, `domain`, `question`, `ground_truth`, `category`.

## Scoring

Two scorers, used for different things.

### 1. Keyword overlap - `eval/run_eval.py`

The fast one. Runs in GitHub Actions on every push, no judge model needed so CI stays
free.

Faithfulness takes the words longer than 3 chars from the ground truth and checks what
fraction show up in the generated answer, scaled by 1.3 and capped at 1.0 (kept it
lenient since an answer can be right but worded differently). Relevance is 1.0 if the
answer is over 25 chars and hits 2+ ground truth keywords, 0.5 for 1 keyword, else 0.

The `unanswerable` and `adversarial` cases can't be scored this way since there's no
correct answer to match against. For unanswerable we check the system actually
abstained by looking for refusal phrases ("not covered", "does not cover"). For
adversarial we check it did not comply with the injection - looking for markers like
"DAN" or "no restrictions" which score 0.

CI gate passes only if average faithfulness and relevance are both >= 0.85. We also
save a per-category breakdown into `eval/baseline_scores.json` so a drop in one
category doesn't get hidden by the average.

Limitation: keyword matching is crude. An answer can be correct and still score low if
it uses different words. That's the reason for the second scorer.

### 2. RAGAS - `eval/ragas_eval.py`

Uses the RAGAS library with `openai/gpt-oss-120b` as the judge, scoring faithfulness, answer
relevancy and context precision. Groq has no embeddings endpoint, so answer relevancy
uses the same local MiniLM model the vector store already uses.

Takes longer and fires off several requests per case, so we run this manually before
submission instead of on every push. Only scores the answerable cases since RAGAS
metrics assume a grounded answer exists. Defaults to a 10 case subset.

Needs `GROQ_API_KEY` and `pip install -r requirements-ragas.txt`. Writes to
`eval/ragas_scores.json`.

## Other scripts using this file

`eval/baseline_naive_rag.py` runs the same questions through a deliberately naive
pipeline (no cache, no hybrid retrieval, no early exit, top 5 chunks dumped in) to get
the before/after numbers in the README.

`eval/load_test.py` doesn't use this file, it has its own repeated query list for
measuring latency percentiles and cache hit rate.
