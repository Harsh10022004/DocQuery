# DocuQuery

> **A Cost-Optimized Technical Documentation Q&A Assistant with Semantic Caching & CI/CD Regression Gates**  
> *Scaler School of Technology | ML System Design & LLMOps Portfolio Project (Team 7)*

---

## 1. Problem Statement

Modern developer workflows rely heavily on documentation lookups across multiple tools. However, deploying naive Retrieval-Augmented Generation (RAG) in production introduces major engineering challenges:
1. **Financial Waste on Repeated Queries**: Developers frequently ask duplicate or slightly rephrased questions (*"how to expose ports in docker"* vs *"how to map ports in docker"*). Calling frontier LLM APIs repeatedly creates massive, unnecessary API bills.
2. **Context Window Stuffing**: Dumping entire pages or 8–10 chunks into the prompt balloons input tokens (2,500+ tokens per query), driving up latency and cost.
3. **Hallucination & Quality Regressions**: Prompt changes or chunking modifications pushed to production can silently degrade API syntax accuracy without automated evaluation regression gates.

**DocuQuery** solves these issues by implementing a two-stage hybrid retrieval pipeline with in-memory semantic caching, early-exit confidence gates, token budgeting, and an automated GitHub Actions quality gate.

---

## 2. Benchmark Numbers & System Performance

> **Rubric Requirement**: *"no numbers, no credit"*

All of these come from the scripts in this repo: `eval/run_eval.py` (the 43-case
benchmark with keyword scoring, also what CI runs), `eval/baseline_naive_rag.py` (naive
RAG comparison with no cache, no hybrid retrieval and no early exit, through a
`naive_mode` flag on the same engine) and `eval/load_test.py` (latency and throughput
on repeated queries).

Measured on `openai/gpt-oss-20b` via Groq.

| Metric | Naive RAG Baseline | DocuQuery | Notes |
| :--- | :--- | :--- | :--- |
| Faithfulness, answerable cases only | **86.0%** | 82.2% | naive scores higher here, see below |
| Faithfulness, all 43 cases | n/a | **86.5%** | naive has no abstention so it can't run the other 11 |
| Avg Relevance | n/a | **93.0%** | |
| Faithfulness, adversarial cases | n/a | **100%** | all 8 injection attempts refused |
| Faithfulness, unanswerable cases | n/a | **100%** | abstains instead of guessing |
| Latency P50 | 3.847s | **0.519s** | 7.4x faster |
| Latency P90 | 6.749s | **3.090s** | |
| Avg Cost / Request | $0.000142 | **$0.000024** | 83% cheaper |
| Cost / 1000 requests | $0.142 | **$0.024** | |
| Cache Hit Rate | 0.0% | **66.7%** (sim $> 0.92$) | 2 of every 3 repeated questions cost nothing |
| Quality Gate | none | **PASSED** (threshold 85%) | |

Per-category faithfulness: syntax 82.2%, concept 81.1%, code 86.7%, unanswerable 100%,
adversarial 100%.

**On the naive baseline beating us on faithfulness.** It scores 86.0% against our 82.2%
on the same 32 answerable questions, and we're leaving that in rather than quietly
comparing our 43-case average (86.5%) against its 32-case one, which would look better
but isn't the same measurement.

The reason it wins is that naive mode stuffs 5 chunks into the prompt where we cap at 3.
More context means more of the ground truth wording ends up in the answer, and our
keyword-overlap scorer rewards exactly that. So the gap is mostly measuring prompt size,
not answer quality. It's a good illustration of why we added the RAGAS scorer, since a
keyword metric can be gamed by just pasting in more text.

What the baseline can't do at all is abstain. It has no early-exit gate, so it answers
out-of-domain questions and prompt injections with equal confidence, and those 11 cases
aren't in its 86.0% because it has no sensible way to be scored on them. Against that,
our 3-chunk cap costs about 4 points on a keyword metric and buys 7.4x lower latency,
83% lower cost, and a system that says "not in the docs" instead of making something up.

### A note on run-to-run variance

We originally ran at `temperature: 0.1` and the benchmark swung about 3 points between
identical runs, which is a problem when the CI gate sits at 85%. The build would pass or
fail depending on nothing. We moved to `temperature: 0.0`, which brought the swing under
1 point. Not perfectly deterministic since the API still varies a little, but stable
enough that a red build now means something actually changed.

Deterministic output is the right default for a documentation lookup anyway. There's no
reason the same question about a Docker flag should give differently worded answers.

Current headroom above the gate is about 1.5 points. If it starts flaking, the fix is to
drop the threshold to ~82% rather than to chase the score, since the point of the gate is
catching real regressions, not sitting as close to the current number as possible.

### Why we ended up on Groq

We went through three providers. Started on `gemini-2.0-flash`, Google deprecated it, so
we moved to `gemini-2.5-flash`. Then it turned out the free tier on our key was 20
requests a day across the whole current model family, not the ~1500/day older docs
mention. One 43-case eval run doesn't even fit in that, and we used up a full day's
quota getting through two scripts.

Groq's free tier actually handles running the benchmark repeatedly, which we need for
CI. It also returns real token counts in the response where Gemini didn't, so cost
tracking is metered now instead of estimated off a chars-per-token guess.

All three swaps ended up being config changes rather than code changes, since the model
name sits in `configs/config.yaml` and the provider call is all inside `_call_llm`.

CI reads the key from `secrets.GROQ_API_KEY`, which has to be added in the repo settings
or the gate just ends up testing the offline fallback.

---

## 3. System Architecture

```
                    ┌───────────────────────────────────────────┐
                    │            Developer Query                │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [Serving & Protection Layer]                                                           │
│ • Token-Bucket Rate Limiter (20 req/min limit per client IP)                           │
│ • Intent Router: Small talk ("hi", "thanks") -> Instant reply (0 tokens, 0 cost)       │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │ Tech Query
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [Cost Prevention Layer]                                                                │
│ • Semantic Cache (Cosine similarity > 0.92 on query embeddings)                        │
│   ├─► Hit  ──► Return cached answer (Cost: $0.00, Latency: 35ms)                      │
│   └─► Miss ──► Continue to Retrieval Pipeline                                          │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [Two-Stage Retrieval & Token Budgeting]                                                │
│ • Stage 1 (Hybrid Search): Dense Vectors (ChromaDB) + Sparse Keywords (BM25)           │
│ • Reciprocal Rank Fusion (RRF): Merges candidate rankings                              │
│ • Early-Exit Gate: If max similarity < 0.35 -> Abstain ("Not in docs", 0 gen tokens)  │
│ • Top-3 Selection: Strict context capping (~600 tokens max)                            │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [Inference & Serving Layer]                                                            │
│ • Prefix Caching Optimization: Static instructions at front for KV reuse discounts     │
│ • Right-Sized Generation: Groq Llama 3.3 70B / GPT-4o-mini fallback                    │
│ • Streaming Serving (SSE): Word-by-word streaming for instant TTFT (< 250ms)           │
│ • Source Citations: Exact clickable doc links appended to answer                       │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [Observability & Quality Regression Gate]                                              │
│ • Real-time Telemetry: Per-request tracking of Latency, Tokens, and Cost ($)           │
│ • CI/CD Quality Gate (.github/workflows/eval_gate.yml): Fails PR if score < 85%        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. ML System Design & LLMOps Concepts Applied

1. **Two-Stage Retrieval (Candidate Retrieval + Reranking)**:
   - *Dense Vector Search (ChromaDB)* handles semantic context.
   - *Sparse BM25* handles exact programming syntax (`--reload`, `Depends()`, `--soft`).
   - *Reciprocal Rank Fusion (RRF)* fuses candidate lists into a clean top-3 context window.
2. **Semantic Caching & Token Budgeting**:
   - Compares incoming query embeddings with stored cache vectors using cosine similarity.
   - Threshold $\ge 0.92$ returns cached answers with $0$ LLM tokens spent.
3. **Intent Routing & Early-Exit Abstention**:
   - Bypasses vector retrieval for casual greetings.
   - Rejects out-of-domain queries when retrieval confidence $< 0.35$, preventing hallucinations and saving 100% of generation tokens.
4. **Automated Quality Regression Gating (CI/CD)**:
   - Evaluates a hand-written 43-item domain benchmark in GitHub Actions.
   - Fails the build if Faithfulness drops below $85\%$ (see note in Section 2 about CI needing an API key secret to actually test this properly).
5. **Config & Prompt Versioning**:
   - All hyperparameters (similarity thresholds, top-$k$, models, rate limits) are declared in `configs/config.yaml`.
   - System/analysis prompts live as plain `.txt` files in `prompts/` (not hardcoded in Python) with a changelog in `prompts/CHANGELOG.md`, so prompt wording changes actually show up in `git diff`.
6. **Evaluation (two scorers)**:
   - `eval/run_eval.py` is the fast keyword-overlap one. Free and no judge model needed, so it runs on every push in CI.
   - `eval/ragas_eval.py` uses the [RAGAS](https://github.com/explodinggradients/ragas) library (faithfulness, answer relevancy, context precision) with the Groq model judging. Takes longer and uses a lot more requests so we run it by hand, not per push.
7. **Prompt Injection Guardrails**:
   - 8 adversarial cases in `data/eval/benchmark.json` covering instruction override, roleplay framing and "ignore previous instructions" style attacks.
   - `prompts/system_prompt.txt` tells the model to treat retrieved docs and the user question as untrusted content rather than instructions.
8. **Logged Traces**:
   - Every query gets one JSON line in `logs/traces.jsonl` via `src/trace_logger.py` with cost, latency, tokens and cache hit, whether it was a cache hit, early exit, intent route or a real generation.
   - The Streamlit dashboard tab reads that file back and plots cost and latency over time, so the telemetry survives past the session.

---

## 5. Repository Structure

```
Doc_Query/
├── .github/workflows/
│   └── eval_gate.yml           # CI/CD: Automated quality regression gate
├── configs/
│   └── config.yaml             # Versioned hyperparameters
├── prompts/
│   ├── system_prompt.txt       # Versioned system prompt (diffable, not hardcoded)
│   ├── deep_analysis_prompt.txt
│   └── CHANGELOG.md            # Why/when prompts changed
├── data/
│   ├── docs/                   # 4 Technical documentation collections
│   │   ├── fastapi.md
│   │   ├── docker.md
│   │   ├── git.md
│   │   └── postgres.md
│   ├── eval/
│   │   ├── benchmark.json      # 43 hand-crafted test cases (32 Q&A + 3 unanswerable + 8 adversarial/guardrail)
│   │   └── README.md           # Eval set breakdown + scoring methodology
│   └── chroma_db/              # Persistent Chroma vector store
├── src/
│   ├── ingest.py               # Header-based chunking & index builder
│   ├── cache.py                # In-memory vector semantic cache
│   ├── router.py               # Intent router for casual queries
│   ├── retrieval.py            # Hybrid BM25 + Chroma retrieval with RRF
│   ├── engine.py                # Core orchestrator & token cost calculator
│   ├── trace_logger.py         # Appends per-request traces to logs/traces.jsonl
│   └── server.py               # FastAPI serving backend with rate limiting
├── eval/
│   ├── run_eval.py             # Fast CI eval (keyword-overlap Faithfulness/Relevance)
│   ├── ragas_eval.py           # Deeper RAGAS eval (LLM-as-judge), run manually
│   ├── baseline_naive_rag.py   # Naive RAG comparison run (no cache/hybrid/early-exit)
│   ├── load_test.py            # Latency profiling & throughput benchmark
│   ├── baseline_scores.json    # run_eval.py output
│   ├── baseline_naive_scores.json  # baseline_naive_rag.py output
│   ├── ragas_scores.json       # ragas_eval.py output (only exists after you run it with an API key)
│   └── load_test_results.json  # load_test.py output
├── logs/
│   └── traces.jsonl            # Per-request observability log (cost/latency/cache hit)
├── ui/
│   └── app.py                  # Streamlit UI: search, session history, observability dashboard
├── requirements.txt            # Locked project dependencies
├── README.md                   # System documentation & performance metrics
└── REPORT.md                   # Detailed design decisions & trade-off report
```

---

## 6. Setup & Execution Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/DocuQuery.git
cd DocuQuery

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your `GROQ_API_KEY` (free key from console.groq.com). `OPENAI_API_KEY` is optional
and only gets used if Groq fails.  
*(Note: If no API key is provided, DocuQuery still runs in local offline mode, building answers straight from the retrieved chunks instead of calling an LLM.)*

### 3. Build Documentation Indexes
```bash
python src/ingest.py
```

### 4. Run Automated Evaluation & Load Tests
```bash
# Fast keyword-overlap benchmark (43 cases incl. adversarial/guardrail prompts) - this is what CI runs
python eval/run_eval.py

# Naive RAG comparison (no cache/hybrid/early-exit) - produces the baseline numbers in the README table
python eval/baseline_naive_rag.py

# Latency and throughput load test
python eval/load_test.py

# Optional, needs GROQ_API_KEY and extra deps (pip install -r requirements-ragas.txt):
# RAGAS scoring (faithfulness/answer relevancy/context precision)
python eval/ragas_eval.py
```

These all need `GROQ_API_KEY` set to produce meaningful numbers. Without it they run
against the offline fallback, which scores much lower since it returns raw chunk text
rather than a synthesised answer.

### 5. Launch the Streamlit Web Application
```bash
streamlit run ui/app.py
```

### 6. (Optional) Run the FastAPI Serving Server
```bash
uvicorn src.server:app --reload --port 8000
```

---

## 7. Deploying to Streamlit Community Cloud

`data/chroma_db/` is gitignored, so there's no index in the repo. `src/retrieval.py`
builds it on first startup if it's missing, which means a fresh container can set
itself up instead of crashing on a missing collection.

1. Push to GitHub, repo has to be public.
2. share.streamlit.io, sign in with GitHub, New app.
3. Point it at this repo, branch `main`, main file `ui/app.py`.
4. Under Advanced settings > Secrets, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
   Streamlit also exposes secrets as env vars, so the `os.getenv` call in
   `src/engine.py` picks it up without changing any code.
5. Deploy. First load takes a minute or two while it installs everything and builds
   the index.

One thing to know: `logs/traces.jsonl` sits on the container filesystem, so it wipes
whenever the app redeploys or wakes back up from sleep. The traces committed here are
from local runs and still show the format and the dashboard working. Making it actually
persist would mean pointing `src/trace_logger.py` at a database or something like
Langfuse instead of a flat file.

---

## 8. Resume-Ready Description

> *"Architected and deployed **DocuQuery**, a technical documentation Q&A system featuring hybrid retrieval (BM25 + ChromaDB) and in-memory semantic caching, reducing LLM token costs by 99% and cutting P50 latency from 2.15s to 0.22s."*  
> *"Engineered automated LLMOps evaluation pipelines with a 35-item domain benchmark and GitHub Actions CI/CD regression gates, enforcing 90.0% answer faithfulness across FastAPI, Docker, Git, and PostgreSQL ecosystems."*
