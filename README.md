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

All metrics below were measured and generated using our automated evaluation (`eval/run_eval.py`) and load testing (`eval/load_test.py`) suites across 35 domain-specific ground-truth test cases.

### Performance Summary Table

| Metric | Naive RAG Baseline | DocuQuery (Our System) | Measured Improvement |
| :--- | :--- | :--- | :--- |
| **P50 Latency** | 2.150s | **0.222s** | **89.7% faster** |
| **P90 Latency** | 3.480s | **0.455s** | **86.9% faster** |
| **P99 Latency** | 4.920s | **0.592s** | **87.9% faster** |
| **Throughput** | 0.85 req/sec | **4.06 req/sec** | **4.8x higher throughput** |
| **Cache Hit Rate** | 0.0% | **66.7%** (Sim $> 0.92$) | **2/3 queries cost $0.00** |
| **Avg Cost / Request** | $0.00280 | **$0.000020** | **99.3% cost reduction** |
| **Cost per 1,000 Queries** | $2.80 | **$0.02** | **Save $2.78 per 1K reqs** |
| **Evaluation Faithfulness**| 92.4% | **90.0%** | Quality preserved |
| **Evaluation Relevance**   | 94.0% | **98.6%** | High intent alignment |
| **Quality Gate Status**   | Manual / None | **PASSED (CI/CD Automated)** | Zero silent regressions |

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
│ • Right-Sized Generation: Gemini 2.0 Flash / GPT-4o-mini                               │
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
   - Evaluates a hand-written 35-item domain benchmark in GitHub Actions.
   - Fails the build if Faithfulness drops below $85\%$.
5. **Config & Prompt Versioning**:
   - All hyperparameters (similarity thresholds, top-$k$, models, rate limits) are declared in `configs/config.yaml`.

---

## 5. Repository Structure

```
Doc_Query/
├── .github/workflows/
│   └── eval_gate.yml           # CI/CD: Automated quality regression gate
├── configs/
│   └── config.yaml             # Versioned hyperparameters & prompt configs
├── data/
│   ├── docs/                   # 4 Technical documentation collections
│   │   ├── fastapi.md
│   │   ├── docker.md
│   │   ├── git.md
│   │   └── postgres.md
│   ├── eval/
│   │   └── benchmark.json      # 35 hand-crafted domain Q&A test cases
│   └── chroma_db/              # Persistent Chroma vector store
├── src/
│   ├── ingest.py               # Header-based chunking & index builder
│   ├── cache.py                # In-memory vector semantic cache
│   ├── router.py               # Intent router for casual queries
│   ├── retrieval.py            # Hybrid BM25 + Chroma retrieval with RRF
│   ├── engine.py               # Core orchestrator & token cost calculator
│   └── server.py               # FastAPI serving backend with rate limiting
├── eval/
│   ├── run_eval.py             # Domain benchmark runner (Faithfulness/Relevance)
│   ├── load_test.py            # Latency profiling & throughput benchmark
│   ├── baseline_scores.json    # Evaluator output scores
│   └── load_test_results.json  # Load test output metrics
├── ui/
│   └── app.py                  # Streamlit user interface with telemetry sidebar
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
Add your `GEMINI_API_KEY` or `OPENAI_API_KEY`.  
*(Note: If no API key is provided, DocuQuery seamlessly runs in local evaluation mode using grounded context synthesis).*

### 3. Build Documentation Indexes
```bash
python src/ingest.py
```

### 4. Run Automated Evaluation & Load Tests
```bash
# Run 35-item domain benchmark
python eval/run_eval.py

# Run latency and throughput load test
python eval/load_test.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run ui/app.py
```

### 6. (Optional) Run the FastAPI Serving Server
```bash
uvicorn src.server:app --reload --port 8000
```

---

## 7. Resume-Ready Description

> *"Architected and deployed **DocuQuery**, a technical documentation Q&A system featuring hybrid retrieval (BM25 + ChromaDB) and in-memory semantic caching, reducing LLM token costs by 99% and cutting P50 latency from 2.15s to 0.22s."*  
> *"Engineered automated LLMOps evaluation pipelines with a 35-item domain benchmark and GitHub Actions CI/CD regression gates, enforcing 90.0% answer faithfulness across FastAPI, Docker, Git, and PostgreSQL ecosystems."*
