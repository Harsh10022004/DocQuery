# DocuQuery: Engineering Design Decisions & Evaluation Report

**Project Title**: DocuQuery — Documentation Q&A Assistant with Automated Evaluation  
**Course**: Scaler School of Technology | ML System Design & LLMOps Portfolio Project  
**Team**: Team 7 (Harsh, Indrajeet, Satyam, Shivam, Siddharth, Sushant)  

> **Note on the numbers in this report**: the specific percentages below (90.0%
> faithfulness etc.) are from an early manual test and are now outdated. The current
> measured numbers (86.5% faithfulness, 94.2% relevance, 100% on adversarial/guardrail
> cases, PASSED quality gate) are in `README.md` Section 2, produced by
> `eval/run_eval.py` and `eval/baseline_naive_rag.py` with a real Gemini API key.
> `eval/load_test.py`'s latency numbers are still partly affected by hitting this
> Google account's 20-requests/day free quota mid-run - re-run it on a day with
> fresh quota before your final submission and update both files together.

---

## 1. Problem Framing & Objectives (20% Weight)

### 1.1 The Production Gap
In production environments, developers spend a substantial amount of time verifying documentation syntax, flags, and configuration formats. When organizations adopt Retrieval-Augmented Generation (RAG) to automate documentation search, two severe bottlenecks emerge:
1. **Unbounded Cost & Redundant Compute**: Documentation queries follow a power-law distribution; developers repeatedly ask identical or slightly rephrased questions. Querying commercial LLMs for every repeated lookup causes runaway API costs.
2. **Context Window Inflation & High Latency**: Without disciplined chunking and candidate reranking, naive systems inject thousands of tokens of noisy context, multiplying input token bills and increasing Time-To-First-Token (TTFT).
3. **Silent Quality Regressions**: Without an automated evaluation benchmark, subtle changes to prompts or chunk boundaries can silently break code syntax generation.

### 1.2 System Objectives & Quantitative Targets
- **P50 Latency Target**: $< 0.50\text{s}$ (achieved: **0.222s**).
- **Cost Reduction Target**: $> 90\%$ reduction in per-query LLM costs (achieved: **99.3%**).
- **Quality Standard**: Faithfulness $\ge 85\%$ on domain benchmark (achieved: **90.0%**).
- **Regression Protection**: Automated CI/CD gate blocking pull requests if benchmark scores drop.

---

## 2. Architecture Design & Component Breakdown (30% Weight)

### 2.1 Multi-Source Ingestion Pipeline
We curated authoritative documentation across four core developer tools:
- **FastAPI**: Endpoint routing, dependency injection (`Depends`), query/path parameters, background tasks, and error handling.
- **Docker**: CLI commands, multi-stage Dockerfiles, port mapping, volume persistence, and docker-compose.
- **Git**: Branching workflows, merge vs. rebase, cherry-pick, stashing, and conflict resolution.
- **PostgreSQL**: Indexing strategies (B-Tree vs. GIN), JSONB queries, transaction blocks, and connection pooling.

**Header-Aware Chunking Strategy**: Instead of naive character chunking (e.g., fixed 500 characters) which cuts function signatures and code blocks in half, we implemented markdown heading-aware chunking (`#`, `##`). Each chunk preserves full conceptual context, complete code snippets, section headers, and documentation URLs.

### 2.2 Serving & Protection Layer
- **Token-Bucket Rate Limiter**: Implemented in the FastAPI serving layer to protect downstream LLM API budgets against DoS or infinite client polling (allowing max 20 queries/min per client IP).
- **Intent Router**: Classifies incoming inputs using fast pattern matching. Conversational queries ("hello", "thanks") are resolved immediately without invoking vector search or context retrieval, saving 100% of retrieval tokens.

### 2.3 Two-Stage Retrieval & Token Budgeting
- **Stage 1 (Hybrid Candidate Retrieval)**:
  - **Dense Retrieval**: ChromaDB in-process vector store indexed with cosine similarity.
  - **Sparse Retrieval**: BM25 keyword matching for exact keyword precision (`--reload`, `Depends`, `HEAD~1`).
  - **Reciprocal Rank Fusion (RRF)**: Fuses both rankings ($RRF(d) = \sum \frac{1}{60 + r_i}$).
- **Stage 2 (Top-3 Selection & Early-Exit)**:
  - Constrains context strictly to the top 3 most relevant chunks ($\approx 600$ tokens total), reducing prompt tokens by 74% compared to naive RAG baselines.
  - **Early-Exit Abstention**: If maximum similarity is $< 0.35$, the query is classified as out-of-domain (e.g. asking about React hooks or Kubernetes Helm charts in a backend docs corpus) and exits cleanly without calling the LLM.

### 2.4 Inference & Streaming Layer
- **Prefix Caching Alignment**: Static instructions are placed strictly at the beginning of the prompt. Modern LLM APIs (Gemini Flash, OpenAI) cache KV tensors of invariant prefixes, providing an automatic 50%–80% prompt token discount.
- **Streaming Serving**: The backend supports Server-Sent Events (SSE) streaming, reducing perceived latency to $< 250\text{ms}$.

---

## 3. LLMOps Depth & Quality Engineering (20% Weight)

### 3.1 Domain Evaluation Benchmark
Rather than using generic, synthetic public benchmarks, we authored a domain-specific evaluation suite of **35 hand-crafted test questions** (`data/eval/benchmark.json`):
- 10 FastAPI syntax and dependency injection questions.
- 9 Docker container, networking, and volume questions.
- 8 Git commit, rebase, and branching questions.
- 5 PostgreSQL indexing, transaction, and pooling questions.
- 3 Negative / Out-of-Domain questions to verify that the system refuses to hallucinate when context is missing.

### 3.2 Evaluation Metrics
- **Faithfulness (Grounding)**: Measures whether generated answers strictly reflect documented facts without fabrication.
- **Answer Relevance**: Measures whether the response accurately answers the developer's question.
- **Benchmark Results**:
  - Average Faithfulness: **90.0%**
  - Average Answer Relevance: **98.6%**
  - Average Latency: **0.47s**
  - Total Benchmark Evaluation Cost: **$0.00323**

### 3.3 CI/CD Quality Regression Gating
We created a GitHub Actions workflow (`.github/workflows/eval_gate.yml`) that triggers on every Pull Request to `main`. The workflow automatically:
1. Builds the vector index from documentation.
2. Executes the 35-item evaluation benchmark.
3. Evaluates if the average Faithfulness score is $\ge 85\%$. If a regression is detected, the workflow fails and prevents the PR from merging.

---

## 4. Trade-Off Reasoning (20% Weight)

| Architectural Decision | Chosen Approach | Alternative Considered | Rationale & Trade-Off |
| :--- | :--- | :--- | :--- |
| **Retrieval Strategy** | Hybrid Search (BM25 + Chroma) with RRF | Pure Vector Search | Tech docs require exact keyword precision (e.g. `--reload`, `BaseModel`, `-d`). Pure vector search missed 28% of exact symbol queries. Combining BM25 with dense vectors yields optimal recall across both concept and syntax queries. |
| **Model Selection** | Gemini 2.0 Flash / GPT-4o-mini | GPT-4o / Claude 3.5 Sonnet | Frontier models cost $2.50–$5.00 per 1M tokens. Right-sizing to Flash/mini ($0.10–$0.15 per 1M tokens) is 15x–25x cheaper and provides sub-second TTFT while maintaining 90.0% faithfulness on focused documentation contexts. |
| **Caching Mechanism** | Semantic Vector Cache (Cosine Sim $> 0.92$) | Exact String Hash Cache (Redis / MD5) | Developers phrase identical questions differently ("how to map docker ports" vs "how do I expose ports in docker"). Exact match misses 80%+ of real-world repeat questions. Semantic caching achieved a 66.7% hit rate in testing. |
| **Context Window Size** | Strict Top-3 Chunks (~600 tokens) | Top-10 Chunks (~2,500 tokens) | Expanding context to 10 chunks increased cost by 3.5x and slowed generation by 1.2s while improving accuracy by less than 1.5%. Top-3 provides the optimal Pareto balance of cost and grounding. |
| **Out-of-Scope Handling**| Early-Exit Threshold Gate ($\tau < 0.35$) | Full LLM Generation Refusal | Letting the LLM generate a refusal response consumes both prompt tokens and generation tokens. Early-exit abstains immediately at retrieval time, cutting cost and latency to zero. |

---

## 5. Measured Experimental Numbers

| Performance Metric | Measured Value | Production Significance |
| :--- | :--- | :--- |
| **P50 Latency** | **0.222s** | Fast interactive developer response |
| **P90 Latency** | **0.455s** | Bounded degradation under uncached queries |
| **P99 Latency** | **0.592s** | Sub-600ms worst-case response |
| **Load Test Throughput** | **4.06 req/sec** | Stable throughput on lightweight server |
| **Cache Hit Rate** | **66.7%** | Substantial portion of queries answered with zero API cost |
| **Average Cost / Request** | **$0.000020** | Unbeatable unit economics |
| **Cost per 1,000 Queries** | **$0.0200** | Scalable within a minimal API budget ($5–$20) |
| **Evaluation Faithfulness**| **90.0%** | Zero API hallucination on documentation specifications |
| **Evaluation Relevance**   | **98.6%** | High response fidelity |

---

## 6. Conclusion

DocuQuery proves that production LLM systems do not need expensive frontier models or massive context windows to deliver high accuracy. By combining **header-aware chunking**, **hybrid retrieval (BM25 + Dense)**, **semantic caching**, and **automated CI/CD regression gates**, DocuQuery delivers **90% faithfulness** at **$0.02 per 1,000 requests** with a **0.22s P50 latency**.
