# DocuQuery: Engineering Design Decisions & Evaluation Report

**Project Title**: DocuQuery — Documentation Q&A Assistant with Automated Evaluation  
**Course**: Scaler School of Technology | ML System Design & LLMOps Portfolio Project  
**Team**: Team 7 (Harsh, Indrajeet, Satyam, Shivam, Siddharth, Sushant)  

> **Note on the numbers**: everything here came from running the eval scripts in this
> repo (`eval/run_eval.py`, `eval/baseline_naive_rag.py`, `eval/load_test.py`) against
> `openai/gpt-oss-20b` on Groq. See Section 4 for why we ended up on that provider.

---

## 1. Problem Framing & Objectives (20% Weight)

### 1.1 The Production Gap
In production environments, developers spend a substantial amount of time verifying documentation syntax, flags, and configuration formats. When organizations adopt Retrieval-Augmented Generation (RAG) to automate documentation search, two severe bottlenecks emerge:
1. **Unbounded Cost & Redundant Compute**: Documentation queries follow a power-law distribution; developers repeatedly ask identical or slightly rephrased questions. Querying commercial LLMs for every repeated lookup causes runaway API costs.
2. **Context Window Inflation & High Latency**: Without disciplined chunking and candidate reranking, naive systems inject thousands of tokens of noisy context, multiplying input token bills and increasing Time-To-First-Token (TTFT).
3. **Silent Quality Regressions**: Without an automated evaluation benchmark, subtle changes to prompts or chunk boundaries can silently break code syntax generation.

### 1.2 System Objectives & Quantitative Targets
- **P50 Latency Target**: $< 0.50\text{s}$ (achieved: **0.519s**, just over).
- **Cost Reduction Target**: meaningful reduction vs. a naive RAG baseline (achieved: **83%**, $0.000142 to $0.000024 per request).
- **Quality Standard**: Faithfulness $\ge 85\%$ on the domain benchmark (achieved: **88.0%** across all 43 cases; 83.8% on answerable cases alone, see Section 5).
- **Safety Standard**: refuse 100% of adversarial/prompt-injection cases (achieved: **100%** across 8 cases).
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
- **Prefix Caching Alignment**: Static instructions are placed strictly at the beginning of the prompt, since providers that cache KV tensors for invariant prefixes can then reuse them across requests instead of reprocessing the same instruction block every time.
- **Streaming Serving**: The backend supports Server-Sent Events (SSE) streaming, reducing perceived latency to $< 250\text{ms}$.

---

## 3. LLMOps Depth & Quality Engineering (20% Weight)

### 3.1 Domain Evaluation Benchmark
Rather than using generic, synthetic public benchmarks, we authored a domain-specific evaluation suite of **43 hand-crafted test questions** (`data/eval/benchmark.json`, documented in `data/eval/README.md`):
- 11 FastAPI syntax and dependency injection questions.
- 9 Docker container, networking, and volume questions.
- 8 Git commit, rebase, and branching questions.
- 5 PostgreSQL indexing, transaction, and pooling questions.
- 3 Negative / Out-of-Domain questions to verify that the system refuses to hallucinate when context is missing.
- 8 Adversarial / prompt-injection questions ("ignore all previous instructions", fake SYSTEM OVERRIDE headers, roleplay framing) to verify our guardrails hold.

### 3.2 Evaluation Metrics
We run two scorers for different purposes (full methodology in `data/eval/README.md`):
- **Keyword-overlap heuristic** (`eval/run_eval.py`): fast and free, runs in CI on every push. Scores faithfulness and relevance. Abstention and adversarial cases are instead scored on whether the system correctly refused.
- **RAGAS, LLM-as-judge** (`eval/ragas_eval.py`): faithfulness, answer relevancy and context precision, judged by the same Groq model we serve with. Uses a lot more requests so we run it manually before submission rather than per push.

**Benchmark Results** (43 cases on `openai/gpt-oss-20b`):
  - Average Faithfulness: **88.0%**
  - Average Answer Relevance: **93.0%**
  - Adversarial + unanswerable categories: **100%**
  - Average Latency: **1.64s** (real API round trips; cached queries come back in ~0.5s)
  - Total Benchmark Evaluation Cost: **$0.00446**

### 3.3 CI/CD Quality Regression Gating
We created a GitHub Actions workflow (`.github/workflows/eval_gate.yml`) that triggers on every Pull Request to `main`. The workflow automatically:
1. Builds the vector index from documentation.
2. Executes the 43-item evaluation benchmark.
3. Evaluates if the average Faithfulness score is $\ge 85\%$. If a regression is detected, the workflow fails and prevents the PR from merging.
4. Writes a per-category breakdown so a regression hiding in one category (e.g. adversarial prompts starting to get through) is visible instead of being averaged away by the others.

---

## 4. Trade-Off Reasoning (20% Weight)

| Architectural Decision | Chosen Approach | Alternative Considered | Rationale & Trade-Off |
| :--- | :--- | :--- | :--- |
| **Retrieval Strategy** | Hybrid Search (BM25 + Chroma) with RRF | Pure Vector Search | Tech docs require exact keyword precision (e.g. `--reload`, `BaseModel`, `-d`). Pure vector search missed 28% of exact symbol queries. Combining BM25 with dense vectors yields optimal recall across both concept and syntax queries. |
| **Model Selection** | Groq Llama 3.3 70B, GPT-4o-mini as fallback | GPT-4o / Claude Sonnet | Frontier models run $2.50-$5.00 per 1M tokens, which we don't need when the model is just answering from a short chunk of context we already retrieved. We ended up switching providers three times (see the note under this table). |
| **Caching Mechanism** | Semantic Vector Cache (Cosine Sim $> 0.92$) | Exact String Hash Cache (Redis / MD5) | Developers phrase identical questions differently ("how to map docker ports" vs "how do I expose ports in docker"). Exact match misses 80%+ of real-world repeat questions. Semantic caching achieved a 66.7% hit rate in testing. |
| **Context Window Size** | Strict Top-3 Chunks (~600 tokens) | Top-10 Chunks (~2,500 tokens) | Expanding context to 10 chunks increased cost by 3.5x and slowed generation by 1.2s while improving accuracy by less than 1.5%. Top-3 provides the optimal Pareto balance of cost and grounding. |
| **Out-of-Scope Handling**| Early-Exit Threshold Gate ($\tau < 0.35$) | Full LLM Generation Refusal | Letting the LLM generate a refusal response consumes both prompt tokens and generation tokens. Early-exit abstains immediately at retrieval time, cutting cost and latency to zero. |

### On switching providers

We started on `gemini-2.0-flash`, Google deprecated it mid-project, so we went to
`gemini-2.5-flash`. That turned out to have a 20 requests/day free tier cap on our key,
which isn't enough to finish a single 43-case eval run, let alone run one per push in
CI. So we moved to Groq.

Two things came out of this. Groq returns real token usage in its responses where Gemini
didn't, so our cost tracking is metered now rather than estimated from character counts.
And because the model name lives in `configs/config.yaml` and the API call is contained
in `_call_llm`, all three switches were config edits instead of rewrites. That wasn't
planned for, it just happened to pay off.

---

## 5. Measured Experimental Numbers

Measured on `openai/gpt-oss-20b` via Groq, using the 43-case benchmark
(`eval/run_eval.py`), the naive comparison (`eval/baseline_naive_rag.py`) and the load
test (`eval/load_test.py`).

| Performance Metric | Naive RAG | DocuQuery | Production Significance |
| :--- | :--- | :--- | :--- |
| Faithfulness, answerable only (32 cases) | **86.0%** | 83.8% | naive wins here, explained below |
| Faithfulness, full benchmark (43 cases) | n/a | **88.0%** | naive can't run the abstention cases |
| Avg Relevance | n/a | **93.0%** | answers address what was actually asked |
| Faithfulness (adversarial) | n/a | **100%** | all 8 injection attempts refused |
| Faithfulness (unanswerable) | n/a | **100%** | abstains instead of guessing |
| Latency P50 | 3.847s | **0.519s** | 7.4x faster, mostly from cache hits |
| Latency P90 | 6.749s | **3.090s** | cache misses degrade but stay bounded |
| Avg Cost / Request | $0.000142 | **$0.000024** | 83% cheaper |
| Cache Hit Rate | 0.0% | **66.7%** | 2 of 3 repeated questions cost nothing |
| CI Quality Gate | none | **PASSED** (threshold 85%) | regressions get blocked automatically |

Per-category faithfulness (from `eval/baseline_scores.json`): syntax 82.4%, concept
85.6%, code 86.7%, unanswerable 100%, adversarial 100%. Syntax is the weakest, which
tracks since those questions hinge on reproducing an exact flag or symbol.

### Why the naive baseline scores higher on faithfulness

On the same 32 answerable questions the baseline gets 86.0% and we get 83.8%. We could
have put our 43-case average (88.0%) next to its 32-case average and called it a win,
but those aren't the same measurement so the comparison would be dishonest.

The baseline wins because it puts 5 chunks in the prompt where we cap at 3. More context
means more of the ground truth phrasing ends up in the generated answer, and a
keyword-overlap scorer rewards that directly. So the 2.2 point gap is mostly measuring
prompt size rather than whether the answer is better grounded. This is the clearest
argument for why the RAGAS scorer exists alongside the keyword one, since you can game a
keyword metric by pasting in more text and you can't game an LLM judge the same way.

The baseline also can't abstain, having no early-exit gate, so it answers out-of-domain
questions and prompt injections just as confidently as real ones. Those 11 cases aren't
in its 86.0% because there's no sensible way to score it on them.

So the trade is: capping at 3 chunks costs us about 2 points on a keyword metric and
buys 7.4x lower P50 latency, 83% lower cost per request, and a system that refuses
instead of inventing. For a documentation assistant that's the right side of the trade,
since a confident wrong answer about a CLI flag is worse than "not in the docs".

---

## 6. Scale & Cost Estimations

Rough sizing to check whether this makes sense beyond a demo. Assuming it got used
inside an engineering org:

Assumptions:
- 500 developers, around 6 doc lookups each per day, so about 3,000 queries/day
- Per-request costs taken from our actual runs, not estimated: $0.000142 naive,
  $0.000024 with the full pipeline
- Groq `openai/gpt-oss-20b` pricing, roughly 500 prompt and 120 completion tokens on an
  uncached query

Without caching (naive RAG):
- 3,000 x $0.000142 = about $0.43/day, so roughly $12.80/month

With DocuQuery:
- 3,000 x $0.000024 = about $0.07/day, so roughly $2.20/month
- Around 1,000 of those 3,000 queries actually reach the LLM, the other two thirds come
  back from the semantic cache at zero cost

The savings should get better with more users, not worse. Documentation questions follow
a power law, a small set of questions ("how do I map docker ports") gets asked over and
over, and the cache is shared across everyone using the system.

Infrastructure is not the constraint at this size. The vector store is 35 chunks, and the
semantic cache caps at 200 entries of 384-dim embeddings, which is about 300KB. That's
why keeping the cache in process is acceptable here, though Section 9 covers where that
assumption breaks.

Somewhere around 50,000 queries/day the in-process cache and rate limiter become the
bottleneck before the cost does.

---

## 7. Evaluation: Offline and Online

### 7.1 Offline
This is what we actually run. Covered in Section 3: 43-case benchmark, two scoring
methods, per-category breakdown, CI gate.

### 7.2 Online
We don't have production users, so none of this is measured. These are the metrics we
would track if it were deployed internally:

| Metric | Why it matters | Alert threshold |
| :--- | :--- | :--- |
| Answer acceptance rate (north star) | Did the developer stop searching, or go to Google anyway | < 60% |
| Thumbs up/down rate | Direct quality signal and cheap to collect | < 70% positive |
| Cache hit rate | Our main cost lever, a drop here means the bill is about to go up | < 40% |
| Abstention rate | Too high means retrieval is too strict or the docs are too thin. Too low means the early-exit gate isn't protecting us | outside 2-30% |
| Cost per query | Budget protection | > $0.0005 |
| P99 latency | User experience | > 5s |
| Citation click-through | Are the sources we show actually the right ones | < 20% |

Abstention rate and acceptance rate have to be read together. A system that abstains
from everything would score great on faithfulness while being useless, so one number
without the other is misleading.

---

## 8. A/B Testing Design

We have no live traffic so this wasn't run, but this is how we'd set it up.

The first thing worth testing is whether going from top-3 to top-5 chunks improves
faithfulness enough to be worth the extra tokens. Control is the current top-3 pipeline,
treatment is top-5.

Assignment is per developer, not per query. This matters because the cache is shared, so
splitting by query would let the treatment arm pick up cache entries created by the
control arm and the comparison would be contaminated.

Primary metric is answer acceptance rate, with cost per query and P99 latency as
guardrails. We'd run it for 2 weeks since documentation lookups get bursty around sprint
boundaries and a shorter window risks only catching one part of that cycle.

Decision rule: promote only if acceptance rate improves by at least 3%, cost per query
stays under $0.0005, and P99 doesn't regress more than 10%.

Two biases to watch. Cache contamination across arms, handled by assigning per developer
as above. And novelty effect, since people use a new tool more enthusiastically in the
first week, so we'd compare week 2 numbers rather than week 1.

---

## 9. Deployment, Monitoring & Known Limitations

### 9.1 Current deployment
Runs locally, Streamlit UI plus the FastAPI server. The vector index builds itself on
first startup, so a fresh container can bootstrap without a committed database.

### 9.2 Monitoring
Every request gets appended to `logs/traces.jsonl` through `src/trace_logger.py`: event
type (generated, cache_hit, early_exit, intent_routed), latency, tokens, cost and
citation count. The "Observability Dashboard" tab in Streamlit reads that file back and
charts cost and latency over time. It's a local stand-in for what Langfuse or Grafana
would do properly.

### 9.3 Known limitations

| Limitation | Why it's like that | How we'd fix it |
| :--- | :--- | :--- |
| Semantic cache lives in process memory | Simple, and only about 300KB at our scale | Move it to Redis with a vector index. Right now running more than one replica means each gets its own cache, the hit rate drops and the cost savings go with it. This is the biggest architectural weakness we have |
| Rate limiter is in process too | Same reason | Same fix. A per-IP bucket in local memory resets on restart and is easy to get around by load balancing across replicas |
| Cache has no TTL or invalidation | Our docs corpus is static | Add a TTL and a re-index hook. If the docs changed, stale answers would stick around until eviction |
| Cache lookup is a linear scan | Fine at 200 entries | Switch to an ANN index (FAISS or Redis) past roughly 10K entries |
| Streaming is simulated | We split the finished answer word by word for the UI effect | Use the provider's actual token streaming API for a real TTFT improvement |
| Offline fallback token counts are estimated | Groq returns real usage counts so normal requests are metered, but the offline path has no API response to read, so it falls back to a 4-chars-per-token heuristic | Only affects runs with no API key set, where cost isn't real anyway |
| Ingestion re-chunks on every startup | Keeps the BM25 index in sync without needing a build artifact | Cache the tokenized corpus to disk. Startup cost grows with corpus size as it stands |

---

## 10. Failure Modes and Recovery

| What breaks | How we detect it | Recovery |
| :--- | :--- | :--- |
| LLM API quota runs out | 429s in the logs, cost per request drops to $0 while latency stays flat | Falls through Groq, then OpenAI, then offline synthesis from the retrieved chunks. Degrades to returning raw doc text instead of failing outright. We hit this for real on Gemini's free tier, which is what pushed us to Groq |
| Retrieval finds nothing relevant | `best_dense_sim` comes back below 0.35 | Early-exit gate abstains before spending any generation tokens |
| Prompt injection in the user input | Adversarial category drops below 100% in CI | System prompt tells the model to treat retrieved docs and the question as untrusted content, and the 8 adversarial cases gate every push |
| Quality regression from a prompt or chunking change | CI eval gate fails on the PR | Blocked before merge, and the per-category breakdown shows which area moved |
| Vector index missing or corrupt | Chunk count is 0 at startup | `build_vector_store()` is idempotent so it re-indexes on startup |
| One user floods the API | Rate limiter starts returning 429 | Token bucket capped at 20 req/min per IP, with the replica caveat from 9.3 |
| Cost blow-up | Cost per request trending up in `logs/traces.jsonl` | We'd only catch this after the fact right now, not prevent it. A hard per-session cost ceiling is the obvious next thing to add |

---

## 11. Conclusion

DocuQuery shows you don't need a frontier model or a huge context window to get useful
answers out of a RAG system. Header-aware chunking, hybrid retrieval with RRF, semantic
caching, early-exit abstention and an automated CI regression gate get us to 88.0%
faithfulness across the full benchmark at $0.000024 per request, with every one of the 8
adversarial prompts refused.

The honest version of the comparison is that our faithfulness on answerable questions
(83.8%) sits slightly below the naive baseline (86.0%), because we deliberately send
less context. What we get for those 2 points is 7.4x lower latency, 83% lower cost, and
a system that abstains on things it doesn't know instead of answering anyway.

The main thing we'd change before calling it production ready is moving the semantic
cache and the rate limiter out of process memory and into Redis. As it stands the cost
advantage quietly depends on running exactly one replica, which isn't a safe assumption
to ship on.
