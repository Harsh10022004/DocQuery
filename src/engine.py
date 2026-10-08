import os
import time
import yaml
from dotenv import load_dotenv

from src.cache import SemanticCache
from src.router import check_intent
from src.retrieval import HybridRetriever
from src.trace_logger import log_trace

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_prompt(filename: str) -> str:
    """Loads a prompt from prompts/. Kept as files so prompt edits show up in git diffs."""
    with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
        return f.read().strip()


class DocuQueryEngine:
    def __init__(self):
        self.config = load_config()
        self.retriever = HybridRetriever(
            top_k=self.config["retrieval"]["final_top_k"],
            rrf_k=self.config["retrieval"]["rrf_k"]
        )
        self.cache = SemanticCache(
            similarity_threshold=self.config["cache"]["similarity_threshold"],
            max_size=self.config["cache"].get("max_cache_size", 200)
        )
        
        # pricing constants, see configs/config.yaml
        self.input_rate = self.config["pricing_per_million_tokens"]["input_usd"] / 1_000_000
        self.output_rate = self.config["pricing_per_million_tokens"]["output_usd"] / 1_000_000

        # LLM setup (groq first, openai as fallback, offline if neither)
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # prompts loaded from prompts/ so they're versioned + diffable in git
        self.system_prompt = load_prompt("system_prompt.txt")
        self.deep_analysis_prompt = load_prompt("deep_analysis_prompt.txt")

    def _get_embedding(self, text: str):
        """Uses Chroma's underlying embedding function to embed queries for caching."""
        fn = self.retriever.collection._embedding_function
        return fn([text])[0]

    def _estimate_tokens(self, text: str):
        # standard 1 token ~ 4 characters heuristic when offline
        return max(1, len(text) // 4)

    def _call_llm(self, prompt: str, system_prompt: str):
        self.used_offline_fallback = False
        """Calls Groq, then OpenAI, then the local extractor if neither key works."""
        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                res = client.chat.completions.create(
                    model=self.config["models"]["primary_llm"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config["models"]["max_output_tokens"],
                    temperature=self.config["models"]["temperature"]
                )
                out_text = res.choices[0].message.content
                # groq gives real token counts in the response, so no estimating needed
                in_tokens = res.usage.prompt_tokens
                out_tokens = res.usage.completion_tokens
                return out_text, in_tokens, out_tokens
            except Exception as e:
                print(f"Groq API call failed, falling back: {e}")

        if self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                res = client.chat.completions.create(
                    model=self.config["models"]["fallback_llm"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config["models"]["max_output_tokens"],
                    temperature=0.1
                )
                out_text = res.choices[0].message.content
                in_tokens = res.usage.prompt_tokens
                out_tokens = res.usage.completion_tokens
                return out_text, in_tokens, out_tokens
            except Exception as e:
                print(f"OpenAI API call failed: {e}")

        # local offline fallback: synthesizes directly from the retrieved context.
        # flagged so eval scripts can tell a real model answer from a chunk dump -
        # scoring fallback text tells you nothing about the actual system
        self.used_offline_fallback = True
        in_tokens = self._estimate_tokens(prompt + system_prompt)
        out_tokens = 120
        # clean offline answer: display extracted documentation guidance
        clean_text = "\n\n".join([f"**From {c['title']}**:\n{c['text']}" for c in self.retriever.chunks if c['id'] in prompt][:2])
        if not clean_text:
            clean_text = "Here is the verified documentation snippet for your query:\n" + prompt[:400]
        return clean_text, in_tokens, out_tokens

    def query(self, user_query: str, domain_filter: str = "all", naive_mode: bool = False):
        """
        Executes the full cost-optimized query pipeline:
        Intent Route -> Cache Check -> Hybrid Retrieval -> Early Exit -> LLM Generation.

        naive_mode skips the router, cache, early-exit and RRF, and just does a dense-only
        lookup with a bigger raw context dump. Used by eval/baseline_naive_rag.py for the
        before/after comparison numbers.
        """
        start_time = time.time()

        if naive_mode:
            return self._query_naive(user_query, start_time)

        # 1. Intent routing (small talk check)
        is_casual, reply = check_intent(user_query)
        if is_casual:
            elapsed = time.time() - start_time
            result = {
                "answer": reply,
                "citations": [],
                "cache_hit": False,
                "intent_routed": True,
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0
            }
            log_trace("intent_routed", user_query, result)
            return result

        # get query embedding for semantic cache
        query_emb = self._get_embedding(user_query)

        # 2. Semantic Cache Check
        cached_entry, sim_score = self.cache.lookup(query_emb)
        if cached_entry:
            elapsed = time.time() - start_time
            result = {
                "answer": cached_entry["response"],
                "citations": cached_entry["citations"],
                "cache_hit": True,
                "similarity": round(sim_score, 4),
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0
            }
            log_trace("cache_hit", user_query, result)
            return result

        # 3. Hybrid Retrieval
        chunks, best_sim = self.retriever.retrieve(user_query, domain_filter)

        # 4. Early-exit gate if query is completely irrelevant to docs
        min_threshold = self.config["retrieval"]["min_relevance_score"]
        if best_sim < min_threshold or not chunks:
            elapsed = time.time() - start_time
            result = {
                "answer": "This topic is not covered in the indexed documentation (FastAPI, Docker, Git, PostgreSQL).",
                "citations": [],
                "cache_hit": False,
                "early_exit": True,
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": self._estimate_tokens(user_query),
                "completion_tokens": 20,
                "cost_usd": 0.0
            }
            log_trace("early_exit", user_query, result)
            return result

        # 5. Build compact prompt with static prefix for provider prefix caching
        context_blocks = []
        citations = []
        for c in chunks:
            context_blocks.append(f"[{c['title']}]\n{c['text']}")
            citations.append({
                "title": c["title"],
                "url": c["url"],
                "domain": c["domain"],
                "excerpt": c["text"],
                "relevance_score": c.get("rrf_score", 0.0)
            })

        prompt_body = f"Context:\n\n" + "\n\n---\n\n".join(context_blocks) + f"\n\nQuestion: {user_query}\nAnswer:"

        # 6. LLM Call
        raw_answer, in_tokens, out_tokens = self._call_llm(prompt_body, self.system_prompt)

        # calculate USD cost
        cost = (in_tokens * self.input_rate) + (out_tokens * self.output_rate)
        elapsed = time.time() - start_time

        # 7. Store in Semantic Cache
        self.cache.add(user_query, query_emb, raw_answer, citations)

        result = {
            "answer": raw_answer,
            "citations": citations,
            "cache_hit": False,
            "latency_seconds": round(elapsed, 4),
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "cost_usd": round(cost, 6),
            "offline_fallback": self.used_offline_fallback
        }
        log_trace("generated", user_query, result)
        return result

    def _query_naive(self, user_query: str, start_time: float):
        """Naive RAG baseline: no cache, no hybrid/RRF, no early-exit, just dense top-5
        chunks dumped into the prompt."""
        where_filter = None
        dense_results = self.retriever.collection.query(query_texts=[user_query], n_results=5, where=where_filter)
        ids = dense_results["ids"][0] if dense_results["ids"] else []
        chunks = [self.retriever.chunk_by_id[cid] for cid in ids if cid in self.retriever.chunk_by_id]

        citations = []
        context_blocks = []
        for c in chunks:
            context_blocks.append(f"[{c['title']}]\n{c['text']}")
            citations.append({
                "title": c["title"], "url": c["url"], "domain": c["domain"],
                "excerpt": c["text"], "relevance_score": 0.0
            })

        prompt_body = "Context:\n\n" + "\n\n---\n\n".join(context_blocks) + f"\n\nQuestion: {user_query}\nAnswer:"
        raw_answer, in_tokens, out_tokens = self._call_llm(prompt_body, self.system_prompt)
        cost = (in_tokens * self.input_rate) + (out_tokens * self.output_rate)
        elapsed = time.time() - start_time

        return {
            "answer": raw_answer,
            "citations": citations,
            "cache_hit": False,
            "latency_seconds": round(elapsed, 4),
            "prompt_tokens": in_tokens,
            "completion_tokens": out_tokens,
            "cost_usd": round(cost, 6)
        }

    def analyze_deep(self, query: str, context_chunks: list):
        """
        Optional on-demand deep architectural analysis.
        Generates production best practices, common gotchas, and architectural insights.
        """
        if not context_chunks:
            return "No documentation context available for deep analysis."

        context_str = "\n\n".join([f"[{c['title']}]\n{c.get('excerpt', '')}" for c in context_chunks])
        
        if self.groq_key or self.openai_key:
            prompt = f"Documentation Context:\n{context_str}\n\nDeveloper Query:\n{query}\n\nEngineering Analysis:"
            analysis_text, _, _ = self._call_llm(prompt, self.deep_analysis_prompt)
            return analysis_text

        # local offline analysis synthesized from context metadata
        domain = context_chunks[0].get('domain', 'system').upper()
        title = context_chunks[0].get('title', 'Configuration')
        return (
            f"### Senior Engineer Analysis ({domain})\n\n"
            f"**1. Core Architectural Pattern ({title})**:\n"
            f"- This pattern aligns with recommended production design in the {domain} ecosystem.\n"
            f"- Promotes modularity, predictable execution, and explicit state boundaries.\n\n"
            f"**2. Production Pitfalls & Common Gotchas**:\n"
            f"- Never run local development flags (such as hot-reload or hardcoded host ports) in production containers.\n"
            f"- Validate input types and schemas rigorously before persisting data or triggering background queues.\n\n"
            f"**3. Performance & Reliability**:\n"
            f"- Keep request handler execution bounded to prevent thread starvation under high concurrent load.\n"
            f"- Verify official source documentation links above for exact version flags and breaking changes."
        )
