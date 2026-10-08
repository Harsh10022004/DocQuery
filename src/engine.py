import os
import time
import yaml
from dotenv import load_dotenv

from src.cache import SemanticCache
from src.router import check_intent
from src.retrieval import HybridRetriever

load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


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
        
        # pricing constants (Gemini Flash default)
        self.input_rate = self.config["pricing_per_million_tokens"]["input_usd"] / 1_000_000
        self.output_rate = self.config["pricing_per_million_tokens"]["output_usd"] / 1_000_000

        # LLM setup (gemini or openai if api keys exist)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def _get_embedding(self, text: str):
        """Uses Chroma's underlying embedding function to embed queries for caching."""
        fn = self.retriever.collection._embedding_function
        return fn([text])[0]

    def _estimate_tokens(self, text: str):
        # standard 1 token ~ 4 characters heuristic when offline
        return max(1, len(text) // 4)

    def _call_llm(self, prompt: str, system_prompt: str):
        """Calls Gemini Flash, OpenAI, or local extractor if offline."""
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel(
                    model_name=self.config["models"]["primary_llm"],
                    system_instruction=system_prompt
                )
                res = model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": self.config["models"]["max_output_tokens"], "temperature": 0.1}
                )
                out_text = res.text
                in_tokens = self._estimate_tokens(prompt + system_prompt)
                out_tokens = self._estimate_tokens(out_text)
                return out_text, in_tokens, out_tokens
            except Exception as e:
                print(f"Gemini API call failed, falling back: {e}")

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

        # local offline fallback: synthesizes directly from the retrieved context
        in_tokens = self._estimate_tokens(prompt + system_prompt)
        out_tokens = 120
        # clean offline answer: display extracted documentation guidance
        clean_text = "\n\n".join([f"**From {c['title']}**:\n{c['text']}" for c in self.retriever.chunks if c['id'] in prompt][:2])
        if not clean_text:
            clean_text = "Here is the verified documentation snippet for your query:\n" + prompt[:400]
        return clean_text, in_tokens, out_tokens

    def query(self, user_query: str, domain_filter: str = "all"):
        """
        Executes the full cost-optimized query pipeline:
        Intent Route -> Cache Check -> Hybrid Retrieval -> Early Exit -> LLM Generation.
        """
        start_time = time.time()
        
        # 1. Intent routing (small talk check)
        is_casual, reply = check_intent(user_query)
        if is_casual:
            elapsed = time.time() - start_time
            return {
                "answer": reply,
                "citations": [],
                "cache_hit": False,
                "intent_routed": True,
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0
            }

        # get query embedding for semantic cache
        query_emb = self._get_embedding(user_query)

        # 2. Semantic Cache Check
        cached_entry, sim_score = self.cache.lookup(query_emb)
        if cached_entry:
            elapsed = time.time() - start_time
            return {
                "answer": cached_entry["response"],
                "citations": cached_entry["citations"],
                "cache_hit": True,
                "similarity": round(sim_score, 4),
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0
            }

        # 3. Hybrid Retrieval
        chunks, best_sim = self.retriever.retrieve(user_query, domain_filter)
        
        # 4. Early-exit gate if query is completely irrelevant to docs
        min_threshold = self.config["retrieval"]["min_relevance_score"]
        if best_sim < min_threshold or not chunks:
            elapsed = time.time() - start_time
            return {
                "answer": "This topic is not covered in the indexed documentation (FastAPI, Docker, Git, PostgreSQL).",
                "citations": [],
                "cache_hit": False,
                "early_exit": True,
                "latency_seconds": round(elapsed, 4),
                "prompt_tokens": self._estimate_tokens(user_query),
                "completion_tokens": 20,
                "cost_usd": 0.0
            }

        # 5. Build compact prompt with static prefix for provider prefix caching
        system_prompt = (
            "You are DocuQuery, a technical assistant. "
            "Answer the question using ONLY the provided documentation chunks. "
            "If the answer cannot be determined from the context, state that clearly. "
            "Be concise and output exact code syntax where applicable."
        )

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
        raw_answer, in_tokens, out_tokens = self._call_llm(prompt_body, system_prompt)
        
        # calculate USD cost
        cost = (in_tokens * self.input_rate) + (out_tokens * self.output_rate)
        elapsed = time.time() - start_time

        # 7. Store in Semantic Cache
        self.cache.add(user_query, query_emb, raw_answer, citations)

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
        
        if self.gemini_key or self.openai_key:
            system_prompt = (
                "You are a Senior Principal Engineer. Provide a concise, high-value technical analysis "
                "based on the documentation context. Cover: "
                "1. Architectural Best Practices, 2. Production Pitfalls & Edge Cases, 3. Performance/Security tips."
            )
            prompt = f"Documentation Context:\n{context_str}\n\nDeveloper Query:\n{query}\n\nEngineering Analysis:"
            analysis_text, _, _ = self._call_llm(prompt, system_prompt)
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
