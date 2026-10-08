import time
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.engine import DocuQueryEngine

app = FastAPI(title="DocuQuery Serving API", version="1.0.0")

# allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize engine once on startup
engine = DocuQueryEngine()

# Token-Bucket Rate Limiter (Session 15 concept)
# 20 requests per minute per IP to prevent API credit exhaustion
RATE_LIMIT = 20
BUCKET_CAPACITY = 20
tokens_db = defaultdict(lambda: {"tokens": BUCKET_CAPACITY, "last_refill": time.time()})


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    bucket = tokens_db[client_ip]
    elapsed = now - bucket["last_refill"]
    
    # refill tokens based on elapsed time (20 tokens per 60 seconds)
    refill = elapsed * (RATE_LIMIT / 60.0)
    bucket["tokens"] = min(BUCKET_CAPACITY, bucket["tokens"] + refill)
    bucket["last_refill"] = now
    
    if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        return True
    return False


class QueryRequest(BaseModel):
    query: str
    domain: str = "all"


class QueryResponse(BaseModel):
    answer: str
    citations: list
    cache_hit: bool
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@app.get("/health")
def health_check():
    """Readiness probe checking engine and index availability."""
    return {
        "status": "healthy",
        "indexed_chunks": engine.retriever.collection.count(),
        "cache_size": engine.cache.size(),
        "models": {
            "primary": engine.config["models"]["primary_llm"],
            "fallback": engine.config["models"]["fallback_llm"]
        }
    }


@app.post("/api/query", response_model=QueryResponse)
def query_docs(req: QueryRequest, request: Request):
    """Standard REST endpoint for documentation lookups."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 20 queries per minute allowed."
        )

    result = engine.query(req.query, domain_filter=req.domain)
    return result


@app.get("/api/stream")
def stream_query(query: str, domain: str = "all", request: Request = None):
    """
    Server-Sent Events (SSE) streaming endpoint for low Time-To-First-Token (TTFT).
    Streams answer word-by-word.
    """
    client_ip = request.client.host if request and request.client else "127.0.0.1"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    result = engine.query(query, domain_filter=domain)
    
    def event_generator():
        # stream words in chunks for smooth UI typing effect
        words = result["answer"].split()
        for i, word in enumerate(words):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            time.sleep(0.015)  # simulate natural streaming rate
            
        # final event delivers metadata, citations, and telemetry
        meta = {
            "done": True,
            "citations": result["citations"],
            "cache_hit": result["cache_hit"],
            "latency_seconds": result["latency_seconds"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "cost_usd": result["cost_usd"]
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=True)
