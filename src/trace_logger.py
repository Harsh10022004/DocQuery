import os
import json
import time

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
TRACE_FILE = os.path.join(LOGS_DIR, "traces.jsonl")

# simple local observability: one json line per request.
# not Langfuse/Grafana, but same idea - every request gets logged so we can
# look at cost/latency trends later instead of only seeing them live in the UI.


def log_trace(event_type: str, query: str, result: dict):
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,  # cache_hit | intent_routed | early_exit | generated
            "query": query,
            "latency_seconds": result.get("latency_seconds"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "cost_usd": result.get("cost_usd"),
            "cache_hit": result.get("cache_hit", False),
            "num_citations": len(result.get("citations", [])),
        }
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        # logging should never break the actual query path
        print(f"trace logging failed (non-fatal): {e}")


def load_traces():
    """Reads all logged traces back as a list of dicts, for the dashboard."""
    if not os.path.exists(TRACE_FILE):
        return []
    traces = []
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces
