import os
import sys
import time
import streamlit as st

# add parent directory to path so imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import DocuQueryEngine

# set page config
st.set_page_config(
    page_title="DocuQuery - Documentation Q&A",
    page_icon="🔍",
    layout="wide"
)

# initialize engine in session state
if "engine" not in st.session_state:
    st.session_state.engine = DocuQueryEngine()

if "history" not in st.session_state:
    st.session_state.history = []

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0

engine = st.session_state.engine

# Sidebar: System telemetry and unit economics
st.sidebar.title("📊 System Telemetry")
st.sidebar.markdown("**DocuQuery LLMOps Monitor**")

# session stats
total_queries = len(st.session_state.history)
hit_pct = round((st.session_state.cache_hits / total_queries * 100), 1) if total_queries > 0 else 0.0

col1, col2 = st.sidebar.columns(2)
col1.metric("Queries", total_queries)
col2.metric("Cache Hit Rate", f"{hit_pct}%")
st.sidebar.metric("Total Spent", f"${st.session_state.total_cost:.5f}")

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Pipeline Configuration")
st.sidebar.markdown(f"- **Primary LLM:** `{engine.config['models']['primary_llm']}`")
st.sidebar.markdown(f"- **Embedding:** `{engine.config['models']['embedding_model']}`")
st.sidebar.markdown(f"- **Top-K Context:** `{engine.config['retrieval']['final_top_k']} chunks`")
st.sidebar.markdown(f"- **Cache Threshold:** `{engine.config['cache']['similarity_threshold']}`")
st.sidebar.markdown(f"- **Total Indexed Chunks:** `{engine.retriever.collection.count()}`")

# Main Header
st.title("🔍 DocuQuery")
st.caption("Technical Documentation Assistant with Semantic Caching & Token Optimization")

# Dedicated Tabs: Search & Q&A vs Session History
tab_search, tab_history = st.tabs(["🔍 Documentation Search & Q&A", "📜 Session History & Analytics"])

# ==========================================
# TAB 1: SEARCH & Q&A
# ==========================================
with tab_search:
    # Filter by documentation domain
    col_domain, col_space = st.columns([1, 2])
    domain_choice = col_domain.selectbox(
        "Filter Documentation Source:",
        ["all", "fastapi", "docker", "git", "postgres"],
        format_func=lambda x: "All Documentation Sources" if x == "all" else x.upper()
    )

    # Sample query buttons for quick evaluation
    st.markdown("**Sample Questions:**")
    btn_cols = st.columns(4)
    sample_q = None

    if btn_cols[0].button("FastAPI Path Params"):
        sample_q = "How do you declare path parameters in FastAPI?"
    if btn_cols[1].button("Docker Port Mapping"):
        sample_q = "How do you map host and container ports in Docker?"
    if btn_cols[2].button("Git Merge vs Rebase"):
        sample_q = "What is the difference between git merge and git rebase?"
    if btn_cols[3].button("Postgres GIN Index"):
        sample_q = "When should you use a GIN index in PostgreSQL?"

    # User query input
    query_input = st.text_input(
        "Ask a technical documentation question:",
        value=sample_q if sample_q else "",
        placeholder="e.g. How to use background tasks in FastAPI?"
    )

    if st.button("Search & Answer", type="primary") or (sample_q and query_input):
        if not query_input.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing through DocuQuery pipeline..."):
                res = engine.query(query_input, domain_filter=domain_choice)

                # update session counters
                st.session_state.total_cost += res["cost_usd"]
                if res.get("cache_hit", False):
                    st.session_state.cache_hits += 1

                st.session_state.history.append({
                    "timestamp": time.strftime("%H:%M:%S"),
                    "query": query_input,
                    "domain": domain_choice,
                    "res": res
                })

                # 1. Answer section
                st.markdown("### Answer")
                
                # telemetry badge line
                b_cols = st.columns(4)
                if res.get("cache_hit"):
                    b_cols[0].success("⚡ Cache Hit (0 cost)")
                elif res.get("intent_routed"):
                    b_cols[0].info("💬 Intent Routed (Direct)")
                elif res.get("early_exit"):
                    b_cols[0].warning("🛑 Early-Exit Abstention")
                else:
                    b_cols[0].info("🔎 Hybrid Retrieved")

                b_cols[1].metric("Latency", f"{res['latency_seconds']:.2f}s")
                b_cols[2].metric("Prompt Tokens", res["prompt_tokens"])
                b_cols[3].metric("Query Cost", f"${res['cost_usd']:.5f}")

                # display formatted response
                st.markdown(res["answer"])

                # 2. Source mapping and verified citations
                if res.get("citations"):
                    st.markdown("---")
                    st.markdown("#### 📚 Source Mapping & Verified Citations")
                    st.caption("Inspect the exact official documentation sections and code retrieved for this answer:")
                    
                    for idx, cite in enumerate(res["citations"], 1):
                        domain_tag = cite.get("domain", "doc").upper()
                        title = cite.get("title", "Section")
                        url = cite.get("url", "#")
                        score = cite.get("relevance_score", 0.0)
                        score_str = f" • Match Score: {score:.3f}" if score > 0 else ""

                        with st.expander(f"📄 [{domain_tag}] {title}{score_str}", expanded=(idx == 1)):
                            st.markdown(f"🔗 **Official Documentation Link:** [{url}]({url})")
                            st.markdown("**Exact Document Section Excerpt:**")
                            excerpt_text = cite.get("excerpt", "")
                            if excerpt_text:
                                st.markdown(excerpt_text)
                            else:
                                st.info("Direct documentation section referenced above.")

                # 3. Optional AI Analysis Section
                if res.get("citations"):
                    st.markdown("---")
                    with st.expander("🤖 Deep AI Analysis & Code Breakdown (Optional)", expanded=False):
                        st.caption("On-demand architectural review, production gotchas, and security considerations:")
                        analysis_output = engine.analyze_deep(query_input, res["citations"])
                        st.markdown(analysis_output)


# ==========================================
# TAB 2: SESSION HISTORY
# ==========================================
with tab_history:
    st.markdown("### 📜 Session History & Query Traces")
    st.caption("Review all past questions, answers, and telemetry metrics from your current session.")

    if st.session_state.history:
        col_hist_top, col_clear = st.columns([3, 1])
        col_hist_top.markdown(f"**Total Queries in Session:** `{len(st.session_state.history)}`")
        if col_clear.button("🗑️ Clear Session History"):
            st.session_state.history = []
            st.session_state.cache_hits = 0
            st.session_state.total_cost = 0.0
            st.rerun()

        for idx, item in enumerate(reversed(st.session_state.history), 1):
            q_text = item["query"]
            t_stamp = item.get("timestamp", "")
            r_data = item["res"]
            cache_flag = "⚡ [CACHE HIT]" if r_data.get("cache_hit") else "🔎 [RETRIEVED]"
            
            with st.expander(f"#{idx} | {t_stamp} | {cache_flag} {q_text}", expanded=(idx == 1)):
                st.markdown(f"**Question:** {q_text}")
                st.markdown(f"**Answer:**\n\n{r_data['answer']}")
                
                # metrics summary line
                m1, m2, m3, m4 = st.columns(4)
                m1.caption(f"⏱️ **Latency:** {r_data['latency_seconds']}s")
                m2.caption(f"🔤 **Prompt Tokens:** {r_data['prompt_tokens']}")
                m3.caption(f"💰 **Cost:** ${r_data['cost_usd']:.5f}")
                m4.caption(f"🏷️ **Source:** {item.get('domain', 'all').upper()}")

                if r_data.get("citations"):
                    st.markdown("**Cited Sources:**")
                    for c in r_data["citations"]:
                        st.markdown(f"- 📄 [{c['domain'].upper()}] [{c['title']}]({c['url']})")
    else:
        st.info("No queries have been submitted in this session yet. Go to the **Search & Q&A** tab to ask a question!")
