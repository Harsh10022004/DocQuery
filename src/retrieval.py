from rank_bm25 import BM25Okapi
from src.ingest import load_markdown_chunks, build_vector_store


class HybridRetriever:
    """
    Two-stage retrieval combining BM25 keyword matching with
    ChromaDB vector similarity using Reciprocal Rank Fusion (RRF).
    Selects top 3 chunks to prevent prompt token explosion.
    """
    def __init__(self, top_k=3, rrf_k=60):
        self.top_k = top_k
        self.rrf_k = rrf_k
        
        # build in-memory bm25 index from raw chunks
        self.chunks = load_markdown_chunks()

        # build_vector_store returns the existing collection if it's already there,
        # otherwise it builds it. using this instead of get_collection so a fresh
        # deploy (chroma_db isn't committed) indexes itself instead of crashing
        self.collection = build_vector_store(self.chunks)
        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, domain_filter: str = None):
        """
        Runs hybrid search and returns top-k chunks with relevance scores.
        """
        # 1. Dense retrieval via Chroma
        where_filter = None
        if domain_filter and domain_filter != "all":
            where_filter = {"domain": domain_filter}

        dense_results = self.collection.query(
            query_texts=[query],
            n_results=min(10, len(self.chunks)),
            where=where_filter
        )
        
        dense_ranked_ids = dense_results["ids"][0] if dense_results["ids"] else []
        dense_distances = dense_results["distances"][0] if dense_results["distances"] else []

        # 2. Sparse retrieval via BM25
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # pair indices with scores, filter by domain if specified
        bm25_ranked = []
        for idx, score in enumerate(bm25_scores):
            chunk = self.chunks[idx]
            if where_filter and chunk["domain"] != domain_filter:
                continue
            bm25_ranked.append((chunk["id"], score))
        
        # sort bm25 by descending score
        bm25_ranked.sort(key=lambda x: x[1], reverse=True)
        sparse_ranked_ids = [item[0] for item in bm25_ranked[:10]]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        for rank, cid in enumerate(dense_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        for rank, cid in enumerate(sparse_ranked_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # sort by fused score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # take top-k
        final_chunks = []
        for cid, score in sorted_candidates[:self.top_k]:
            chunk = self.chunk_by_id[cid].copy()
            chunk["rrf_score"] = round(score, 4)
            final_chunks.append(chunk)

        # estimate max similarity from dense distance (cosine distance: 0=identical, 1=orthogonal)
        best_dense_sim = 1.0 - (dense_distances[0] if dense_distances else 1.0)
        
        return final_chunks, best_dense_sim
