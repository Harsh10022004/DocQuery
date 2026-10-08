import numpy as np


class SemanticCache:
    """
    In-memory semantic cache using cosine similarity on query embeddings.
    If a new question is semantically identical (> 0.92 cosine sim),
    we return the cached answer with zero LLM API cost.
    """
    def __init__(self, similarity_threshold=0.92, max_size=200):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.entries = []  # list of dicts: {'query', 'embedding', 'response', 'citations'}

    def _cosine_similarity(self, vec_a, vec_b):
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def lookup(self, query_embedding):
        """Returns cached item if similarity exceeds threshold, else None."""
        if not self.entries:
            return None, 0.0

        best_score = -1.0
        best_entry = None

        for item in self.entries:
            sim = self._cosine_similarity(query_embedding, item["embedding"])
            if sim > best_score:
                best_score = sim
                best_entry = item

        if best_score >= self.threshold and best_entry is not None:
            return best_entry, best_score

        return None, best_score

    def add(self, query, query_embedding, response, citations):
        """Adds a newly answered query to the cache."""
        # simple eviction if cache gets too large
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)

        self.entries.append({
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "citations": citations
        })

    def size(self):
        return len(self.entries)
