import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """
    Cache that detects semantically similar queries
    """

    def __init__(self, similarity_threshold=0.80):

        self.cache = []
        self.similarity_threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        best_score = 0
        best_entry = None

        for entry in self.cache:

            score = cosine_similarity(
                query_embedding,
                entry["embedding"]
            )[0][0]

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold:

            self.hit_count += 1
            return True, best_entry, best_score

        self.miss_count += 1
        return False, None, best_score

    def add(self, query, embedding, result, cluster):

        entry = {
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        }

        self.cache.append(entry)

    def stats(self):

        total = len(self.cache)

        if self.hit_count + self.miss_count == 0:
            hit_rate = 0
        else:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0