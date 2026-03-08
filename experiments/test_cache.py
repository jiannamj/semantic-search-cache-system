import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embeddings.embedder import Embedder
from app.cache.semantic_cache import SemanticCache


embedder = Embedder()

cache = SemanticCache(similarity_threshold=0.85)


query1 = "How do rockets launch?"
query2 = "Explain rocket launch mechanism"


emb1 = embedder.embed(query1)
emb2 = embedder.embed(query2)


cache.add(query1, emb1, "Rocket launch explanation", 1)


hit, entry, score = cache.lookup(emb2)

print("Cache hit:", hit)
print("Similarity score:", score)

if hit:
    print("Matched query:", entry["query"])