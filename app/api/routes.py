from fastapi import APIRouter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.embeddings.embedder import Embedder
from app.cache.semantic_cache import SemanticCache

router = APIRouter()

embedder = Embedder()
cache = SemanticCache(similarity_threshold=0.85)


@router.post("/query")
def query(data: dict):

    query_text = data["query"]

    query_embedding = embedder.embed(query_text)

    hit, entry, score = cache.lookup(query_embedding)

    if hit:
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    # dummy result (we will replace later with vector search)
    result = "Result generated for query"

    cluster = 0

    cache.add(query_text, query_embedding, result, cluster)

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(score),
        "result": result,
        "dominant_cluster": cluster
    }


@router.get("/cache/stats")
def cache_stats():
    return cache.stats()


@router.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}