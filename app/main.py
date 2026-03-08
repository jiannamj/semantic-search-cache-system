from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Semantic Search Cache API",
    description="Semantic search with clustering and caching"
)

app.include_router(router)