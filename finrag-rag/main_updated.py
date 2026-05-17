"""
main.py — FastAPI entry point for the FinRAG Python service.

Registers the ingest and query routers.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from app.routers import ingest, query

app = FastAPI(
    title="FinRAG Python Service",
    description="RAG pipeline for financial document Q&A",
    version="0.1.0",
)

# Register routers
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
def health():
    """Liveness check — Java gateway pings this on startup."""
    return {"status": "ok", "service": "finrag-rag"}
