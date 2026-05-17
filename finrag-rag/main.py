"""
main.py — FastAPI entry point for the FinRAG Python service.

Two endpoints:
    POST /ingest  — fetch, chunk, embed and store documents for a company
    POST /query   — retrieve relevant chunks, rerank, and generate an answer

The Java Spring Boot gateway calls these endpoints over HTTP.
Users never talk to this service directly — the gateway handles auth.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline    import ingest_company
from query_engine import query_engine
from reranker    import reranker
from synthesizer import synthesizer, FinRagResponse
from config      import settings

app = FastAPI(
    title="FinRAG Python Service",
    description="RAG pipeline for financial document Q&A",
    version="0.1.0",
)


# ── Request / Response models ─────────────────────────────────────────────────
# Pydantic models validate incoming JSON automatically —
# if a required field is missing, FastAPI returns a 422 error immediately.

class IngestRequest(BaseModel):
    ticker:       str          # e.g. "AAPL"
    company_name: str          # e.g. "Apple"

class IngestResponse(BaseModel):
    ticker:        str
    company:       str
    chunks_stored: int
    status:        str

class QueryRequest(BaseModel):
    query:   str               # user's natural language question
    tickers: list[str] = []    # e.g. ["AAPL", "MSFT"] — empty means all companies

class QueryResponse(BaseModel):
    answer:      str
    sources:     list[dict]
    companies:   list[str]
    chunks_used: int
    model:       str


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Simple liveness check — Java gateway pings this on startup."""
    return {"status": "ok", "service": "finrag-rag"}


# ── Ingest endpoint ───────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Trigger the ingestion pipeline for one company.

    Flow:
        fetch_10k + fetch_news → chunk → embed → store in Qdrant

    This can take 1-5 minutes depending on document size.
    In production you'd run this as a background job, not a blocking request.
    For v1 blocking is fine — the Java gateway can set a long timeout.

    Example request body:
        {"ticker": "AAPL", "company_name": "Apple"}
    """
    try:
        result = ingest_company(
            ticker=       req.ticker,
            company_name= req.company_name,
        )
        return IngestResponse(**result)

    except ValueError as e:
        # e.g. ticker not found in EDGAR
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── Query endpoint ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Answer a financial question using RAG.

    Flow:
        embed query → retrieve from Qdrant → rerank → call Claude → return answer

    Example request body:
        {
            "query": "Compare Apple and Microsoft revenue growth in 2023",
            "tickers": ["AAPL", "MSFT"]
        }

    Example response:
        {
            "answer": "## AAPL\\nApple's net sales grew 3% YoY... [Source 1]\\n\\n## MSFT...",
            "sources": [{"company": "AAPL", "doc_type": "10-K", "year": 2023, ...}],
            "companies": ["AAPL", "MSFT"],
            "chunks_used": 6,
            "model": "claude-sonnet-4-20250514"
        }
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Stage 1: retrieve candidate chunks from Qdrant
        chunks = query_engine.retrieve(
            query=   req.query,
            tickers= req.tickers or None,
        )

        # Stage 2: rerank to find the most relevant chunks
        top_chunks = reranker.rerank(
            query=  req.query,
            chunks= chunks,
        )

        # Stage 3: generate answer with Claude
        response: FinRagResponse = synthesizer.generate(
            query=   req.query,
            chunks=  top_chunks,
            tickers= req.tickers,
        )

        return QueryResponse(
            answer=      response.answer,
            sources=     response.sources,
            companies=   response.companies,
            chunks_used= response.chunks_used,
            model=       response.model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
