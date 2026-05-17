"""
routers/query.py — FastAPI router for the /query endpoint.

Handles the full RAG query flow:
    embed query → retrieve from Qdrant → rerank → call Claude → return answer
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.retrieval.query_engine import query_engine
from app.retrieval.reranker import reranker
from app.retrieval.synthesizer import synthesizer

router = APIRouter()


class QueryRequest(BaseModel):
    query:   str               # user's natural language question
    tickers: list[str] = []    # e.g. ["AAPL", "MSFT"]


class QueryResponse(BaseModel):
    answer:      str
    sources:     list[dict]
    companies:   list[str]
    chunks_used: int
    model:       str


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Answer a financial question using RAG.

    Flow:
        1. Embed the query using the same model used at ingestion
        2. Retrieve top-K chunks from Qdrant (per company if tickers provided)
        3. Rerank with cross-encoder to find the most relevant chunks
        4. Call Claude with the chunks as context
        5. Return Claude's grounded answer with citations
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Stage 1: retrieve
        chunks = query_engine.retrieve(
            query=   req.query,
            tickers= req.tickers or None,
        )

        # Stage 2: rerank
        top_chunks = reranker.rerank(
            query=  req.query,
            chunks= chunks,
        )

        # Stage 3: generate
        response = synthesizer.generate(
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
