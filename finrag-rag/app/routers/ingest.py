"""
routers/ingest.py — FastAPI router for the /ingest endpoint.

Separating routes into their own files keeps main.py clean.
This router handles triggering the ingestion pipeline for a company.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ingestion.pipeline import ingest_company

router = APIRouter()


class IngestRequest(BaseModel):
    ticker:       str    # e.g. "AAPL"
    company_name: str    # e.g. "Apple"


class IngestResponse(BaseModel):
    ticker:        str
    company:       str
    chunks_stored: int
    status:        str


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Trigger document ingestion for one company.

    Fetches 10-K from SEC EDGAR + news articles, chunks them,
    embeds them, and stores them in Qdrant.

    This is a slow operation (1-5 min) — it blocks until complete.
    """
    if not req.ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")
    if not req.company_name.strip():
        raise HTTPException(status_code=400, detail="Company name cannot be empty.")

    try:
        result = ingest_company(
            ticker=       req.ticker.upper(),
            company_name= req.company_name,
        )
        return IngestResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
