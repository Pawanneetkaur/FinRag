"""
pipeline.py — Orchestrates the full ingestion pipeline.

Flow:
    fetch_10k() / fetch_news()
        → chunk_documents()
            → embedder.embed_documents()
                → store in Qdrant

Main function:
    ingest_company(ticker, company_name)

This is the only function you need to call to get a company's documents
into the system. The Java gateway will call the /ingest REST endpoint
which calls this function.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import uuid

from fetcher  import fetch_10k, fetch_news
from chunker  import chunk_documents
from embedder import embedder
from config   import settings


# ── Qdrant client (connects to local or remote Qdrant instance) ───────────────

client = QdrantClient(url=settings.QDRANT_URL)


def _ensure_collection():
    """
    Create the Qdrant collection if it doesn't exist yet.

    A Qdrant collection is like a table in a relational database —
    it holds all your vectors plus their metadata (called 'payload' in Qdrant).

    We configure it once with the vector size (must match our embedding model)
    and distance metric (Cosine — standard for text similarity).

    Cosine similarity measures the angle between two vectors.
    Two chunks about the same topic → vectors point in same direction → angle ≈ 0 → score ≈ 1.
    Two unrelated chunks → vectors point in different directions → score ≈ 0.
    This is the same cosine similarity you've seen in ML courses.
    """
    existing = [c.name for c in client.get_collections().collections]

    if settings.QDRANT_COLLECTION not in existing:
        print(f"[Pipeline] Creating Qdrant collection '{settings.QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBED_DIMENSIONS,   # 384 for bge-small
                distance=Distance.COSINE,
            ),
        )
        print("[Pipeline] Collection created.")
    else:
        print(f"[Pipeline] Collection '{settings.QDRANT_COLLECTION}' already exists.")


def _store_chunks(chunk_vector_pairs: list[tuple]) -> int:
    """
    Store (chunk, vector) pairs in Qdrant as Points.

    A Qdrant Point has three parts:
        id      — a unique identifier (we use UUID)
        vector  — the embedding float list
        payload — any metadata dict (company, year, section, text, etc.)

    We upload in batches of 100 to avoid memory issues with large ingestions.
    Returns the count of stored points.
    """
    UPLOAD_BATCH = 100
    points       = []

    for chunk, vector in chunk_vector_pairs:
        point = PointStruct(
            id=str(uuid.uuid4()),     # random unique ID for each chunk
            vector=vector,
            payload={
                **chunk.metadata,    # company, year, doc_type, section_name, etc.
                "text": chunk.text,  # store the raw text so we can return it at query time
            }
        )
        points.append(point)

    total = len(points)
    for i in range(0, total, UPLOAD_BATCH):
        batch = points[i : i + UPLOAD_BATCH]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=batch,
        )
        print(f"[Pipeline] Stored {min(i + UPLOAD_BATCH, total)}/{total} points...")

    return total


def _delete_existing(ticker: str, doc_type: str):
    """
    Delete any existing points for this company + doc_type before re-ingesting.

    This prevents duplicate chunks if you run ingestion twice for the same company.
    Qdrant's Filter system works like a WHERE clause in SQL:
        DELETE FROM finrag_docs WHERE company = 'AAPL' AND doc_type = '10-K'
    """
    print(f"[Pipeline] Removing existing {doc_type} chunks for {ticker}...")
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(key="company",  match=MatchValue(value=ticker.upper())),
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type)),
            ]
        ),
    )


def ingest_company(ticker: str, company_name: str) -> dict:
    """
    Full ingestion pipeline for one company.

    Fetches 10-K + news, chunks, embeds, and stores everything in Qdrant.
    Returns a summary dict that the FastAPI router will send back to Java.

    Usage:
        result = ingest_company("AAPL", "Apple")
        # {"ticker": "AAPL", "chunks_stored": 412, "status": "ok"}
    """
    print(f"\n{'='*60}")
    print(f"[Pipeline] Starting ingestion for {ticker} ({company_name})")
    print(f"{'='*60}")

    _ensure_collection()
    total_stored = 0

    # ── Step 1: 10-K filing ───────────────────────────────────────────────────
    try:
        print("\n[Pipeline] --- 10-K filing ---")
        raw_10k    = fetch_10k(ticker)
        chunks_10k = chunk_documents(raw_10k)
        pairs_10k  = embedder.embed_documents(chunks_10k)

        _delete_existing(ticker, "10-K")
        stored_10k  = _store_chunks(pairs_10k)
        total_stored += stored_10k
        print(f"[Pipeline] 10-K done: {stored_10k} chunks stored.")

    except Exception as e:
        print(f"[Pipeline] WARNING: 10-K ingestion failed for {ticker}: {e}")

    # ── Step 2: News articles ─────────────────────────────────────────────────
    try:
        print("\n[Pipeline] --- News articles ---")
        raw_news    = fetch_news(company_name, ticker)
        chunks_news = chunk_documents(raw_news)
        pairs_news  = embedder.embed_documents(chunks_news)

        _delete_existing(ticker, "news")
        stored_news  = _store_chunks(pairs_news)
        total_stored += stored_news
        print(f"[Pipeline] News done: {stored_news} chunks stored.")

    except Exception as e:
        print(f"[Pipeline] WARNING: News ingestion failed for {ticker}: {e}")

    print(f"\n[Pipeline] Ingestion complete for {ticker}. "
          f"Total chunks stored: {total_stored}")

    return {
        "ticker":        ticker.upper(),
        "company":       company_name,
        "chunks_stored": total_stored,
        "status":        "ok",
    }
