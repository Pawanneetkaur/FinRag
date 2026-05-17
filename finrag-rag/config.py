"""
config.py — Central settings for the FinRAG Python service.

Reads values from environment variables or a .env file in the project root.
Never put real API keys in this file — put them in .env (which is git-ignored).

Usage anywhere in the codebase:
    from config import settings
    print(settings.QDRANT_URL)
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── Anthropic (Claude) ────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────
    # EDGAR requires a User-Agent header identifying your app.
    # Format: "AppName your@email.com"  (no auth token needed)
    EDGAR_USER_AGENT: str = "FinRAG finrag@example.com"

    # ── NewsAPI ───────────────────────────────────────────────────────────────
    # Free tier at newsapi.org — 100 requests/day, 1 month history
    NEWS_API_KEY: str = ""

    # ── Qdrant ────────────────────────────────────────────────────────────────
    # When running locally via docker-compose, this is http://localhost:6333
    QDRANT_URL:        str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "finrag_docs"

    # ── Embedding model ───────────────────────────────────────────────────────
    # We use a small, fast sentence-transformers model that runs locally —
    # no API call needed for embeddings, which keeps costs down.
    # "BAAI/bge-small-en-v1.5" produces 384-dimensional vectors.
    # Larger option: "BAAI/bge-large-en-v1.5" (1024-dim, slower, more accurate)
    EMBED_MODEL_NAME: str  = "BAAI/bge-small-en-v1.5"
    EMBED_DIMENSIONS: int  = 384

    # ── Chunking ──────────────────────────────────────────────────────────────
    # CHUNK_SIZE: how many characters per chunk.
    # Too large  → chunk has multiple topics, retrieval gets confused.
    # Too small  → chunk loses context, answer is incomplete.
    # 1500 chars ≈ ~250 words ≈ one or two paragraphs — good for financial text.
    CHUNK_SIZE:    int = 1500
    CHUNK_OVERLAP: int = 200   # overlap so sentences don't get cut mid-thought

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # How many chunks to pull from Qdrant before reranking
    RETRIEVAL_TOP_K: int = 20
    # How many chunks to pass to Claude after reranking
    RERANK_TOP_N:    int = 6

    # ── FastAPI ───────────────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    class Config:
        # Automatically load from a .env file if it exists
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere
settings = Settings()
