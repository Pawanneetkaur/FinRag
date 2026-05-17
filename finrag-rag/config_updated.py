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

    # ── OpenAI (embeddings) ───────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""

    # ── SEC EDGAR ─────────────────────────────────────────────────────────────
    EDGAR_USER_AGENT: str = "FinRAG finrag@example.com"

    # ── NewsAPI ───────────────────────────────────────────────────────────────
    NEWS_API_KEY: str = ""

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_URL:        str = "http://localhost:6333"
    QDRANT_API_KEY:    str = ""    # required for Qdrant Cloud, empty for local
    QDRANT_COLLECTION: str = "finrag_docs"

    # ── Embedding model (OpenAI) ──────────────────────────────────────────────
    EMBED_MODEL_NAME: str = "text-embedding-3-small"
    EMBED_DIMENSIONS: int = 1536

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE:    int = 1500
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_N:    int = 6

    # ── FastAPI ───────────────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
