# FinRAG — Financial RAG System

A RAG (Retrieval-Augmented Generation) system for portfolio managers and
retail investors to ask natural language questions about financial documents.

## Architecture

Two services:

### finrag-rag (Python, port 8000)
FastAPI service handling all AI logic:
- Ingestion: fetches 10-Ks from SEC EDGAR + news from NewsAPI, chunks,
  embeds with OpenAI text-embedding-3-small, stores in Qdrant
- Retrieval: per-company vector search, cross-encoder reranking, Claude generation

### finrag-gateway (Java Spring Boot, port 8080)
REST gateway handling auth and routing:
- JWT authentication (HS256)
- Forwards /api/query and /api/ingest to the Python service

### Qdrant (port 6333)
Vector database storing document embeddings with metadata.

## Key Files

```
finrag-rag/
  app/ingestion/   fetcher.py, chunker.py, embedder.py, pipeline.py
  app/retrieval/   query_engine.py, reranker.py, synthesizer.py
  app/routers/     ingest.py, query.py
  main.py          FastAPI entry point
  config.py        Settings (reads from .env)

finrag-gateway/
  src/.../controller/   QueryController, IngestController, AuthController
  src/.../security/     JwtUtil, AuthFilter, SecurityConfig
  src/.../service/      RagClient
```

## Environment Variables

Required in `finrag-rag/.env`:
- `ANTHROPIC_API_KEY` — Claude API key
- `OPENAI_API_KEY` — for text-embedding-3-small
- `QDRANT_URL` — Qdrant instance URL
- `QDRANT_API_KEY` — Qdrant Cloud API key
- `NEWS_API_KEY` — newsapi.org key
- `EDGAR_USER_AGENT` — e.g. "FinRAG your@email.com"

Required in environment for gateway:
- `FINRAG_JWT_SECRET` — min 32 chars
- `FINRAG_RAG_SERVICE_URL` — URL of Python service

## Running Locally

```bash
docker-compose up
```

## API Endpoints

### Auth
POST /auth/login
{"username": "finrag-user", "password": "finrag-demo-2024"}

### Ingest a company
POST /api/ingest  (requires Bearer token)
{"ticker": "AAPL", "company_name": "Apple"}

### Ask a question
POST /api/query  (requires Bearer token)
{"query": "Compare Apple and Microsoft revenue growth", "tickers": ["AAPL", "MSFT"]}

## Stack
- Python 3.11, FastAPI, LlamaIndex, OpenAI embeddings, sentence-transformers
- Java 21, Spring Boot 3, Spring Security, JJWT
- Qdrant vector database
- Claude (claude-sonnet-4-20250514) for generation
