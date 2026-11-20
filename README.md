# FinRAG

A retrieval-augmented question-answering system over SEC filings. Ask a plain-English question and get a source-cited answer.

## Architecture
- **finrag-rag** (Python): ingestion, chunking, embeddings, and retrieval with LangChain and ChromaDB, using hybrid vector + keyword search
- **finrag-gateway** (Java): API gateway that fronts the RAG service
- **LLM**: Llama 3.2 served via Groq
- **docker-compose**: runs the full stack locally

## Why
Financial answers need receipts. Every response is grounded in retrieved filing passages, so it can be traced back to the source.

## Run locally
1. Set `GROQ_API_KEY` in your environment
2. `docker-compose up`

## Status
Actively developed. Next up: expanding the evaluation suite.
