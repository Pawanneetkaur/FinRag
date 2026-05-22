# FinRAG — Financial RAG System

A RAG system for portfolio managers and retail investors to ask natural 
language questions about financial documents (10-Ks, earnings transcripts, news).

## Stack
- Python 3.11, FastAPI, LlamaIndex (RAG service)
- Java 21, Spring Boot 3 (REST gateway)
- Qdrant (vector database)
- Llama 3.2 via Groq API (LLM)
- BAAI/bge-small-en-v1.5 (embeddings, local)

## Status
- [x] Ingestion pipeline (fetch, chunk, embed, store)
- [x] Retrieval layer (vector search, reranker, synthesizer)
- [x] Java gateway (auth, routing)
- [ ] LLM integration (Groq + Llama 3.2)
- [ ] Deployment

## Docs
See individual service READMEs for setup instructions.
