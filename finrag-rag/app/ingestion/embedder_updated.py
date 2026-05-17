"""
embedder.py — Converts text chunks into vector embeddings using OpenAI API.

We use OpenAI's text-embedding-3-small instead of a local model because:
    - We're running in the cloud (Railway) — no benefit to local models
    - No heavy torch/sentence-transformers dependency to install
    - Faster, better quality, costs almost nothing (~$0.02/million tokens)
    - One 10-K filing ≈ ~500K chars ≈ ~125K tokens ≈ $0.0025 to embed

The same EmbedderService must be used for both:
    - Ingestion: embedding chunks before storing in Qdrant
    - Query time: embedding the user's question before searching Qdrant
"""

from openai import OpenAI
from llama_index.core import Document
from config import settings


class EmbedderService:
    """
    Wraps the OpenAI embeddings API with batching and logging.

    Loading the client once and reusing it is more efficient than
    creating a new client per request.
    """

    def __init__(self):
        self.client     = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model      = settings.EMBED_MODEL_NAME   # text-embedding-3-small
        self.dimensions = settings.EMBED_DIMENSIONS   # 1536
        print(f"[Embedder] OpenAI embeddings ready. "
              f"Model: {self.model}, Dimensions: {self.dimensions}")

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string at search time.
        Returns a plain Python list of floats (what Qdrant expects).
        """
        response = self.client.embeddings.create(
            input=  text,
            model=  self.model,
        )
        return response.data[0].embedding

    def embed_documents(self, chunks: list[Document]) -> list[tuple]:
        """
        Embed a list of chunk Documents in batches of 100.

        OpenAI allows up to 2048 inputs per request — we use 100
        to stay well within limits and keep memory usage low.

        Returns a list of (Document, vector) pairs.
        """
        BATCH_SIZE = 100
        texts      = [chunk.text for chunk in chunks]
        vectors    = []

        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(texts), BATCH_SIZE):
            batch_num  = i // BATCH_SIZE + 1
            batch_text = texts[i : i + BATCH_SIZE]

            print(f"[Embedder] Batch {batch_num}/{total_batches} "
                  f"({len(batch_text)} chunks)...")

            response = self.client.embeddings.create(
                input= batch_text,
                model= self.model,
            )

            # OpenAI returns embeddings in the same order as input
            batch_vectors = [item.embedding for item in response.data]
            vectors.extend(batch_vectors)

        print(f"[Embedder] Embedded {len(vectors)} chunks. "
              f"Vector dimensions: {len(vectors[0])}")

        return list(zip(chunks, vectors))


# Module-level singleton
embedder = EmbedderService()
