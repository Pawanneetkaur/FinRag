"""
embedder.py — Converts text chunks into vector embeddings.

We use a local sentence-transformers model (no API call, no cost per token).
The model runs on your CPU — slow for very large batches, but fine for our
ingestion volumes. If you later need GPU speed, just add device="cuda".

The same EmbedderService instance must be used for both:
    - Ingestion: embedding chunks before storing in Qdrant
    - Query time: embedding the user's question before searching Qdrant

Main class:
    EmbedderService
        .embed_documents(chunks)  -> chunks with vectors attached (for ingestion)
        .embed_query(text)        -> single vector (for query time)
"""

import torch
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from config import settings


class EmbedderService:
    """
    Wraps a sentence-transformers model with batch embedding and
    a progress log so you can see it working during ingestion.

    Why a class instead of plain functions?
        Loading the model from disk takes ~1-2 seconds. By keeping it
        as an instance variable we load it once and reuse it for every
        call — same pattern as loading a TensorFlow model once then
        calling model.predict() many times.
    """

    def __init__(self):
        print(f"[Embedder] Loading model '{settings.EMBED_MODEL_NAME}'...")

        # SentenceTransformer automatically downloads the model on first run
        # and caches it at ~/.cache/torch/sentence_transformers/
        self.model = SentenceTransformer(settings.EMBED_MODEL_NAME)

        # Detect if a GPU is available — use it if so, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        print(f"[Embedder] Model loaded. Running on {self.device}. "
              f"Output dimensions: {settings.EMBED_DIMENSIONS}")

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string at search time.

        Returns a plain Python list of floats (what Qdrant expects).

        The 'query_instruction' prefix is specific to BGE models —
        they were trained to prepend this string to queries (but NOT
        to documents) to improve retrieval accuracy. Other models
        don't need this.
        """
        # BGE models expect this prefix on queries during retrieval
        prefixed = f"Represent this sentence for searching relevant passages: {text}"

        vector = self.model.encode(
            prefixed,
            normalize_embeddings=True,   # L2-normalize so cosine similarity = dot product
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_documents(self, chunks: list[Document]) -> list[tuple[Document, list[float]]]:
        """
        Embed a list of chunk Documents in batches.

        Returns a list of (Document, vector) pairs — the Document carries
        the text and metadata, the vector is what goes into Qdrant.

        Batching matters: encoding one text at a time is ~10x slower than
        encoding 64 at once because the GPU/CPU can parallelise the matrix
        operations. Same reason TensorFlow uses batch_size in model.fit().
        """
        BATCH_SIZE = 64   # tune down to 32 if you run out of RAM

        texts   = [chunk.text for chunk in chunks]
        vectors = []

        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(texts), BATCH_SIZE):
            batch_num  = i // BATCH_SIZE + 1
            batch_text = texts[i : i + BATCH_SIZE]

            print(f"[Embedder] Batch {batch_num}/{total_batches} "
                  f"({len(batch_text)} chunks)...")

            # Documents don't get the query prefix — only queries do
            batch_vectors = self.model.encode(
                batch_text,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors.extend(batch_vectors.tolist())

        print(f"[Embedder] Embedded {len(vectors)} chunks. "
              f"Vector shape: [{len(vectors[0])}]")

        # Pair each chunk with its vector so the pipeline can store both together
        return list(zip(chunks, vectors))


# ── Module-level singleton ────────────────────────────────────────────────────
# We instantiate once here so any file that does:
#     from embedder import embedder
# gets the same already-loaded model, not a fresh load each time.
embedder = EmbedderService()
