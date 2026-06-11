"""
reranker.py — Re-scores retrieved chunks using a cross-encoder model.

Two-stage retrieval is standard in production RAG systems:
    Stage 1 (query_engine.py): Fast vector search → top 20 candidates
    Stage 2 (this file):       Slow but accurate reranking → top 6 to send to Claude

Why two stages?
    Vector search (bi-encoder) encodes the query and document independently,
    then compares their vectors. It's fast because vectors are pre-computed.

    Reranking (cross-encoder) sees the query AND the document together in
    one forward pass — like reading both side by side. This gives much better
    relevance scores but is ~100x slower, so we only run it on the top-N
    candidates from stage 1, not the full database.

    Analogy from your ML background:
    - Bi-encoder = approximate nearest neighbour (fast, good recall)
    - Cross-encoder = exact scoring (slow, high precision)

Main class:
    Reranker
        .rerank(query, chunks) -> top-N RetrievedChunk objects
"""

from sentence_transformers import CrossEncoder
from app.retrieval.query_engine import RetrievedChunk
from config import settings


class Reranker:
    """
    Wraps a cross-encoder model for reranking retrieved chunks.

    Model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
        - Trained on MS MARCO (Microsoft's passage retrieval dataset)
        - Fast (MiniLM = distilled, 6-layer transformer)
        - Scores query-passage pairs on a scale that higher = more relevant
        - Downloads once and caches like the embedding model
    """

    def __init__(self):
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"[Reranker] Loading cross-encoder '{model_name}'...")
        self.model = CrossEncoder(model_name)
        print("[Reranker] Cross-encoder loaded.")

    def rerank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_n:  int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank a list of retrieved chunks and return the top-N most relevant.

        Args:
            query:  The user's original question.
            chunks: Candidates from vector search (typically 20).
            top_n:  How many to return after reranking.
                    Defaults to settings.RERANK_TOP_N (6).

        Returns:
            A shorter list of RetrievedChunk, reordered by cross-encoder score.

        How cross-encoder scoring works:
            For each chunk, we create an input pair: [query, chunk_text].
            The model reads both together and outputs a single relevance score.
            Higher score = the chunk better answers the query.

            This is different from cosine similarity — the cross-encoder
            explicitly models the *interaction* between query and passage,
            catching things like:
            - A chunk that shares keywords with the query but is off-topic
            - A chunk that answers the query without sharing exact words
        """
        if not chunks:
            return []

        top_n = top_n or settings.RERANK_TOP_N

        print(f"[Reranker] Reranking {len(chunks)} chunks → top {top_n}...")

        # Build (query, passage) pairs for the cross-encoder
        # The model expects exactly this format
        pairs = [(query, chunk.text) for chunk in chunks]

        # Run cross-encoder inference — this is the slow step
        # scores is a numpy array of floats, one per pair
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach cross-encoder scores to chunks and sort
        # We preserve the original chunk objects — just reorder them
        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Log the top results so you can see what's being passed to Claude
        print(f"[Reranker] Top {min(top_n, len(scored))} after reranking:")
        for i, (chunk, score) in enumerate(scored[:top_n]):
            print(f"  {i+1}. [{chunk.company} {chunk.doc_type} {chunk.year}] "
                  f"score={score:.3f} | {chunk.text[:80].strip()}...")

        # Return just the RetrievedChunk objects (drop the scores —
        # the original vector score is still on chunk.score if needed)
        top_chunks = [chunk for chunk, _ in scored[:top_n]]

        return top_chunks


# Module-level singleton
reranker = Reranker()
