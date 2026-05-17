"""
query_engine.py — Searches Qdrant for chunks relevant to a user's question.

The key design decision here is per-company retrieval:
    Instead of one global search, we search once per company mentioned
    in the query, then merge results. This prevents one company's documents
    from dominating the results in a comparison query.

Main class:
    QueryEngine
        .retrieve(query, tickers) -> list of RetrievedChunk
"""

from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from embedder import embedder
from config   import settings


@dataclass
class RetrievedChunk:
    """
    A single retrieved passage with its metadata and relevance score.

    We use a dataclass (like a lightweight struct) instead of a raw dict
    so the rest of the code gets autocomplete and type safety.
    """
    text:         str
    score:        float    # cosine similarity score, 0.0 to 1.0
    company:      str
    doc_type:     str
    year:         int | None
    section_name: str
    source:       str
    chunk_index:  int


class QueryEngine:
    """
    Handles vector search against Qdrant.

    Supports two retrieval modes:
        1. Multi-company (default): retrieve top-K per company, then merge
        2. Single-company: standard retrieval with an optional company filter
    """

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        print("[QueryEngine] Connected to Qdrant.")

    def _search_one_company(
        self,
        query_vector: list[float],
        ticker:       str,
        top_k:        int,
    ) -> list[RetrievedChunk]:
        """
        Search Qdrant for the top-K most similar chunks for one specific company.

        The Filter here is like:
            SELECT * FROM finrag_docs
            WHERE company = 'AAPL'
            ORDER BY cosine_similarity(vector, query_vector) DESC
            LIMIT top_k

        Qdrant does this efficiently using HNSW indexing — an approximate
        nearest-neighbour algorithm. Think of it like a spatial index
        (similar to a KD-tree you may have seen in ML) but optimised for
        high-dimensional vectors.
        """
        company_filter = Filter(
            must=[
                FieldCondition(
                    key="company",
                    match=MatchValue(value=ticker.upper()),
                )
            ]
        )

        results = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=company_filter,
            limit=top_k,
            with_payload=True,    # include the metadata + text in results
        )

        chunks = []
        for r in results:
            p = r.payload   # payload is the dict we stored at ingestion time
            chunks.append(RetrievedChunk(
                text=         p.get("text", ""),
                score=        r.score,
                company=      p.get("company", ticker),
                doc_type=     p.get("doc_type", ""),
                year=         p.get("year"),
                section_name= p.get("section_name", ""),
                source=       p.get("source", ""),
                chunk_index=  p.get("chunk_index", 0),
            ))

        return chunks

    def _search_global(
        self,
        query_vector: list[float],
        top_k:        int,
    ) -> list[RetrievedChunk]:
        """
        Search across all companies with no filter.
        Used when no specific tickers are mentioned in the query.
        """
        results = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for r in results:
            p = r.payload
            chunks.append(RetrievedChunk(
                text=         p.get("text", ""),
                score=        r.score,
                company=      p.get("company", ""),
                doc_type=     p.get("doc_type", ""),
                year=         p.get("year"),
                section_name= p.get("section_name", ""),
                source=       p.get("source", ""),
                chunk_index=  p.get("chunk_index", 0),
            ))

        return chunks

    def retrieve(
        self,
        query:   str,
        tickers: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Main retrieval function. Called by synthesizer.py.

        Args:
            query:   The user's natural language question.
            tickers: List of company tickers mentioned in the query.
                     e.g. ["AAPL", "MSFT"] for a comparison query.
                     Pass None or [] to search across all companies.

        Returns:
            A merged, deduplicated list of RetrievedChunk objects,
            sorted by score descending.

        Per-company retrieval logic:
            If tickers = ["AAPL", "MSFT"], we retrieve top-K for AAPL
            and top-K for MSFT separately, then merge both lists.
            Each company gets an equal budget of chunks — this is what
            prevents one company from dominating the results.
        """
        print(f"[QueryEngine] Retrieving for query: '{query[:80]}...'")
        print(f"[QueryEngine] Tickers: {tickers or 'all companies'}")

        # Step 1: embed the query using the same model used at ingestion time
        # (critical — mismatched models = wrong vector space = bad results)
        query_vector = embedder.embed_query(query)

        # Step 2: retrieve chunks
        if tickers and len(tickers) > 0:
            # Per-company retrieval: search each company separately
            # Divide the total budget evenly across companies
            per_company_k = max(
                settings.RETRIEVAL_TOP_K // len(tickers),
                5,   # minimum 5 chunks per company even if many tickers
            )

            all_chunks = []
            for ticker in tickers:
                print(f"[QueryEngine] Searching {ticker} "
                      f"(top {per_company_k})...")
                company_chunks = self._search_one_company(
                    query_vector, ticker, per_company_k
                )
                print(f"[QueryEngine]   → {len(company_chunks)} chunks "
                      f"(best score: "
                      f"{company_chunks[0].score:.3f if company_chunks else 'n/a'})")
                all_chunks.extend(company_chunks)

        else:
            # No tickers specified — global search
            all_chunks = self._search_global(
                query_vector, settings.RETRIEVAL_TOP_K
            )

        # Step 3: sort merged results by score descending
        # After merging, an AAPL chunk with score 0.91 will rank above
        # an MSFT chunk with score 0.78, which is correct — we just
        # guaranteed both companies are represented in the pool.
        all_chunks.sort(key=lambda c: c.score, reverse=True)

        print(f"[QueryEngine] Retrieved {len(all_chunks)} total chunks "
              f"across {len(set(c.company for c in all_chunks))} companies.")

        return all_chunks


# Module-level singleton — same pattern as embedder.py
query_engine = QueryEngine()
