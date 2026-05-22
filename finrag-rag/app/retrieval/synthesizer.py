"""
synthesizer.py — Builds a prompt from retrieved chunks and calls Llama 3.2 via Groq.

Groq runs Llama 3.2 on their servers for free — extremely fast inference.
We replaced the Anthropic Claude API with Groq here. Everything else
(retrieval, reranking, prompt design) stays exactly the same.

Main class:
    Synthesizer
        .generate(query, chunks, tickers) -> FinRagResponse
"""

from groq import Groq
from dataclasses import dataclass
from app.retrieval.query_engine import RetrievedChunk
from config import settings


@dataclass
class FinRagResponse:
    answer:      str
    sources:     list[dict]
    companies:   list[str]
    chunks_used: int
    model:       str


class Synthesizer:

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model  = "llama-3.2-3b-preview"
        print(f"[Synthesizer] Groq client ready. Model: {self.model}")

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[SOURCE {i}]\n"
                f"Company: {chunk.company} | "
                f"Document: {chunk.doc_type} | "
                f"Year: {chunk.year or 'N/A'} | "
                f"Section: {chunk.section_name}\n"
                f"{'─' * 40}\n"
                f"{chunk.text}\n"
                f"{'=' * 40}"
            )
            lines.append(header)
        return "\n\n".join(lines)

    def _build_system_prompt(self, tickers: list[str]) -> str:
        is_comparison = len(tickers) > 1
        companies_str = ", ".join(tickers) if tickers else "the relevant companies"

        base = f"""You are FinRAG, a financial research assistant for portfolio managers and investors.
You are answering a question about: {companies_str}

STRICT GROUNDING RULES:
1. Base your answer ONLY on the numbered sources provided. Do NOT use outside knowledge.
2. Cite every factual claim with its source number in brackets, e.g. [Source 1].
3. If sources lack enough information, say: "The available documents do not contain sufficient information to answer this question."
4. Never invent numbers, dates, or statistics.
5. Use precise financial language.

OUTPUT FORMAT:"""

        if is_comparison:
            base += f"""
Structure your answer as:
    ## [Company 1]
    [findings with citations]

    ## [Company 2]
    [findings with citations]

    ## Comparative Summary
    [direct comparison]"""
        else:
            base += "\nWrite a clear, well-structured answer with citations."

        return base

    def _build_user_prompt(self, query: str, context_block: str) -> str:
        return f"""Here are the relevant excerpts from financial documents:

{context_block}

─────────────────────────────────────────────────────────
QUESTION: {query}
─────────────────────────────────────────────────────────

Answer using only the sources above. Cite every claim."""

    def generate(
        self,
        query:   str,
        chunks:  list[RetrievedChunk],
        tickers: list[str] | None = None,
    ) -> FinRagResponse:

        tickers = tickers or []

        if not chunks:
            return FinRagResponse(
                answer="No relevant documents found. Please ingest the companies first.",
                sources=[],
                companies=tickers,
                chunks_used=0,
                model=self.model,
            )

        print(f"[Synthesizer] Building prompt with {len(chunks)} chunks...")

        context_block = self._build_context_block(chunks)
        system_prompt = self._build_system_prompt(tickers)
        user_prompt   = self._build_user_prompt(query, context_block)

        print(f"[Synthesizer] Calling Llama 3.2 via Groq...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.1,
        )

        answer = response.choices[0].message.content
        print(f"[Synthesizer] Response received. "
              f"Tokens used: {response.usage.total_tokens}")

        sources = [
            {
                "source_number": i + 1,
                "company":       chunk.company,
                "doc_type":      chunk.doc_type,
                "year":          chunk.year,
                "section":       chunk.section_name,
                "relevance":     round(chunk.score, 3),
                "excerpt":       chunk.text[:200] + "...",
            }
            for i, chunk in enumerate(chunks)
        ]

        return FinRagResponse(
            answer=      answer,
            sources=     sources,
            companies=   list(set(chunk.company for chunk in chunks)),
            chunks_used= len(chunks),
            model=       self.model,
        )


synthesizer = Synthesizer()
