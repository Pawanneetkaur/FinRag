"""
synthesizer.py — Builds a prompt from retrieved chunks and calls Claude.

This is the final step in the RAG pipeline. It takes the top-N reranked
chunks and constructs a prompt that instructs Claude to:
    1. Answer only from the provided context (no hallucination)
    2. Cite the source (company, document type, year) for each claim
    3. Explicitly say when information is missing rather than guessing

Main class:
    Synthesizer
        .generate(query, chunks, tickers) -> FinRagResponse
"""

import anthropic
from dataclasses import dataclass, field
from query_engine import RetrievedChunk
from config import settings


@dataclass
class FinRagResponse:
    """
    The structured response returned to the FastAPI router,
    which then returns it to the Java gateway as JSON.
    """
    answer:         str               # Claude's answer in plain text
    sources:        list[dict]        # list of source citations
    companies:      list[str]         # which companies were covered
    chunks_used:    int               # how many chunks Claude saw
    model:          str               # which Claude model was used


class Synthesizer:
    """
    Builds a grounded prompt and calls the Anthropic API.

    Prompt engineering decisions:
        - System prompt establishes Claude's role and strict grounding rules
        - Each chunk is labelled with company, doc type, year, section
          so Claude can cite precisely
        - We explicitly tell Claude to say "not enough information" rather
          than drawing on its training data — critical for financial accuracy
        - For comparison queries we ask Claude to structure the answer
          with one section per company before a comparative summary
    """

    def __init__(self):
        # Anthropic client reads ANTHROPIC_API_KEY from environment automatically
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model  = "claude-sonnet-4-20250514"
        print(f"[Synthesizer] Anthropic client ready. Model: {self.model}")

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a numbered context block for the prompt.

        Each chunk is wrapped with clear labels so Claude knows exactly
        where the information came from. This structure makes it easy for
        Claude to produce precise citations like [Source 3].

        Example output:
            [SOURCE 1]
            Company: AAPL | Document: 10-K | Year: 2023 | Section: Item 7...
            ---
            Net sales increased 3% year-over-year to $394.3 billion...
            ========
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"[SOURCE {i}]\n"
                f"Company: {chunk.company} | "
                f"Document: {chunk.doc_type} | "
                f"Year: {chunk.year or 'N/A'} | "
                f"Section: {chunk.section_name}\n"
                f"Relevance score: {chunk.score:.3f}\n"
                f"{'─' * 40}\n"
                f"{chunk.text}\n"
                f"{'=' * 40}"
            )
            lines.append(header)
        return "\n\n".join(lines)

    def _build_system_prompt(self, tickers: list[str]) -> str:
        """
        The system prompt defines Claude's role and hard rules.

        Key rules:
            1. Answer only from provided sources — never from training data
            2. Cite [Source N] for every factual claim
            3. Admit when information is missing rather than guessing
            4. For comparisons, structure the answer clearly by company
        """
        is_comparison = len(tickers) > 1
        companies_str = ", ".join(tickers) if tickers else "the relevant companies"

        base = f"""You are FinRAG, a financial research assistant for portfolio managers and investors.

You are answering a question about: {companies_str}

STRICT GROUNDING RULES — follow these exactly:
1. Base your answer ONLY on the numbered sources provided below the question.
   Do NOT use your training knowledge about these companies.
2. Cite every factual claim with its source number in brackets, e.g. [Source 1].
3. If the sources do not contain enough information to answer the question,
   say exactly: "The available documents do not contain sufficient information
   to answer this question about [topic]." Do not guess or extrapolate.
4. Never invent numbers, dates, or statistics. If a number is not in the sources, omit it.
5. Use precise financial language. Do not oversimplify for retail investors —
   they can handle terms like EBITDA, gross margin, YoY, etc.

OUTPUT FORMAT:"""

        if is_comparison:
            base += f"""
Structure your answer as follows for comparison queries:
    ## [Company 1 ticker]
    [findings for company 1 with citations]

    ## [Company 2 ticker]
    [findings for company 2 with citations]

    ## Comparative Summary
    [direct comparison across companies]

Keep each company section focused and parallel — cover the same dimensions
for each company so the comparison is fair and readable."""
        else:
            base += """
Write a clear, well-structured answer with citations.
Use bullet points or short paragraphs as appropriate for the content.
Lead with the most important finding."""

        return base

    def _build_user_prompt(self, query: str, context_block: str) -> str:
        """
        The user prompt contains the question and all the retrieved sources.

        We put the sources BEFORE the question (as recommended by Anthropic
        for long-context tasks) — this gives Claude time to process the
        context before seeing what it needs to answer.
        """
        return f"""Here are the relevant excerpts from financial documents:

{context_block}

─────────────────────────────────────────────────────────
QUESTION: {query}
─────────────────────────────────────────────────────────

Answer the question using only the sources above. Cite every claim."""

    def generate(
        self,
        query:   str,
        chunks:  list[RetrievedChunk],
        tickers: list[str] | None = None,
    ) -> FinRagResponse:
        """
        Build the prompt and call Claude to generate a grounded answer.

        Args:
            query:   The user's original question.
            chunks:  Top-N reranked chunks from reranker.py.
            tickers: Company tickers involved in the query.

        Returns:
            FinRagResponse with Claude's answer and source metadata.
        """
        tickers = tickers or []

        if not chunks:
            return FinRagResponse(
                answer="No relevant documents were found for your query. "
                       "Please make sure the companies have been ingested first.",
                sources=[],
                companies=tickers,
                chunks_used=0,
                model=self.model,
            )

        print(f"[Synthesizer] Building prompt with {len(chunks)} chunks...")

        context_block = self._build_context_block(chunks)
        system_prompt = self._build_system_prompt(tickers)
        user_prompt   = self._build_user_prompt(query, context_block)

        # Log approximate prompt size — useful for debugging token limits
        total_chars = len(system_prompt) + len(user_prompt)
        print(f"[Synthesizer] Prompt size: ~{total_chars:,} chars "
              f"(~{total_chars // 4:,} tokens)")

        print(f"[Synthesizer] Calling Claude ({self.model})...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )

        answer = response.content[0].text
        print(f"[Synthesizer] Response received. "
              f"Output tokens: {response.usage.output_tokens}")

        # Build source list for the API response
        # The Java gateway will include these so the frontend can show citations
        sources = [
            {
                "source_number": i + 1,
                "company":       chunk.company,
                "doc_type":      chunk.doc_type,
                "year":          chunk.year,
                "section":       chunk.section_name,
                "relevance":     round(chunk.score, 3),
                "excerpt":       chunk.text[:200] + "...",  # preview only
            }
            for i, chunk in enumerate(chunks)
        ]

        companies = list(set(chunk.company for chunk in chunks))

        return FinRagResponse(
            answer=      answer,
            sources=     sources,
            companies=   companies,
            chunks_used= len(chunks),
            model=       self.model,
        )


# Module-level singleton
synthesizer = Synthesizer()
