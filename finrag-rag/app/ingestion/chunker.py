"""
chunker.py — Splits large financial documents into smaller overlapping chunks.

Why chunking matters:
    Embedding models have a token limit (~512 tokens ≈ 400 words). A 10-K
    filing can be 100,000+ words. We split it into overlapping passages so:
    1. Each chunk fits in the embedding model.
    2. Overlap ensures sentences cut at a boundary still appear complete
       in the neighbouring chunk.
    3. Each chunk inherits the parent document's metadata (company, year, etc.)
       so we always know where a retrieved passage came from.

Main function:
    chunk_documents(documents) -> list of smaller Document objects
"""

import re
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from config import settings


def _clean_text(text: str) -> str:
    """
    Light cleaning for SEC filings — they contain a lot of HTML artifacts,
    excessive whitespace, and page-break markers when downloaded as raw text.

    We don't want to strip too aggressively — financial numbers and
    formatting carry meaning — so we only fix obvious noise.
    """
    # Collapse runs of whitespace/newlines into a single space or newline
    text = re.sub(r'\r\n', '\n', text)           # normalize Windows line endings
    text = re.sub(r'\n{3,}', '\n\n', text)       # max two consecutive newlines
    text = re.sub(r'[ \t]{2,}', ' ', text)       # collapse horizontal whitespace

    # Remove common HTML artifacts that sneak through in EDGAR filings
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'<[^>]+>', '', text)          # strip any remaining HTML tags

    # Remove EDGAR page markers like "-----BEGIN PRIVACY-ENHANCED MESSAGE-----"
    text = re.sub(r'-{5,}.*?-{5,}', '', text)

    return text.strip()


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """
    Try to detect named sections in a 10-K (e.g. "Item 1A. Risk Factors").

    Returns a list of (section_name, section_text) tuples.
    If no sections are detected, returns the whole text as one unnamed section.

    This is a best-effort heuristic — SEC filings aren't perfectly consistent.
    Having section names in metadata helps the LLM give more precise answers.
    """
    # Common 10-K section headers — Item 1, Item 1A, Item 7, etc.
    section_pattern = re.compile(
        r'(?i)(item\s+\d+[a-z]?\.\s+[A-Z][^\n]{5,80})',
        re.MULTILINE
    )
    matches = list(section_pattern.finditer(text))

    if len(matches) < 3:
        # Not enough structure detected — treat as one blob
        return [("full_document", text)]

    sections = []
    for i, match in enumerate(matches):
        section_name  = match.group(1).strip()
        section_start = match.end()
        section_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text  = text[section_start:section_end].strip()

        if len(section_text) > 200:   # skip near-empty sections
            sections.append((section_name, section_text))

    return sections if sections else [("full_document", text)]


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Take a list of large Documents (one per filing or article) and return
    a larger list of smaller chunk Documents ready for embedding.

    Each chunk Document:
    - Has text of ~CHUNK_SIZE characters with CHUNK_OVERLAP overlap
    - Inherits all metadata from its parent (company, year, doc_type, etc.)
    - Gets additional metadata: section_name, chunk_index

    Usage:
        from app.ingestion.fetcher import fetch_10k
        from app.ingestion.chunker import chunk_documents

        docs   = fetch_10k("AAPL")
        chunks = chunk_documents(docs)
        print(len(chunks))          # e.g. 340 chunks from one 10-K
        print(chunks[0].metadata)   # {'company': 'AAPL', 'year': 2023, 'section_name': 'Item 1A...', ...}
    """
    # LlamaIndex's SentenceSplitter respects sentence boundaries —
    # it tries not to break mid-sentence, unlike a naive character splitter.
    splitter = SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # Separator hierarchy: try to split on paragraphs first, then sentences
        paragraph_separator="\n\n",
    )

    all_chunks = []

    for doc in documents:
        print(f"[Chunker] Processing {doc.metadata.get('company')} "
              f"{doc.metadata.get('doc_type')} {doc.metadata.get('year', '')}...")

        # Step 1: clean the raw text
        clean = _clean_text(doc.text)

        # Step 2: detect sections (only meaningful for 10-K filings)
        if doc.metadata.get("doc_type") == "10-K":
            sections = _detect_sections(clean)
        else:
            # News articles don't have SEC sections — treat as one block
            sections = [("article", clean)]

        # Step 3: chunk each section independently
        # This keeps chunks within a single section — a chunk won't span
        # from "Risk Factors" into "Financial Statements"
        chunk_index = 0
        for section_name, section_text in sections:
            # Wrap section text in a temporary Document for the splitter
            section_doc = Document(text=section_text)
            nodes       = splitter.get_nodes_from_documents([section_doc])

            for node in nodes:
                # Build the chunk's metadata by merging parent metadata
                # with section-specific info
                chunk_metadata = {
                    **doc.metadata,                  # company, year, doc_type, source...
                    "section_name": section_name,
                    "chunk_index":  chunk_index,
                    "chunk_chars":  len(node.text),
                }

                chunk_doc = Document(
                    text=node.text,
                    metadata=chunk_metadata,
                )
                all_chunks.append(chunk_doc)
                chunk_index += 1

        print(f"[Chunker] → {chunk_index} chunks produced.")

    print(f"[Chunker] Total chunks across all documents: {len(all_chunks)}")
    return all_chunks
