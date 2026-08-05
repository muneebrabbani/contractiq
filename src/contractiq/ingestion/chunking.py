from __future__ import annotations

import tiktoken

from contractiq.ingestion.models import Chunk, SourceDocument

ENCODING = tiktoken.get_encoding("cl100k_base")


def split_text_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token windows. Shared by the fixed-window chunker below
    and the clause-aware chunker's oversized-clause fallback."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    tokens = ENCODING.encode(text)
    if not tokens:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        pieces.append(ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap

    return pieces


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        for piece in split_text_by_tokens(page.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-{chunk_index}",
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    page_number=page.page_number,
                    text=piece,
                    token_count=len(ENCODING.encode(piece)),
                )
            )
            chunk_index += 1

    return chunks
