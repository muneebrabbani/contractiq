from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from contractiq.extraction.models import ClauseChunk, RedactedDocument, RedactedPage
from contractiq.extraction.redaction import load_redacted_document
from contractiq.ingestion.chunking import ENCODING, split_text_by_tokens

PROCESSED_DIR = Path("data/processed")

# Multi-level numbers ("4.1", "10.2.1") are distinctive enough on their own;
# the trailing period is optional. Single-level numbers ("4") require a
# literal trailing period before the whitespace -- without it, ordinary
# prose like "30 days written notice" or "10% of the total value" would
# false-positive as clause boundaries "30." / "10.".
_MULTI_LEVEL_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(.+)$")
_SINGLE_LEVEL_RE = re.compile(r"^(\d+)\.\s+(.+)$")
_SECTION_WORD_RE = re.compile(r"^(?:Section|SECTION|Article|ARTICLE)\s+(\d+|[IVXLCM]+)\.?\s+(.+)$")

HEADING_MAX_WORDS = 10
MAX_CHUNK_TOKENS = 800
FALLBACK_OVERLAP = 50


def _is_heading_like(remainder: str) -> bool:
    """A heading ("TERMINATION", "Governing Law") vs. a short one-line
    clause body ("Time is of the essence.") can't be told apart by length
    alone -- plenty of real clauses are short. Terminal punctuation is the
    stronger signal: clause bodies are sentences and almost always end in
    '.', '!', or '?'; headings almost never do."""
    if not remainder or remainder[-1] in ".!?":
        return False
    return len(remainder.split()) <= HEADING_MAX_WORDS


def _match_boundary(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line:
        return None
    for pattern in (_SECTION_WORD_RE, _MULTI_LEVEL_RE, _SINGLE_LEVEL_RE):
        m = pattern.match(line)
        if m:
            remainder = m.group(2).lstrip("-:–— \t")
            return m.group(1), remainder
    return None


def _iter_lines_with_pages(pages: list[RedactedPage]) -> list[tuple[int, str]]:
    lines = []
    for page in pages:
        for line in page.text.split("\n"):
            lines.append((page.page_number, line))
    return lines


@dataclass
class _RawChunk:
    page: int
    clause_number: str | None
    heading: str | None = None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _build_raw_chunks(pages: list[RedactedPage]) -> list[_RawChunk]:
    """One raw chunk per boundary line. A boundary whose own remainder is
    short/title-like (e.g. "TERMINATION" in "1. TERMINATION") is recorded as
    a heading rather than folded into the chunk's body text, so that heading
    can be distinguished from a numbered clause whose body starts on the
    same line (e.g. "2. Governing Law" immediately followed on the next line
    by the clause's own prose, with no separate numbered sub-clause)."""
    raw_chunks: list[_RawChunk] = []
    current: _RawChunk | None = None

    for page_number, line in _iter_lines_with_pages(pages):
        boundary = _match_boundary(line)
        if boundary:
            clause_number, remainder = boundary
            if _is_heading_like(remainder):
                current = _RawChunk(page=page_number, clause_number=clause_number, heading=remainder)
            else:
                current = _RawChunk(page=page_number, clause_number=clause_number)
                current.lines.append(remainder)
            raw_chunks.append(current)
        else:
            if current is None:
                current = _RawChunk(page=page_number, clause_number=None)
                raw_chunks.append(current)
            current.lines.append(line)

    return raw_chunks


def _merge_headings(raw_chunks: list[_RawChunk]) -> list[tuple[_RawChunk, str | None]]:
    """Track the running section title from headings, and drop chunks that
    are purely a heading with no body of their own (their next boundary line
    was another boundary, so nothing was ever appended to them)."""
    result: list[tuple[_RawChunk, str | None]] = []
    current_section_title: str | None = None

    for chunk in raw_chunks:
        if chunk.heading is not None:
            current_section_title = chunk.heading
            if not chunk.lines:
                continue  # pure heading, no body -> don't emit as its own chunk
        if not chunk.text:
            continue
        result.append((chunk, current_section_title))

    return result


def chunk_document(
    document: RedactedDocument,
    max_tokens: int = MAX_CHUNK_TOKENS,
    fallback_overlap: int = FALLBACK_OVERLAP,
) -> list[ClauseChunk]:
    raw_chunks = _build_raw_chunks(document.pages)
    merged = _merge_headings(raw_chunks)
    source_file = Path(document.file_path).name

    chunks: list[ClauseChunk] = []
    chunk_index = 0

    for raw, section_title in merged:
        token_count = len(ENCODING.encode(raw.text))
        pieces = (
            [raw.text]
            if token_count <= max_tokens
            else split_text_by_tokens(raw.text, max_tokens, fallback_overlap)
        )

        for piece in pieces:
            chunks.append(
                ClauseChunk(
                    chunk_id=f"{document.doc_id}-{chunk_index}",
                    doc_id=document.doc_id,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    clause_number=raw.clause_number,
                    section_title=section_title,
                    page=raw.page,
                    text=piece,
                    token_count=len(ENCODING.encode(piece)),
                )
            )
            chunk_index += 1

    return chunks


def load_all_clause_chunks(input_dir: Path = PROCESSED_DIR) -> list[ClauseChunk]:
    """Load every redacted document under input_dir and clause-chunk it.
    Shared corpus-loading logic for anything that needs the full chunked
    corpus (the eval harness's stub pipeline, retrieval indexing)."""
    chunks: list[ClauseChunk] = []
    for path in sorted(input_dir.glob("*.redacted.json")):
        document = load_redacted_document(path)
        chunks.extend(chunk_document(document))
    return chunks
