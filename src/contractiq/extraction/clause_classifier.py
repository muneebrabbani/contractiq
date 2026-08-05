from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import BaseModel

from contractiq.config import settings
from contractiq.extraction.models import ClauseChunk, ClauseType

logger = logging.getLogger(__name__)

# classify_clause_type's LLM fallback is one blocking HTTP call per chunk;
# run serially over a few thousand chunks (indexing.py's corpus re-index),
# that's the dominant cost by a wide margin -- almost all wall-clock time is
# spent waiting on network round-trips, not computing. This is I/O-bound, not
# CPU-bound, so a thread pool (not multiprocessing, not an async rewrite of
# the client) is the right tool: each classification call is independent, and
# the GIL isn't a bottleneck for I/O waits.
CLASSIFY_MAX_WORKERS = 8

_KEYWORD_MAP: list[tuple[re.Pattern, ClauseType]] = [
    (re.compile(r"terminat", re.IGNORECASE), ClauseType.TERMINATION),
    (re.compile(r"governing law|applicable law|choice of law", re.IGNORECASE), ClauseType.GOVERNING_LAW),
    (re.compile(r"payment|fees?\b|pricing|invoic", re.IGNORECASE), ClauseType.PAYMENT_TERMS),
    (re.compile(r"confidential|non-disclosure|nondisclosure", re.IGNORECASE), ClauseType.CONFIDENTIALITY),
    (re.compile(r"indemnif", re.IGNORECASE), ClauseType.INDEMNIFICATION),
    (re.compile(r"limitation of liability|limit.*liabilit", re.IGNORECASE), ClauseType.LIMITATION_OF_LIABILITY),
    (re.compile(r"notices?\b", re.IGNORECASE), ClauseType.NOTICES),
    (re.compile(r"warrant", re.IGNORECASE), ClauseType.WARRANTIES),
    (re.compile(r"assign", re.IGNORECASE), ClauseType.ASSIGNMENT),
    (re.compile(r"force majeure", re.IGNORECASE), ClauseType.FORCE_MAJEURE),
    (re.compile(r"dispute|arbitrat|venue|jurisdiction", re.IGNORECASE), ClauseType.DISPUTE_RESOLUTION),
    (re.compile(r"intellectual property|\bIP\b", re.IGNORECASE), ClauseType.INTELLECTUAL_PROPERTY),
    (re.compile(r"insurance", re.IGNORECASE), ClauseType.INSURANCE),
    (re.compile(r"data protection|privacy|personal data", re.IGNORECASE), ClauseType.DATA_PROTECTION),
    (re.compile(r"definitions?\b", re.IGNORECASE), ClauseType.DEFINITIONS),
    (re.compile(r"scope of work|statement of work|\bSOW\b", re.IGNORECASE), ClauseType.SCOPE_OF_WORK),
    (re.compile(r"entire agreement", re.IGNORECASE), ClauseType.ENTIRE_AGREEMENT),
    (re.compile(r"amendment|modification", re.IGNORECASE), ClauseType.AMENDMENT),
    (re.compile(r"severability", re.IGNORECASE), ClauseType.SEVERABILITY),
]


class _ClauseTypeGuess(BaseModel):
    clause_type: ClauseType


def _match_by_title(section_title: str | None) -> ClauseType | None:
    if not section_title:
        return None
    for pattern, clause_type in _KEYWORD_MAP:
        if pattern.search(section_title):
            return clause_type
    return None


def _classify_by_text_llm(chunk: ClauseChunk, client: OpenAI) -> ClauseType:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the following contract clause into exactly one clause type "
                    "from the given set. Choose 'other' if none clearly fit."
                ),
            },
            {"role": "user", "content": chunk.text[:2000]},
        ],
        response_format=_ClauseTypeGuess,
    )
    return completion.choices[0].message.parsed.clause_type


def classify_clause_type(chunk: ClauseChunk, client: OpenAI | None = None) -> ClauseType:
    """Keyword match against section_title first (free, deterministic --
    covers the common case since real contracts mostly use conventional
    headings). Only falls back to an LLM classification of the clause text
    when the title is missing or doesn't match any known pattern."""
    matched = _match_by_title(chunk.section_title)
    if matched is not None:
        return matched

    if client is None:
        return ClauseType.OTHER

    try:
        return _classify_by_text_llm(chunk, client)
    except Exception:
        logger.exception("Clause-type LLM classification failed for chunk %s", chunk.chunk_id)
        return ClauseType.OTHER


def classify_clause_types_batch(
    chunks: list[ClauseChunk], client: OpenAI, max_workers: int = CLASSIFY_MAX_WORKERS
) -> dict[str, ClauseType]:
    """Same classification as classify_clause_type (keyword match first, LLM
    fallback only when that doesn't match) for many chunks at once: the free
    keyword pass runs for all of them up front, and only the LLM-fallback
    subset -- the actual bottleneck when re-indexing a whole corpus -- is
    classified concurrently instead of one blocking call at a time. Same
    results as calling classify_clause_type per chunk; only the wall-clock
    time differs."""
    results: dict[str, ClauseType] = {}
    needs_llm: list[ClauseChunk] = []

    for chunk in chunks:
        matched = _match_by_title(chunk.section_title)
        if matched is not None:
            results[chunk.chunk_id] = matched
        else:
            needs_llm.append(chunk)

    if not needs_llm:
        return results

    def _classify_one(chunk: ClauseChunk) -> tuple[str, ClauseType]:
        try:
            return chunk.chunk_id, _classify_by_text_llm(chunk, client)
        except Exception:
            logger.exception("Clause-type LLM classification failed for chunk %s", chunk.chunk_id)
            return chunk.chunk_id, ClauseType.OTHER

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for chunk_id, clause_type in pool.map(_classify_one, needs_llm):
            results[chunk_id] = clause_type

    return results
