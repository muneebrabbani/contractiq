from __future__ import annotations

import logging
import re

from openai import OpenAI
from pydantic import BaseModel

from contractiq.config import settings
from contractiq.extraction.models import ClauseChunk, ClauseType

logger = logging.getLogger(__name__)

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
