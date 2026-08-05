from __future__ import annotations

from openai import OpenAI

from contractiq.agents.vendor_resolution import (
    extract_vendor_hint,
    resolve_vendor_doc_ids,
    strip_vendor_mention,
)
from contractiq.config import settings
from contractiq.retrieval.models import AnswerResult
from contractiq.retrieval.rag import answer as retrieval_answer


def rag_agent(question: str, fallback_doc_ids: set[str] | None = None) -> tuple[AnswerResult, set[str] | None]:
    """Wraps retrieval.answer, additionally narrowing to a named vendor's
    documents when the question identifies one specifically -- retrieval
    internals (dense/BM25/fusion/rerank) untouched.

    `fallback_doc_ids` is the document scope the session was already
    narrowed to (from an earlier turn in the same conversation), used only
    when THIS question's own vendor-hint extraction comes up empty --
    contextualization (agents/contextualize.py) already tries to re-embed
    the vendor name into every follow-up, so this is a second line of
    defense for cases where that still doesn't yield an extractable hint,
    not the primary mechanism.

    Returns the doc_ids actually used (or None for an unfiltered search) so
    the caller can carry it forward as next turn's fallback.
    """
    client = OpenAI(api_key=settings.openai_api_key)
    hint = extract_vendor_hint(question, client)
    doc_ids = resolve_vendor_doc_ids(hint) if hint else None

    # Only strip the vendor name when THIS turn's own hint is what produced
    # doc_ids -- the search text still needs to name the vendor when doc_ids
    # instead came from the sticky fallback (a prior turn's scope), since
    # nothing else in this question would otherwise identify what's being
    # asked about within that scope.
    retrieval_query = strip_vendor_mention(question, hint) if doc_ids and hint else None

    if doc_ids is None:
        doc_ids = fallback_doc_ids
    result = retrieval_answer(question, doc_ids=doc_ids, retrieval_query=retrieval_query)
    return result, doc_ids
