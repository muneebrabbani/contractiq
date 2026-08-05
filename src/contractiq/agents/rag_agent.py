from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from contractiq.config import settings
from contractiq.extraction.db import ContractRecord, get_session
from contractiq.retrieval.models import AnswerResult
from contractiq.retrieval.rag import answer as retrieval_answer

# A question naming a specific vendor can lose to near-identical boilerplate
# clauses in other vendors' contracts once cross-encoder rerank can no longer
# tell the passages apart by text alone (this corpus has ~17 contracts built
# from the same template). Resolving the named vendor to its doc_ids and
# passing them as a hard pre-filter sidesteps that tie entirely.
MAX_VENDOR_MATCHES = 5

VENDOR_HINT_PROMPT = (
    "Extract the specific vendor/company name explicitly mentioned in this "
    "question about a contract corpus, if any. Return only the distinguishing "
    "identifier (e.g. \"SMK\", \"Fayakoon\", \"A&B\"), not generic contract-type "
    "language like \"Civil Works Agreement\" or \"the vendor\". If no specific "
    "company is named, return null."
)


class _VendorHint(BaseModel):
    vendor_name: str | None


def _extract_vendor_hint(question: str, client: OpenAI) -> str | None:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": VENDOR_HINT_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format=_VendorHint,
    )
    return completion.choices[0].message.parsed.vendor_name


def _resolve_vendor_doc_ids(hint: str) -> set[str] | None:
    session = get_session()
    records = session.query(ContractRecord).all()
    session.close()

    needle = hint.lower()
    matches = {
        r.doc_id
        for r in records
        if needle in (r.vendor or "").lower() or needle in r.source_file.lower()
    }
    if not matches or len(matches) > MAX_VENDOR_MATCHES:
        return None
    return matches


def rag_agent(question: str) -> AnswerResult:
    """Wraps retrieval.answer, additionally narrowing to a named vendor's
    documents when the question identifies one specifically -- retrieval
    internals (dense/BM25/fusion/rerank) untouched."""
    client = OpenAI(api_key=settings.openai_api_key)
    hint = _extract_vendor_hint(question, client)
    doc_ids = _resolve_vendor_doc_ids(hint) if hint else None
    return retrieval_answer(question, doc_ids=doc_ids)
