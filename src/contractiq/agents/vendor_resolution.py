from __future__ import annotations

import re

from openai import OpenAI
from pydantic import BaseModel

from contractiq.config import settings
from contractiq.extraction.db import ContractRecord, get_session

# A question/brief naming a specific vendor can lose to near-identical
# boilerplate clauses in other vendors' contracts once a cross-encoder rerank
# (or a single-best-match dense search, in drafting) can no longer tell the
# passages apart by text alone (this corpus has ~17 contracts built from the
# same template). Resolving the named vendor to its doc_ids and passing them
# as a hard pre-filter sidesteps that tie entirely. Shared between the RAG
# agent and the drafting agent, which both hit this exact problem.
MAX_VENDOR_MATCHES = 5

VENDOR_HINT_PROMPT = (
    "Extract the specific vendor/company name explicitly mentioned in this "
    "text about a contract corpus, if any. Return only the distinguishing "
    "identifier (e.g. \"SMK\", \"Fayakoon\", \"A&B\"), not generic contract-type "
    "language like \"Civil Works Agreement\" or \"the vendor\". If no specific "
    "company is named, return null."
)


class _VendorHint(BaseModel):
    vendor_name: str | None


def extract_vendor_hint(text: str, client: OpenAI) -> str | None:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        temperature=0,  # deterministic extraction, not creative -- identical
        # input should resolve the same vendor every time, not flip on
        # sampling noise.
        messages=[
            {"role": "system", "content": VENDOR_HINT_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format=_VendorHint,
    )
    return completion.choices[0].message.parsed.vendor_name


# Corpus vendor names are spelled inconsistently across records and filenames
# for the same real vendor -- e.g. "Asif & Co" (question), "ASIF AND CO"
# (metadata field), "Asif n Co" (source filename, a typo'd "&"). A plain
# lowercase substring check treats these as unrelated strings. Normalizing
# "&"/standalone "n" to "and" and stripping punctuation before comparing
# collapses all three onto "asif and co", which is what the naive check was
# trying to do in the first place.
def normalize_vendor_text(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bn\b", "and", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_vendor_doc_ids(hint: str) -> set[str] | None:
    session = get_session()
    records = session.query(ContractRecord).all()
    session.close()

    needle = normalize_vendor_text(hint)
    matches = {
        r.doc_id
        for r in records
        if needle in normalize_vendor_text(r.vendor or "")
        or needle in normalize_vendor_text(r.source_file)
    }
    if not matches or len(matches) > MAX_VENDOR_MATCHES:
        return None
    return matches
