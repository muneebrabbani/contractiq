from __future__ import annotations

from contractiq.extraction.models import ClauseType

# Hardcoded, reviewable per agreement type -- this is a legal/business
# judgment call, not something to let an LLM infer.
CHECKLISTS: dict[str, list[ClauseType]] = {
    "MSA": [
        ClauseType.SCOPE_OF_WORK,
        ClauseType.TERMINATION,
        ClauseType.GOVERNING_LAW,
        ClauseType.PAYMENT_TERMS,
        ClauseType.CONFIDENTIALITY,
        ClauseType.INDEMNIFICATION,
        ClauseType.LIMITATION_OF_LIABILITY,
        ClauseType.NOTICES,
        ClauseType.WARRANTIES,
        ClauseType.ASSIGNMENT,
        ClauseType.FORCE_MAJEURE,
        ClauseType.DISPUTE_RESOLUTION,
        ClauseType.ENTIRE_AGREEMENT,
    ],
    "NDA": [
        ClauseType.CONFIDENTIALITY,
        ClauseType.TERMINATION,
        ClauseType.GOVERNING_LAW,
        ClauseType.NOTICES,
        ClauseType.ENTIRE_AGREEMENT,
    ],
    "SOW": [
        ClauseType.SCOPE_OF_WORK,
        ClauseType.PAYMENT_TERMS,
        ClauseType.TERMINATION,
        ClauseType.GOVERNING_LAW,
        ClauseType.WARRANTIES,
        ClauseType.INTELLECTUAL_PROPERTY,
    ],
}

DEFAULT_CHECKLIST: list[ClauseType] = [
    ClauseType.TERMINATION,
    ClauseType.GOVERNING_LAW,
    ClauseType.PAYMENT_TERMS,
    ClauseType.CONFIDENTIALITY,
    ClauseType.NOTICES,
    ClauseType.ENTIRE_AGREEMENT,
]


def get_checklist(agreement_type: str) -> tuple[list[ClauseType], bool]:
    """Returns (checklist, recognized). recognized=False means agreement_type
    wasn't a known key and the generic default was used instead -- callers
    should surface that rather than silently drafting against the fallback."""
    checklist = CHECKLISTS.get(agreement_type.upper())
    if checklist is not None:
        return checklist, True
    return DEFAULT_CHECKLIST, False
