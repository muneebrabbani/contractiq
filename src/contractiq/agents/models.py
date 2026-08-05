from __future__ import annotations

import datetime
from enum import Enum

from pydantic import BaseModel, Field

from contractiq.extraction.models import ClauseType
from contractiq.retrieval.models import Citation

REVIEW_BANNER = "AI-GENERATED DRAFT — REQUIRES HUMAN LEGAL REVIEW"


class Intent(str, Enum):
    NARRATIVE = "narrative"
    ANALYTICS = "analytics"
    DRAFTING = "drafting"
    OUT_OF_SCOPE = "out_of_scope"


class RouteDecision(BaseModel):
    intent: Intent
    reasoning: str = Field(description="Brief justification for the routing decision")


class RouteTrace(BaseModel):
    question: str
    contextualized_question: str = ""  # == question when there's no history to rewrite from
    intent: Intent
    reasoning: str
    agent: str
    timestamp: datetime.datetime


class DraftClause(BaseModel):
    clause_type: ClauseType
    status: str  # "drafted" | "missing"
    text: str | None = None
    source_document: str | None = None
    source_section: str | None = None
    source_page: int | None = None
    smoothed: bool = False  # False if LLM smoothing was rejected and raw precedent was used
    # None: no vendor was named in the business brief, so vendor-scoping never
    # applied. True: this precedent came from the named vendor's own
    # document. False: a vendor was named, but it has no precedent of this
    # clause type -- this clause fell back to the best corpus-wide match.
    from_named_vendor: bool | None = None


class CompletenessReport(BaseModel):
    agreement_type: str
    agreement_type_recognized: bool
    required: list[ClauseType]
    present: list[ClauseType]
    missing: list[ClauseType]


class DraftResult(BaseModel):
    agreement_type: str
    business_brief: str
    clauses: list[DraftClause]
    completeness: CompletenessReport
    docx_path: str
    banner: str = REVIEW_BANNER


class AgentResponse(BaseModel):
    answer: str
    agent: str  # "rag" | "sql" | "drafting" | "decline"
    citations: list[Citation] = Field(default_factory=list)  # populated by the RAG agent
    contexts: list[str] = Field(default_factory=list)  # retrieved passage text, RAG agent only
    sql: str | None = None  # populated by the SQL agent, for auditability
    draft: DraftResult | None = None  # populated by the drafting agent
    # RAG's resolved document scope for this turn, sticky across the session --
    # pass back in as run_supervisor(..., active_doc_ids=...) on the next call
    # so a follow-up that doesn't re-name the vendor still stays scoped to it.
    active_doc_ids: set[str] | None = None
    trace: RouteTrace
