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
    trace: RouteTrace
