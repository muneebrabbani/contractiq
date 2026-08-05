from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    clause_number: str | None
    section_title: str | None
    page: int
    text: str
    score: float


class Citation(BaseModel):
    document: str
    section: str | None
    page: int
    clause_number: str | None


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[str]
