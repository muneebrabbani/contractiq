from __future__ import annotations

from pydantic import BaseModel


class EvalExample(BaseModel):
    id: str
    question: str
    gold_answer: str
    source_document: str  # raw filename, e.g. "MSA_Acme.docx"
    source_section: str  # human-readable, e.g. "Section 4 - Notices" or "Clause 10.2"
    category: str | None = None
    notes: str | None = None


class EvalResult(BaseModel):
    id: str
    question: str
    gold_answer: str
    generated_answer: str
    contexts: list[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float

    @property
    def min_score(self) -> float:
        return min(self.faithfulness, self.answer_relevancy, self.context_precision)


class EvalReport(BaseModel):
    results: list[EvalResult]
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
