from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

from docx import Document as DocxDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor
from openai import OpenAI

from contractiq.agents.checklists import get_checklist
from contractiq.agents.models import (
    REVIEW_BANNER,
    CompletenessReport,
    DraftClause,
    DraftResult,
)
from contractiq.config import settings
from contractiq.extraction.db import ContractRecord, get_session
from contractiq.extraction.models import ClauseType
from contractiq.retrieval.models import RetrievedChunk
from contractiq.retrieval.vector_store import dense_search

logger = logging.getLogger(__name__)

DRAFTS_DIR = Path("data/processed/drafts")

NUMERIC_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

SMOOTHING_SYSTEM_PROMPT = (
    "You lightly edit a single contract clause for a reusable draft template. You may ONLY:\n"
    '- Fix grammatical seams left by placeholder substitution (e.g. "The [VENDOR NAME] Inc. '
    'shall" -> "[VENDOR NAME] shall")\n'
    "- Normalize capitalization and spacing\n"
    "- Improve sentence flow\n\n"
    "You must NOT change, add, or remove any number, date, percentage, dollar amount, day "
    "count, or any substantive obligation, right, or condition. Return ONLY the edited clause "
    "text, with no commentary."
)


def _get_vendor_name(doc_id: str) -> str | None:
    session = get_session()
    record = session.query(ContractRecord).filter_by(doc_id=doc_id).one_or_none()
    session.close()
    return record.vendor if record else None


def _substitute_placeholders(text: str, vendor_name: str | None) -> str:
    if not vendor_name:
        return text
    return text.replace(vendor_name, "[VENDOR NAME]")


def _extract_numeric_tokens(text: str) -> set[str]:
    return set(NUMERIC_RE.findall(text))


def _smooth_clause(text: str, client: OpenAI) -> tuple[str, bool]:
    """Returns (final_text, was_smoothed). Falls back to the unmodified
    input if the LLM's edit changes the set of numeric tokens present --
    this check, not the prompt instruction, is the actual safety net."""
    try:
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": SMOOTHING_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        smoothed = completion.choices[0].message.content or ""
    except Exception:
        logger.exception("Clause smoothing failed; using unmodified precedent text.")
        return text, False

    if _extract_numeric_tokens(smoothed) != _extract_numeric_tokens(text):
        logger.warning("Smoothing altered numeric content; reverting to raw precedent text.")
        return text, False

    return smoothed, True


def _find_precedent(
    clause_type: ClauseType, business_brief: str, agreement_type: str, client: OpenAI
) -> RetrievedChunk | None:
    query = f"{clause_type.value.replace('_', ' ')} clause for a {agreement_type} agreement. {business_brief}"
    results = dense_search(query, client, top_k=1, where={"clause_type": clause_type.value})
    return results[0] if results else None


def _build_docx(
    agreement_type: str,
    business_brief: str,
    clauses: list[DraftClause],
    completeness: CompletenessReport,
    docx_path: Path,
) -> None:
    doc = DocxDocument()

    banner_para = doc.add_paragraph()
    banner_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    banner_run = banner_para.add_run(f"⚠ {REVIEW_BANNER} ⚠")
    banner_run.bold = True
    banner_run.font.size = Pt(14)
    banner_run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    doc.add_heading(f"{agreement_type} — Draft", level=1)
    doc.add_paragraph(f"Generated: {datetime.datetime.utcnow().isoformat()}Z")
    doc.add_paragraph("Business context (as provided, verbatim): " + business_brief)

    doc.add_heading("Completeness Report", level=2)
    if not completeness.agreement_type_recognized:
        note = doc.add_paragraph()
        note.add_run(
            f'Agreement type "{agreement_type}" was not recognized -- a generic checklist '
            "was used instead."
        ).italic = True
    for clause_type in completeness.required:
        status = "present" if clause_type in completeness.present else "MISSING"
        doc.add_paragraph(f"  - {clause_type.value}: {status}")

    doc.add_heading("Clauses", level=2)
    for clause in clauses:
        doc.add_heading(clause.clause_type.value.replace("_", " ").title(), level=3)
        if clause.status == "missing":
            p = doc.add_paragraph()
            p.add_run("⚠ MISSING — no precedent found for this clause type").italic = True
            continue

        doc.add_paragraph(clause.text)
        source_p = doc.add_paragraph()
        source_text = f"Source: {clause.source_document}, {clause.source_section}, p.{clause.source_page}"
        if not clause.smoothed:
            source_text += " (raw precedent text -- automated smoothing was rejected)"
        source_run = source_p.add_run(source_text)
        source_run.italic = True
        source_run.font.size = Pt(9)

    footer = doc.sections[0].footer
    footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    footer_run = footer_para.add_run(REVIEW_BANNER)
    footer_run.bold = True
    footer_run.font.size = Pt(8)

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))


def drafting_agent(agreement_type: str, business_brief: str) -> DraftResult:
    """Assembles a draft from retrieved precedent clauses -- never
    generates clause language freely. The OpenAI API is used only to
    lightly smooth each selected precedent's phrasing (validated, see
    _smooth_clause), never to select clauses or invent missing ones."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before running the drafting agent."
        )
    client = OpenAI(api_key=settings.openai_api_key)

    checklist, recognized = get_checklist(agreement_type)

    clauses: list[DraftClause] = []
    for clause_type in checklist:
        precedent = _find_precedent(clause_type, business_brief, agreement_type, client)
        if precedent is None:
            clauses.append(DraftClause(clause_type=clause_type, status="missing"))
            continue

        vendor_name = _get_vendor_name(precedent.doc_id)
        prepared_text = _substitute_placeholders(precedent.text, vendor_name)
        final_text, smoothed = _smooth_clause(prepared_text, client)

        clauses.append(
            DraftClause(
                clause_type=clause_type,
                status="drafted",
                text=final_text,
                source_document=precedent.source_file,
                source_section=precedent.section_title
                or (f"Clause {precedent.clause_number}" if precedent.clause_number else None),
                source_page=precedent.page,
                smoothed=smoothed,
            )
        )

    present = [c.clause_type for c in clauses if c.status == "drafted"]
    missing = [c.clause_type for c in clauses if c.status == "missing"]
    completeness = CompletenessReport(
        agreement_type=agreement_type,
        agreement_type_recognized=recognized,
        required=checklist,
        present=present,
        missing=missing,
    )

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_type = re.sub(r"[^A-Za-z0-9_-]", "_", agreement_type)
    docx_path = DRAFTS_DIR / f"{safe_type}_{timestamp}.docx"
    _build_docx(agreement_type, business_brief, clauses, completeness, docx_path)

    return DraftResult(
        agreement_type=agreement_type,
        business_brief=business_brief,
        clauses=clauses,
        completeness=completeness,
        docx_path=str(docx_path),
    )
