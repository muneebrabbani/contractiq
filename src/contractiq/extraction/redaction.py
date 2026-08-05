from __future__ import annotations

import csv
import ipaddress
import re
from pathlib import Path

import spacy

from contractiq.extraction.models import RedactedDocument, RedactedPage, RedactionRecord
from contractiq.ingestion.loaders import SUPPORTED_SUFFIXES, load_document

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

_NLP = spacy.load("en_core_web_sm")

# --- Phone / email / IP: regex, validated where a stdlib validator exists ---

PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
IPV4_CANDIDATE_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,2})?\b")
IPV6_CANDIDATE_RE = re.compile(
    r"(?<![\w:])[0-9a-fA-F]*(?::[0-9a-fA-F]*){2,}(?:/\d{1,3})?(?![\w:])"
)

# --- Address: regex anchor (structural evidence) + NER extension (city/state) ---

_STREET_TYPES = (
    r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|"
    r"Lane|Ln\.?|Way|Court|Ct\.?|Plaza|Highway|Hwy\.?|Place|Pl\.?)"
)
_SECONDARY_UNIT = (
    r"(?:,?\s*(?:Suite|Ste\.?|Floor|Fl\.?|Unit|Apt\.?|Room|Rm\.?|#)\s*[A-Za-z0-9-]+)?"
)
STREET_ANCHOR_RE = re.compile(
    rf"\b\d{{1,6}}[A-Za-z]?\s+(?:[A-Z][a-zA-Z'.-]*\s+){{0,4}}{_STREET_TYPES}\.?{_SECONDARY_UNIT}"
)
STATE_ZIP_RE = re.compile(r"\b(?:[A-Z]{2}\s+)?\d{5}(?:-\d{4})?\b")
ADDRESS_WINDOW_CHARS = 200


def _validate_ip(candidate: str) -> bool:
    try:
        ipaddress.ip_network(candidate, strict=False)
        return True
    except ValueError:
        return False


def _extend_address_span(text: str, anchor_end: int) -> int:
    """Extend a street-address anchor through trailing city/state/zip only.

    NER (GPE/LOC) may only extend the match, never trigger it, and stops
    immediately at any non-place entity (PERSON, ORG) or non-place text so a
    party name on the same or next line is never absorbed.
    """
    window_end = min(len(text), anchor_end + ADDRESS_WINDOW_CHARS)
    window = text[anchor_end:window_end]
    doc = _NLP(window)

    extend_to = 0
    for ent in doc.ents:
        gap = window[extend_to : ent.start_char]
        if gap.strip(" ,\n\r\t") != "":
            break
        if ent.label_ in ("GPE", "LOC"):
            extend_to = ent.end_char
        else:
            break

    probe = extend_to
    while probe < len(window) and window[probe] in " ,\t":
        probe += 1
    zip_match = STATE_ZIP_RE.match(window, probe)
    if zip_match:
        extend_to = zip_match.end()

    return anchor_end + extend_to


def find_address_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in STREET_ANCHOR_RE.finditer(text):
        end = _extend_address_span(text, match.end())
        spans.append((match.start(), end))
    return spans


def apply_redactions(
    text: str, file_name: str, page_number: int
) -> tuple[str, list[RedactionRecord]]:
    candidates: list[tuple[int, int, str, str]] = []

    for m in PHONE_RE.finditer(text):
        candidates.append((m.start(), m.end(), "PHONE", m.group()))
    for m in EMAIL_RE.finditer(text):
        candidates.append((m.start(), m.end(), "EMAIL", m.group()))
    for m in list(IPV4_CANDIDATE_RE.finditer(text)) + list(IPV6_CANDIDATE_RE.finditer(text)):
        if _validate_ip(m.group()):
            candidates.append((m.start(), m.end(), "IP", m.group()))
    for start, end in find_address_spans(text):
        candidates.append((start, end, "ADDRESS", text[start:end]))

    candidates.sort(key=lambda c: c[0])
    accepted: list[tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, category, snippet in candidates:
        if start < last_end:
            continue  # overlaps a previously accepted match
        accepted.append((start, end, category, snippet))
        last_end = end

    new_text = text
    for start, end, category, snippet in sorted(accepted, key=lambda c: c[0], reverse=True):
        placeholder = f"[{category}]"
        new_text = new_text[:start] + placeholder + new_text[end:]

    records = [
        RedactionRecord(
            file=file_name,
            category=category,
            original_snippet=snippet,
            replacement=f"[{category}]",
            page_number=page_number,
        )
        for _, _, category, snippet in accepted
    ]
    return new_text, records


def redact_document(
    path: Path, output_dir: Path = PROCESSED_DIR
) -> tuple[RedactedDocument, list[RedactionRecord]]:
    """Redact a single document and write its .redacted.json. Shared by the
    corpus-wide batch below and the single-upload pipeline (extraction/pipeline.py)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = load_document(path)

    cleaned_pages = []
    records: list[RedactionRecord] = []
    for page in document.pages:
        cleaned_text, page_records = apply_redactions(page.text, path.name, page.page_number)
        cleaned_pages.append(RedactedPage(page_number=page.page_number, text=cleaned_text))
        records.extend(page_records)

    redacted_document = RedactedDocument(
        doc_id=document.doc_id, file_path=str(path), pages=cleaned_pages
    )
    out_path = output_dir / f"{document.doc_id}.redacted.json"
    out_path.write_text(redacted_document.model_dump_json(indent=2), encoding="utf-8")

    return redacted_document, records


def redact_directory(
    input_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR
) -> list[RedactionRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[RedactionRecord] = []

    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        _, records = redact_document(path, output_dir)
        all_records.extend(records)

    _write_redaction_log(all_records, output_dir / "redaction_log.csv")
    return all_records


def load_redacted_document(path: Path) -> RedactedDocument:
    return RedactedDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _write_redaction_log(records: list[RedactionRecord], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RedactionRecord.model_fields))
        writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())


def append_redaction_log(records: list[RedactionRecord], out_path: Path) -> None:
    """Append-mode variant for a single upload -- must not clobber the
    corpus-wide log that redact_directory() (over)writes."""
    file_exists = out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RedactionRecord.model_fields))
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(r.model_dump())


if __name__ == "__main__":
    redact_directory()
