from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from openai import OpenAI

from contractiq.config import settings
from contractiq.extraction.clause_chunking import chunk_document
from contractiq.extraction.db import ContractRecord, get_session, save_record
from contractiq.extraction.metadata import extract_metadata
from contractiq.extraction.redaction import append_redaction_log, redact_document
from contractiq.ingestion.loaders import SUPPORTED_SUFFIXES
from contractiq.retrieval.indexing import index_chunks_for

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def process_uploaded_document(
    file_bytes: bytes,
    filename: str,
    related_doc_id: str | None = None,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> ContractRecord:
    """End-to-end single-document pipeline for a newly uploaded contract (or
    a renewal/addendum of an existing one): save to data/raw, redact,
    clause-chunk, extract structured metadata, and index into the retrieval
    store -- the same stages the batch corpus pipeline runs, scoped to one
    file so a single new upload doesn't require re-running the whole corpus.
    Recon is intentionally skipped here: it's a standalone audit report
    (recon_report.csv) that nothing downstream reads."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before uploading a document."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r} -- only PDF and DOCX are supported.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != hashlib.sha256(file_bytes).hexdigest():
            raise ValueError(
                f"{filename!r} already exists in {raw_dir} with different content -- "
                "rename the file (e.g. add a version suffix) before uploading."
            )
        logger.info("Re-uploading identical content for %s; reprocessing idempotently.", filename)
    else:
        dest.write_bytes(file_bytes)

    redacted_document, redaction_records = redact_document(dest, processed_dir)
    append_redaction_log(redaction_records, processed_dir / "redaction_log.csv")

    chunks = chunk_document(redacted_document)

    client = OpenAI(api_key=settings.openai_api_key)
    metadata = extract_metadata(redacted_document, client)

    session = get_session()
    record = save_record(
        session,
        doc_id=redacted_document.doc_id,
        source_file=dest.name,
        metadata=metadata,
        related_doc_id=related_doc_id,
    )
    session.close()

    index_chunks_for(chunks, client)

    return record
