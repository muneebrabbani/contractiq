from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from contractiq.config import settings
from contractiq.extraction.clause_chunking import load_all_clause_chunks
from contractiq.extraction.clause_classifier import classify_clause_type
from contractiq.extraction.db import ContractRecord, get_session
from contractiq.extraction.models import ClauseChunk
from contractiq.retrieval.vector_store import index_chunks

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def _metadata_for_chunk(chunk: ClauseChunk, record: ContractRecord | None, client: OpenAI) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "source_file": chunk.source_file,
        "clause_number": chunk.clause_number,
        "section_title": chunk.section_title,
        "page": chunk.page,
        "segment": record.segment if record else None,
        "status": record.status if record else None,
        "clause_type": classify_clause_type(chunk, client).value,
    }


def index_chunks_for(chunks: list[ClauseChunk], client: OpenAI) -> int:
    """Index an arbitrary subset of chunks (e.g. a single newly-uploaded
    document) into the same persistent Chroma collection build_index() uses
    -- looks up only the ContractRecords needed for this subset's doc_ids and
    joins segment/status the same way build_index() does for the whole corpus."""
    if not chunks:
        return 0

    session = get_session()
    doc_ids = {c.doc_id for c in chunks}
    records = {
        r.doc_id: r
        for r in session.query(ContractRecord).filter(ContractRecord.doc_id.in_(doc_ids)).all()
    }
    session.close()

    chunks_with_metadata = [(c, _metadata_for_chunk(c, records.get(c.doc_id), client)) for c in chunks]
    return index_chunks(chunks_with_metadata, client)


def build_index(input_dir: Path = PROCESSED_DIR) -> int:
    """Embed every clause chunk and upsert into the persistent Chroma
    collection. Joins each chunk to its parent document's segment/status
    (from the metadata-extraction stage's SQLite table, by doc_id) for
    filterable metadata -- a document that hasn't been through metadata
    extraction yet indexes fine, just without segment/status filters.
    Also classifies each chunk's clause_type (keyword match against
    section_title, LLM fallback only when that doesn't match) so the
    drafting agent's precedent retrieval can filter by it."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before building the retrieval index."
        )

    chunks = load_all_clause_chunks(input_dir)
    if not chunks:
        logger.warning("No chunks found under %s; nothing to index.", input_dir)
        return 0

    client = OpenAI(api_key=settings.openai_api_key)
    return index_chunks_for(chunks, client)


if __name__ == "__main__":
    print(f"Indexed {build_index()} chunks.")
