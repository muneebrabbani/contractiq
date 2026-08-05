from __future__ import annotations

import logging
from pathlib import Path

from contractiq.ingestion.chunking import chunk_document
from contractiq.ingestion.loaders import SUPPORTED_SUFFIXES, load_document
from contractiq.ingestion.models import Chunk

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def ingest_file(path: Path, output_dir: Path = PROCESSED_DIR) -> list[Chunk]:
    document = load_document(path)
    chunks = chunk_document(document)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{document.doc_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    return chunks


def ingest_directory(
    input_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR
) -> dict[str, int]:
    results: dict[str, int] = {}
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            chunks = ingest_file(path, output_dir=output_dir)
        except Exception:
            logger.exception("Failed to ingest %s", path)
            continue
        results[path.name] = len(chunks)
    return results
