from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from contractiq.config import settings
from contractiq.extraction.db import get_session, save_record
from contractiq.extraction.models import ContractMetadata, RedactedDocument
from contractiq.extraction.redaction import load_redacted_document

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

SYSTEM_PROMPT = (
    "You are a contract data extraction assistant for a telecom procurement team. "
    "Extract the requested fields from the contract text exactly as stated. Do not "
    "infer or guess values that are not present in the text -- leave a field null "
    "if it is not stated. Dates should be ISO 8601 (YYYY-MM-DD) when a specific date "
    "is given, otherwise use the contract's own wording. Monetary values should be "
    "plain numbers with no currency symbol or thousands separators."
)


def _document_text(document: RedactedDocument) -> str:
    return "\n\n".join(page.text for page in document.pages)


def extract_metadata(document: RedactedDocument, client: OpenAI) -> ContractMetadata:
    completion = client.chat.completions.parse(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _document_text(document)},
        ],
        response_format=ContractMetadata,
    )
    return completion.choices[0].message.parsed


def extract_all(input_dir: Path = PROCESSED_DIR) -> list[ContractMetadata]:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in "
            "before running metadata extraction."
        )
    client = OpenAI(api_key=settings.openai_api_key)
    session = get_session()
    results: list[ContractMetadata] = []

    for path in sorted(input_dir.glob("*.redacted.json")):
        document = load_redacted_document(path)
        try:
            metadata = extract_metadata(document, client=client)
        except Exception:
            logger.exception("Failed to extract metadata for %s", document.doc_id)
            continue

        save_record(
            session,
            doc_id=document.doc_id,
            source_file=Path(document.file_path).name,
            metadata=metadata,
        )
        results.append(metadata)

    session.close()
    return results


if __name__ == "__main__":
    extract_all()
