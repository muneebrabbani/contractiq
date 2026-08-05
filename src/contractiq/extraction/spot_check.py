from __future__ import annotations

from pathlib import Path

from contractiq.extraction.db import ContractRecord, get_session
from contractiq.extraction.redaction import load_redacted_document

PROCESSED_DIR = Path("data/processed")
SNIPPET_CHARS = 400


def _document_snippets(doc_id: str, input_dir: Path) -> tuple[str, str]:
    path = input_dir / f"{doc_id}.redacted.json"
    if not path.exists():
        return "(source document not found)", ""

    document = load_redacted_document(path)
    full_text = "\n\n".join(page.text for page in document.pages)
    head = full_text[:SNIPPET_CHARS].strip()
    tail = full_text[-SNIPPET_CHARS:].strip()
    return head, tail


def print_spot_check(input_dir: Path = PROCESSED_DIR) -> None:
    session = get_session()
    records = session.query(ContractRecord).all()
    session.close()

    if not records:
        print("No extracted contract records found. Run extraction.metadata.extract_all() first.")
        return

    for record in records:
        head, tail = _document_snippets(record.doc_id, input_dir)

        print("=" * 80)
        print(f"{record.source_file}  (doc_id={record.doc_id})")
        print("-" * 80)
        print(f"  contract_number:  {record.contract_number}")
        print(f"  agreement_title:  {record.agreement_title}")
        print(f"  contract_type:    {record.contract_type}")
        print(f"  agreement_category: {record.agreement_category}")
        print(f"  vendor:           {record.vendor}")
        print(f"  business_unit:    {record.business_unit}")
        print(f"  department:       {record.department}")
        print(f"  effective_date:   {record.effective_date}")
        print(f"  expiry_date:      {record.expiry_date}")
        print(f"  value:            {record.value}")
        print(f"  currency:         {record.currency}")
        print(f"  signatory_names:  {record.signatory_names}")
        print(f"  payment_terms:    {record.payment_terms}")
        print(f"  notice_period:    {record.notice_period}")
        print()
        print(f"  --- document head (first {SNIPPET_CHARS} chars) ---")
        print(f"  {head}")
        print()
        print(f"  --- document tail (last {SNIPPET_CHARS} chars) ---")
        print(f"  {tail}")
        print()


if __name__ == "__main__":
    print_spot_check()
