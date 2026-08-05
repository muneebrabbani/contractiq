from __future__ import annotations

import streamlit as st

from contractiq.extraction.db import ContractRecord, get_session
from contractiq.extraction.pipeline import process_uploaded_document

_NEW_CONTRACT_LABEL = "— New / standalone contract —"


def _existing_contract_options() -> dict[str, str | None]:
    session = get_session()
    records = session.query(ContractRecord).order_by(ContractRecord.vendor, ContractRecord.agreement_title).all()
    session.close()

    options: dict[str, str | None] = {_NEW_CONTRACT_LABEL: None}
    for r in records:
        label = f"{r.agreement_title or r.source_file} ({r.vendor or 'Unknown vendor'})"
        options[label] = r.doc_id
    return options


def render() -> None:
    st.title("Upload a Contract")
    st.caption(
        "Add a new contract, or a renewal/addendum/variation order for an existing one -- "
        "runs redaction, metadata extraction, and indexing automatically."
    )

    uploaded = st.file_uploader("Contract file", type=["pdf", "docx"])

    options = _existing_contract_options()
    choice = st.selectbox(
        "Is this a renewal, addendum, or variation order for an existing contract?",
        list(options.keys()),
    )
    related_doc_id = options[choice]

    if st.button("Process document", disabled=uploaded is None):
        try:
            with st.spinner("Redacting, extracting metadata, and indexing..."):
                record = process_uploaded_document(
                    uploaded.getvalue(), uploaded.name, related_doc_id=related_doc_id
                )
            st.session_state.last_upload = record
            st.session_state.last_upload_parent_label = (
                choice if related_doc_id else None
            )
        except Exception as exc:
            st.session_state.last_upload = None
            st.error(f"Could not process {uploaded.name!r}: {exc}")

    record = st.session_state.get("last_upload")
    if record is None:
        return

    st.divider()
    st.success(f"Processed: {record.agreement_title or record.source_file}")
    parent_label = st.session_state.get("last_upload_parent_label")
    if parent_label:
        st.info(f"Linked as a renewal/addendum of: {parent_label} -- that contract is now superseded.")

    st.table(
        [
            {
                "Field": field,
                "Value": getattr(record, field) or "-",
            }
            for field in (
                "agreement_title",
                "contract_type",
                "vendor",
                "segment",
                "status",
                "effective_date",
                "expiry_date",
                "value",
                "currency",
            )
        ]
    )
