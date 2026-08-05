from __future__ import annotations

from pathlib import Path

import streamlit as st

from contractiq.agents import drafting_agent
from contractiq.agents.checklists import CHECKLISTS
from contractiq.agents.models import REVIEW_BANNER


def render() -> None:
    st.title("Draft a New Agreement")
    st.warning(REVIEW_BANNER)

    agreement_type = st.selectbox("Agreement type", options=sorted(CHECKLISTS.keys()) + ["Other"])
    if agreement_type == "Other":
        agreement_type = st.text_input("Enter agreement type")

    brief = st.text_area(
        "Business brief",
        placeholder="Describe the deal: vendor, subject matter, key context...",
        height=150,
    )

    if st.button("Generate Draft", disabled=not (agreement_type and brief)):
        with st.spinner("Retrieving precedent clauses and assembling draft..."):
            draft = drafting_agent(agreement_type, brief)
        st.session_state.last_draft = draft

    draft = st.session_state.get("last_draft")
    if draft is None:
        return

    st.divider()
    st.subheader("Completeness Report")
    if not draft.completeness.agreement_type_recognized:
        st.info(
            f'Agreement type "{draft.agreement_type}" was not recognized -- a generic '
            "checklist was used."
        )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Present**")
        for ct in draft.completeness.present:
            st.markdown(f"- {ct.value}")
    with cols[1]:
        st.markdown("**Missing**")
        if not draft.completeness.missing:
            st.markdown("_none_")
        for ct in draft.completeness.missing:
            st.markdown(f"- ⚠ {ct.value}")

    docx_path = Path(draft.docx_path)
    if docx_path.exists():
        st.download_button(
            "Download draft (DOCX)",
            docx_path.read_bytes(),
            file_name=docx_path.name,
            key=f"download_draft_{draft.docx_path}",
        )

    with st.expander("Preview clauses"):
        for clause in draft.clauses:
            st.markdown(f"**{clause.clause_type.value.replace('_', ' ').title()}**")
            if clause.status == "missing":
                st.markdown("_MISSING -- no precedent found_")
            else:
                st.write(clause.text)
                st.caption(
                    f"Source: {clause.source_document}, {clause.source_section}, "
                    f"p.{clause.source_page}"
                )
            st.markdown("---")
