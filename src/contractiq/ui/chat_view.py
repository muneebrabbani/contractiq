from __future__ import annotations

from pathlib import Path

import streamlit as st

from contractiq.agents import run_supervisor


def _render_extras(message: dict) -> None:
    if message.get("citations"):
        with st.expander(f"Sources ({len(message['citations'])})"):
            for c in message["citations"]:
                st.markdown(f"- **{c['document']}**, {c['section'] or 'General'}, p.{c['page']}")

    if message.get("sql"):
        with st.expander("View query"):
            st.code(message["sql"], language="sql")

    if message.get("docx_path"):
        docx_path = Path(message["docx_path"])
        if docx_path.exists():
            st.download_button(
                "Download draft (DOCX)",
                docx_path.read_bytes(),
                file_name=docx_path.name,
                key=f"download_{message['docx_path']}",
            )


def render() -> None:
    st.title("ContractIQ Chat")
    st.caption("Ask about your contracts, request analytics, or ask for a draft.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("agent"):
                st.caption(f"Routed to: {message['agent']} agent")
            _render_extras(message)

    question = st.chat_input("Ask a question...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        response = run_supervisor(question)

    assistant_message: dict = {
        "role": "assistant",
        "content": response.answer,
        "agent": response.agent,
    }
    if response.citations:
        assistant_message["citations"] = [
            {"document": c.document, "section": c.section, "page": c.page}
            for c in response.citations
        ]
    if response.sql:
        assistant_message["sql"] = response.sql
    if response.draft:
        assistant_message["docx_path"] = response.draft.docx_path

    st.session_state.messages.append(assistant_message)
    st.rerun()
