from __future__ import annotations

import streamlit as st

from contractiq.ui import alerts_view, chat_view, draft_view, upload_view

st.set_page_config(page_title="ContractIQ", page_icon="📄", layout="wide")

# pages = [
#     st.Page(chat_view.render, title="Chat", icon="💬", default=True),
#     st.Page(alerts_view.render, title="Alerts", icon="⏰"),
#     st.Page(draft_view.render, title="Draft", icon="📝"),
# ]

pages = [
    st.Page(chat_view.render, title="Chat", icon="💬", url_path="chat", default=True),
    st.Page(alerts_view.render, title="Alerts", icon="⏰", url_path="alerts"),
    st.Page(draft_view.render, title="Draft", icon="📝", url_path="draft"),
    st.Page(upload_view.render, title="Upload", icon="⬆️", url_path="upload"),
]

navigation = st.navigation(pages)
navigation.run()
