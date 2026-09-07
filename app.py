"""ミナモ サポート研修AI — Streamlit エントリポイント。"""

import streamlit as st

st.set_page_config(
    page_title="ミナモ サポート研修AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    {
        "研修": [
            st.Page("views/home.py", title="はじめに", icon=":material/home:", default=True),
            st.Page("views/training.py", title="模擬対応", icon=":material/support_agent:"),
            st.Page("views/review.py", title="評価", icon=":material/assessment:"),
        ],
        "技術": [
            st.Page("views/tech_demo.py", title="RAG技術デモ", icon=":material/manage_search:"),
        ],
    }
)
navigation.run()
