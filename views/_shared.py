"""Page setup shared by every view: theme, sidebar status, cached resources."""

import streamlit as st

from src.config import ConfigError, cloud_requested_but_unavailable, set_session_api_key, settings
from src.conversation.persona import load_scenarios
from src.conversation.scoring import load_criteria
from src.errors import log_error, safe_message
from src.rag.loader import load_knowledge
from src.rag.retriever import Retriever

CSS = """
<style>
:root {
  --paper: #f6f7f9;
  --card: #ffffff;
  --ink: #1a1d21;
  --muted: #5a6270;
  --line: #dfe3e8;
  --brand: #0f6e63;
  --brand-soft: #e4f1ef;
  --warn: #9a5b12;
  --warn-soft: #fbf0df;
}
.stApp { background: var(--paper); }
.block-container { max-width: 1080px; padding-top: 3.4rem; padding-bottom: 4rem; }
h1 { font-size: 1.9rem !important; letter-spacing: -.01em; }
h2 { font-size: 1.25rem !important; margin-top: 1.6rem !important; }
h3 { font-size: 1.02rem !important; }
.kicker {
  color: var(--brand); font-size: .7rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; margin-bottom: .3rem;
}
.lede { color: var(--muted); max-width: 62ch; margin: .2rem 0 1.4rem; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 1rem 1.2rem; margin: .5rem 0 1rem;
}
.pill {
  border-radius: 999px; display: inline-block; font-size: .74rem;
  font-weight: 700; padding: .15rem .6rem;
}
.pill-local { background: var(--warn-soft); color: var(--warn); }
.pill-cloud { background: var(--brand-soft); color: var(--brand); }
.step { border-left: 2px solid var(--line); padding: .1rem 0 .1rem 1rem; margin-bottom: .9rem; }
.step b { display: block; color: var(--ink); }
.step span { color: var(--muted); font-size: .9rem; }
.plain { white-space: pre-wrap; line-height: 1.75; }
.excerpt {
  background: #f1f4f6; border-left: 3px solid var(--brand); border-radius: 0 6px 6px 0;
  font-size: .88rem; line-height: 1.7; margin: .3rem 0 .6rem; padding: .55rem .8rem;
  white-space: pre-wrap;
}
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .7rem .9rem;
}
section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--line); }
@media (max-width: 768px) { .block-container { padding: 2.5rem 1rem 3rem; } }
</style>
"""


def setup(kicker: str, title: str, lede: str = "") -> None:
    st.html(CSS)
    render_sidebar()
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def render_sidebar() -> None:
    # このセッションで直前に入力されたキーを、他の設定読み込みより先に反映する。
    # セッションをまたいで共有されない contextvar に置くだけで、.env やディスクは触らない。
    set_session_api_key(st.session_state.get("session_api_key"))
    config = settings()
    with st.sidebar:
        st.markdown("### ミナモ サポート研修AI")
        st.caption("架空のクラウド予約管理サービスを題材にした研修デモ")
        if config.is_cloud:
            st.markdown('<span class="pill pill-cloud">クラウドモード</span>', unsafe_allow_html=True)
            st.caption(f"生成: {config.chat_model}")
        else:
            st.markdown('<span class="pill pill-local">ローカルモード</span>', unsafe_allow_html=True)
            st.caption("外部APIを呼ばずに動作しています")
        if not config.is_cloud and cloud_requested_but_unavailable():
            st.warning("APP_MODE=cloud が指定されていますが、OPENAI_API_KEY が未設定のためローカルモードで動作しています。")

        with st.expander("🔑 自分のAPIキーで試す", expanded=False):
            st.caption(
                "入力したキーはこのブラウザセッション内だけで保持され、"
                "`.env` やディスクには一切書き込まれません。"
                "タブを閉じる・再読み込みすると失われます。"
            )
            st.text_input(
                "OpenAI APIキー",
                type="password",
                key="session_api_key",
                placeholder="sk-...",
                help="入力するとこのセッションだけクラウドモードで動作します。空にすると解除されます。",
            )
            if st.session_state.get("session_api_key"):
                st.caption("✓ このセッションはクラウドモードで動作しています。")
            st.caption(
                "使い切りや利用上限を設定したキーの使用を推奨します。"
                "共有環境でこのアプリを動かしている場合、検索用の埋め込みは"
                "最初にクラウドモードで検索した人のキーで計算され、"
                "以降の利用者と共有されることがあります。"
            )


def show_error(message: str, error: Exception | None = None) -> None:
    """Show a user-safe message. Internal detail goes to the log, redacted."""
    if error is not None:
        log_error(message, error)
        st.error(f"{message}\n\n{safe_message(error)}")
    else:
        st.error(message)


@st.cache_resource(show_spinner=False)
def get_retriever(use_embeddings: bool) -> Retriever:
    """Cached per mode, not globally.

    A UI-entered key can turn one session's requests cloud while another
    session in the same process stays local. Keying the cache on the mode
    keeps whichever session builds the index first from freezing that choice
    for everyone — otherwise a session with a key could still get a
    BM25-only retriever cached by an earlier, keyless visitor.
    """
    return Retriever(load_knowledge(), use_embeddings=use_embeddings)


@st.cache_data(show_spinner=False)
def get_scenarios():
    return load_scenarios()


@st.cache_data(show_spinner=False)
def get_criteria():
    return load_criteria()


def load_resources():
    """Load knowledge, scenarios and criteria, reporting failures safely."""
    try:
        return get_retriever(settings().is_cloud), get_scenarios(), get_criteria()
    except (ConfigError, ValueError, FileNotFoundError) as error:
        show_error("データの読み込みに失敗しました。", error)
        st.stop()
