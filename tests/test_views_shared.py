"""views/_shared.py: retriever caching and the session-key sidebar wiring.

Only the parts that don't require a live Streamlit runtime — `st.cache_resource`
works outside one (with a harmless warning), but rendering the sidebar itself
does not, so that part stays exercised manually and in `test_config.py` /
`test_llm_client.py`.
"""

from src import config
from views._shared import get_retriever


def test_get_retriever_is_cached_per_mode(documents, monkeypatch):
    monkeypatch.setattr("views._shared.load_knowledge", lambda: documents)
    local_a = get_retriever(False)
    local_b = get_retriever(False)
    cloud = get_retriever(True)

    assert local_a is local_b
    assert local_a is not cloud
    assert local_a.use_embeddings is False
    assert cloud.use_embeddings is True


def test_get_retriever_does_not_freeze_a_session_key_users_mode(documents, monkeypatch):
    """A visitor who pastes a key must not be stuck with an earlier BM25-only cache.

    Regression test for the bug this feature would otherwise introduce: before
    keying the cache by mode, whichever session built the retriever first
    decided `use_embeddings` for every later session in the same process.
    """
    monkeypatch.setattr("views._shared.load_knowledge", lambda: documents)
    monkeypatch.setenv("APP_MODE", "local")
    config.settings.cache_clear()

    # 最初の訪問者はキーなし。
    first_visitor = get_retriever(config.settings().is_cloud)
    assert first_visitor.use_embeddings is False

    # 次の訪問者がセッションキーを入力してクラウドモードになる。
    config.set_session_api_key("sk-visitor-b")
    try:
        second_visitor = get_retriever(config.settings().is_cloud)
        assert second_visitor.use_embeddings is True
        assert second_visitor is not first_visitor
    finally:
        config.set_session_api_key(None)
