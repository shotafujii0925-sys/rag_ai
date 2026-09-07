"""The OpenAI client cache. No network — only that the client is scoped per key.

This matters because a UI-entered key can differ between concurrent sessions in
the same process: caching the client globally would let one visitor's requests
run — and bill — against another visitor's key.
"""

from src import config
from src.llm import _client, _client_for


def test_client_for_returns_the_same_instance_for_the_same_key():
    first = _client_for("sk-aaa", None, 30, 2)
    second = _client_for("sk-aaa", None, 30, 2)
    assert first is second


def test_client_for_returns_different_instances_for_different_keys():
    first = _client_for("sk-aaa", None, 30, 2)
    second = _client_for("sk-bbb", None, 30, 2)
    assert first is not second
    assert first.api_key == "sk-aaa"
    assert second.api_key == "sk-bbb"


def test_client_for_is_scoped_by_endpoint_too():
    first = _client_for("sk-aaa", None, 30, 2)
    second = _client_for("sk-aaa", "https://gateway.example/v1", 30, 2)
    assert first is not second


def test_client_reflects_the_current_session_key(monkeypatch):
    monkeypatch.setenv("APP_MODE", "cloud")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    config.settings.cache_clear()
    try:
        assert _client().api_key == "sk-env-key"

        config.set_session_api_key("sk-session-key")
        assert _client().api_key == "sk-session-key"
    finally:
        config.set_session_api_key(None)


def test_two_concurrent_sessions_never_share_a_client(monkeypatch):
    """Simulates two visitors with different keys in the same process."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("APP_MODE", "local")
    config.settings.cache_clear()
    try:
        config.set_session_api_key("sk-visitor-a")
        client_a = _client()

        config.set_session_api_key("sk-visitor-b")
        client_b = _client()

        assert client_a.api_key == "sk-visitor-a"
        assert client_b.api_key == "sk-visitor-b"
        assert client_a is not client_b
    finally:
        config.set_session_api_key(None)
