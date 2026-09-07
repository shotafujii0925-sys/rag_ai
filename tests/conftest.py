"""Shared fixtures. No test may reach the network.

`offline` forces local mode, so any cloud-only path raises ConfigError instead of
dialling out. Tests that need model behaviour install a fake through `fake_llm`,
which patches the names *in the module under test* — the RAG modules import
`chat` lazily from `..llm`, so patching must target where the name is looked up.
"""

import json

import pytest

from src import config


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in (
        "APP_MODE", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CHAT_MODEL",
        "EMBEDDING_MODEL", "TRANSCRIPTION_MODEL", "TEMPERATURE", "TIMEOUT_SECONDS",
        "MAX_RETRIES", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "SCORE_THRESHOLD",
        "MAX_UPLOAD_MB",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_MODE", "local")
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()


@pytest.fixture
def cloud_mode(monkeypatch):
    """Switch to cloud mode without a real key. Only use with fake_llm."""
    monkeypatch.setenv("APP_MODE", "cloud")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "test-key-for-tests-only")
    config.settings.cache_clear()
    return config.settings()


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("テスト中にネットワークへ接続しようとしました。")

    monkeypatch.setattr("socket.socket.connect", refuse)
    monkeypatch.setattr("socket.create_connection", refuse)


class FakeLLM:
    def __init__(self) -> None:
        self.chat_replies: list[str] = []
        self.json_replies: list = []
        self.calls: list[tuple[str, str]] = []
        self.system_prompts: list[str] = []
        self.embedding_map: dict[str, list[float]] = {}
        self.raise_on_chat: Exception | None = None

    def chat(self, system: str, user: str, **kwargs) -> str:
        if self.raise_on_chat:
            raise self.raise_on_chat
        self.calls.append(("chat", user))
        self.system_prompts.append(system)
        return self.chat_replies.pop(0) if self.chat_replies else "生成された回答"

    def chat_json(self, system: str, user: str, **kwargs) -> dict:
        if self.raise_on_chat:
            raise self.raise_on_chat
        self.calls.append(("chat_json", user))
        self.system_prompts.append(system)
        if not self.json_replies:
            return {}
        reply = self.json_replies.pop(0)
        return json.loads(reply) if isinstance(reply, str) else reply

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        self.calls.append(("embed", " ".join(texts)[:60]))
        return [self.embedding_map.get(text, [1.0, 0.0, 0.0]) for text in texts]

    def transcribe(self, audio: bytes, **kwargs) -> str:
        self.calls.append(("transcribe", str(len(audio))))
        return "文字起こし結果"


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()

    def install(*module_paths: str) -> FakeLLM:
        for path in module_paths:
            for name in ("chat", "chat_json", "embed", "transcribe"):
                monkeypatch.setattr(f"{path}.{name}", getattr(fake, name), raising=False)
        return fake

    # 遅延 import される src.llm 側も差し替える。
    for name in ("chat", "chat_json", "embed", "transcribe"):
        monkeypatch.setattr(f"src.llm.{name}", getattr(fake, name), raising=False)
    return install


@pytest.fixture
def documents():
    from src.rag.loader import Document

    return [
        Document(
            name="faq.md",
            text=(
                "## プラン\nスタンダードプランのスタッフ枠は5名です。月額5,500円です。\n\n"
                "## リマインド\nリマインドメールが届かないときは迷惑メールフォルダを確認します。"
                "送信履歴にバウンスと表示されている場合は再送しません。"
            ),
        ),
        Document(
            name="manual.md",
            text=(
                "## 応対手順\n名乗りと受付、傾聴、事実確認、説明、合意の順に進めます。\n\n"
                "## エスカレーション\n返金や補償の要求があるときはサポート責任者へ引き継ぎます。"
            ),
        ),
    ]


@pytest.fixture
def retriever(documents):
    from src.rag.retriever import Retriever

    return Retriever(documents)


@pytest.fixture
def scenario():
    from src.conversation.persona import Persona, Scenario

    return Scenario(
        id="test-01",
        category="不具合対応",
        title="リマインドが届かない",
        difficulty="標準",
        persona=Persona(
            name="急いでいる店舗スタッフ",
            description="要点を早く知りたい",
            tone="早口",
            hidden_context="送信履歴を見ていない",
            style="hurried",
        ),
        opening="リマインドが届いていないようです。",
        must_cover=["迷惑メールフォルダを確認", "送信履歴を確認"],
        reference_hint="faq.md",
    )
