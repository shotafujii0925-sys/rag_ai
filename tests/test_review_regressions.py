"""Regression checks for document loss and inconsistent application state."""

from copy import deepcopy

import pytest

from src.conversation.persona import asked_question
from src.conversation.scoring import evaluate
from src.conversation.session import add_turn, finish_session, start_session
from src.rag.generator import answer_question
from src.rag.retriever import Retriever


@pytest.mark.parametrize("stage", ["retrieval", "generation"])
def test_failed_turn_leaves_session_unchanged(stage, cloud_mode, scenario, documents, monkeypatch, fake_llm):
    index = Retriever(documents, use_embeddings=False)
    fake = fake_llm("src.conversation.persona")
    session = start_session(scenario)
    before = deepcopy(session)
    if stage == "retrieval":
        def fail(*args, **kwargs):
            raise TimeoutError("timeout")
        monkeypatch.setattr(index, "retrieve", fail)
    else:
        fake.raise_on_chat = TimeoutError("timeout")
    with pytest.raises(TimeoutError):
        add_turn(session, "送信履歴を確認します。", scenario, index)
    assert session == before


def test_model_scoring_api_error_falls_back(cloud_mode, scenario, documents, fake_llm):
    index = Retriever(documents, use_embeddings=False)
    fake = fake_llm("src.conversation.persona")
    session = start_session(scenario)
    add_turn(session, "送信履歴を確認します。", scenario, index)
    finish_session(session)
    fake.raise_on_chat = TimeoutError("upstream timeout")
    report = evaluate(session, scenario)
    assert report["method"] == "rules"
    assert len(report["scores"]) == 5
    assert "ルールベース" in report["note"]


def test_existing_retrieval_result_is_used_without_search(retriever, monkeypatch):
    result = retriever.retrieve("リマインド", threshold=1.0)
    def fail(*args, **kwargs):
        raise AssertionError("duplicate retrieval")
    monkeypatch.setattr(retriever, "retrieve", fail)
    answer = answer_question("リマインド", retriever, result=result)
    assert answer.passages == result.passages
    assert answer.elapsed_ms >= result.elapsed_ms


def test_threshold_one_keeps_the_best_positive_match(retriever):
    result = retriever.retrieve("リマインド", threshold=1.0)
    assert result.passages
    assert all(p.score == 1.0 for p in result.passages)
    assert not retriever.retrieve("quantum chromodynamics lattice", threshold=0.0).passages


def test_hybrid_score_is_normalized_after_combination(cloud_mode, documents, fake_llm):
    fake_llm("src.rag.retriever")
    index = Retriever(documents)
    result = index.retrieve("リマインド", threshold=1.0)
    assert result.method == "hybrid"
    assert result.passages
    assert result.passages[0].score == 1.0


def test_compatible_endpoint_key_is_redacted(cloud_mode, monkeypatch):
    from src import config
    from src.errors import redact

    key = "custom-" + "credential-for-test"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    config.settings.cache_clear()
    assert key not in redact("upstream rejected " + key)


@pytest.mark.parametrize("text,expected", [
    ("確認しますからお待ちください", False),
    ("重要ですから確認します", False),
    ("送信履歴を確認できますか。", True),
    ("ご利用のプランは何ですか", True),
    ("本当？", True),
])
def test_question_detection_uses_sentence_endings(text, expected):
    assert asked_question(text) is expected
