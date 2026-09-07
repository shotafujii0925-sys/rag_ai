"""RAG評価指標。ラベルベースの2指標はモデルなしで動くこと。"""

import pytest

from src.config import ConfigError
from src.rag.evaluator import (
    answer_relevancy,
    context_precision,
    context_recall,
    context_recall_llm,
    faithfulness,
)
from src.rag.retriever import Passage

CONTEXTS = [
    "スタンダードプランのスタッフ枠は5名です。",
    "リマインドが届かないときは迷惑メールフォルダを確認します。",
]


def passage(source: str) -> Passage:
    return Passage(source=source, text="本文", score=1.0, excerpt="本文")


# --- Context Precision（ラベル） ---------------------------------------------


def test_context_precision_counts_matching_sources():
    passages = [passage("faq.md"), passage("faq.md"), passage("manual.md"), passage("other.md")]
    result = context_precision(passages, "faq.md")
    assert result["score"] == 0.5
    assert result["method"] == "label"


def test_context_precision_is_one_when_all_match():
    assert context_precision([passage("faq.md")], "faq.md")["score"] == 1.0


def test_context_precision_of_an_empty_result_is_zero():
    assert context_precision([], "faq.md")["score"] == 0.0


def test_context_precision_needs_no_model():
    assert context_precision([passage("faq.md")], "faq.md")["method"] == "label"


# --- Context Recall（ラベル） -------------------------------------------------


def test_context_recall_is_one_when_the_context_covers_the_answer():
    result = context_recall("スタンダードプランのスタッフ枠は5名です。", CONTEXTS)
    assert result["score"] == 1.0
    assert result["method"] == "label"


def test_context_recall_is_zero_when_the_context_is_unrelated():
    assert context_recall("解約後90日でデータは削除されます。", ["天気の話題です。"])["score"] == 0.0


def test_context_recall_counts_each_sentence():
    result = context_recall(
        "スタンダードプランのスタッフ枠は5名です。解約後90日でデータは削除されます。", CONTEXTS
    )
    assert result["total"] == 2
    assert 0.0 < result["score"] < 1.0


def test_context_recall_with_no_context_is_zero():
    assert context_recall("何かの文です。", [])["score"] == 0.0


def test_context_recall_rejects_an_empty_ground_truth():
    with pytest.raises(ValueError, match="想定回答"):
        context_recall("   ", CONTEXTS)


# --- Faithfulness（モデル） ---------------------------------------------------


def test_faithfulness_requires_cloud_mode():
    with pytest.raises(ConfigError, match="クラウドモード"):
        faithfulness("回答", CONTEXTS)


def test_faithfulness_scores_supported_claims(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [
        {
            "claims": [
                {"claim": "枠は5名", "supported": True},
                {"claim": "月額は無料", "supported": False},
            ]
        }
    ]
    result = faithfulness("回答本文", CONTEXTS)
    assert result["score"] == 0.5
    assert result["method"] == "llm"


def test_faithfulness_counts_only_an_explicit_true(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{"claims": [{"claim": "曖昧", "supported": "true"}]}]
    assert faithfulness("回答本文", CONTEXTS)["score"] == 0.0


def test_faithfulness_handles_a_reply_without_claims(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{}]
    assert faithfulness("回答本文", CONTEXTS)["total"] == 0


def test_faithfulness_ignores_malformed_entries(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{"claims": ["文字列", None, {"claim": "有効", "supported": True}]}]
    assert faithfulness("回答本文", CONTEXTS)["total"] == 1


def test_faithfulness_rejects_blank_input(cloud_mode, fake_llm):
    fake_llm("src.rag.evaluator")
    with pytest.raises(ValueError):
        faithfulness("   ", CONTEXTS)
    with pytest.raises(ValueError):
        faithfulness("回答", [])


# --- Answer Relevancy（モデル） -----------------------------------------------


def test_answer_relevancy_requires_cloud_mode():
    with pytest.raises(ConfigError, match="クラウドモード"):
        answer_relevancy("質問", "回答")


def test_answer_relevancy_scores_similar_questions_high(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{"questions": ["スタッフ枠は何名ですか"], "noncommittal": False}]
    fake.embedding_map = {
        "スタッフ枠は何人までですか": [1.0, 0.0],
        "スタッフ枠は何名ですか": [1.0, 0.0],
    }
    result = answer_relevancy("スタッフ枠は何人までですか", "5名です。")
    assert result["score"] == pytest.approx(1.0)


def test_answer_relevancy_is_zero_for_an_evasive_answer(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{"questions": ["何かの質問"], "noncommittal": True}]
    assert answer_relevancy("質問", "確認できません。")["score"] == 0.0


def test_answer_relevancy_is_low_for_an_off_topic_answer(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{"questions": ["解約について"], "noncommittal": False}]
    fake.embedding_map = {"スタッフ枠は何人ですか": [1.0, 0.0], "解約について": [0.0, 1.0]}
    assert answer_relevancy("スタッフ枠は何人ですか", "解約は90日です。")["score"] == 0.0


def test_answer_relevancy_handles_a_reply_without_questions(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [{"questions": []}]
    assert answer_relevancy("質問", "回答")["score"] == 0.0


# --- Context Recall（モデル版） -----------------------------------------------


def test_context_recall_llm_requires_cloud_mode():
    with pytest.raises(ConfigError, match="クラウドモード"):
        context_recall_llm("想定回答です。", CONTEXTS)


def test_context_recall_llm_scores_attributed_sentences(cloud_mode, fake_llm):
    fake = fake_llm("src.rag.evaluator")
    fake.json_replies = [
        {"sentences": [{"sentence": "a", "attributed": True}, {"sentence": "b", "attributed": False}]}
    ]
    result = context_recall_llm("aです。bです。", CONTEXTS)
    assert result["score"] == 0.5
    assert result["method"] == "llm"
