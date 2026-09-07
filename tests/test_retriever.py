"""検索。ローカルモードでもすべて動くこと。"""

import pytest

from src.rag.retriever import Retriever, excerpt_for


def test_index_is_built_from_the_documents(retriever):
    assert retriever.chunk_count > 0
    assert retriever.sources == ["faq.md", "manual.md"]


def test_empty_document_list_is_rejected():
    with pytest.raises(ValueError, match="登録されていません"):
        Retriever([])


def test_documents_without_body_are_rejected():
    from src.rag.loader import Document

    with pytest.raises(ValueError):
        Retriever([Document(name="a.md", text="   ")])


def test_retrieve_returns_the_matching_document_first(retriever):
    result = retriever.retrieve("スタンダードプランのスタッフ枠")
    assert result.passages
    assert result.passages[0].source == "faq.md"


def test_retrieve_discriminates_between_documents(retriever):
    assert retriever.retrieve("エスカレーション 返金").passages[0].source == "manual.md"


def test_retrieve_result_shape(retriever):
    passage = retriever.retrieve("リマインド").passages[0]
    payload = passage.as_dict()
    assert set(payload) == {"source", "text", "score", "excerpt"}
    assert isinstance(payload["score"], float)
    assert payload["excerpt"]


def test_retrieve_reports_elapsed_time_and_method(retriever):
    result = retriever.retrieve("リマインド")
    assert result.elapsed_ms >= 0
    assert result.method == "bm25"


def test_retrieve_rejects_an_empty_query(retriever):
    with pytest.raises(ValueError, match="質問を入力"):
        retriever.retrieve("   ")


def test_retrieve_returns_nothing_when_nothing_matches(retriever):
    result = retriever.retrieve("quantum chromodynamics lattice")
    assert result.is_empty
    assert result.passages == []


def test_threshold_filters_lower_ranked_matches(documents):
    """しきい値は「最上位に対する相対スコア」に効く。

    BM25 スコアは検索ごとに最大値で正規化しているため、最上位は常に 1.0 になり、
    1.0 未満のしきい値では落ちない。落とせるのは 2 位以下と、
    そもそも一語も一致しない（全チャンクが 0 点の）場合。
    """
    index = Retriever(documents, chunk_size=60, overlap=0)
    loose = index.retrieve("リマインド 確認 手順", top_k=10, threshold=0.0)
    strict = index.retrieve("リマインド 確認 手順", top_k=10, threshold=0.9)
    assert len(strict.passages) < len(loose.passages)
    assert strict.passages[0].score == pytest.approx(1.0)


def test_threshold_cannot_remove_the_top_match(retriever):
    assert retriever.retrieve("リマインド", threshold=0.99).passages


def test_top_k_limits_the_result(retriever):
    assert len(retriever.retrieve("リマインド 確認", top_k=1).passages) <= 1


def test_scores_are_ordered_descending(retriever):
    scores = [p.score for p in retriever.retrieve("リマインド 確認 手順", top_k=5).passages]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_relative_not_absolute(retriever):
    """既知の性質: 正規化しているため最上位は常に 1.0 になる。UI にも注記がある。"""
    assert retriever.retrieve("リマインド").passages[0].score == pytest.approx(1.0)


def test_smaller_chunks_produce_more_chunks(documents):
    coarse = Retriever(documents, chunk_size=800, overlap=0)
    fine = Retriever(documents, chunk_size=80, overlap=0)
    assert fine.chunk_count > coarse.chunk_count


def test_excerpt_centres_on_the_query_term():
    content = "前段" * 150 + "バウンスと表示されます" + "後段" * 150
    assert "バウンス" in excerpt_for("バウンス", content)


def test_excerpt_returns_short_text_unchanged():
    assert excerpt_for("何か", "短い本文") == "短い本文"


def test_embeddings_are_used_in_cloud_mode(cloud_mode, documents, fake_llm):
    fake = fake_llm("src.rag.retriever")
    index = Retriever(documents)
    assert index.use_embeddings is True
    result = index.retrieve("リマインド")
    assert result.method == "hybrid"
    assert any(call[0] == "embed" for call in fake.calls)


def test_embeddings_are_skipped_when_explicitly_disabled(cloud_mode, documents):
    assert Retriever(documents, use_embeddings=False).retrieve("リマインド").method == "bm25"
