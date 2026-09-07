"""文書読み込みと分割。"""

import json

import pytest

from src.rag.loader import (
    decode,
    escape_for_display,
    html_to_text,
    load_json,
    load_knowledge,
    normalize,
    read_bytes,
    read_file,
)
from src.rag.splitter import split_documents, split_text, tokenize

HTML = """<!doctype html>
<html><head><title>社内メモ</title><style>.a{color:red}</style></head>
<body><nav>ホーム</nav><h1>第1章</h1>
<p>領収書は&nbsp;必要です。</p>
<table><tr><th>区分</th><td>上限</td></tr></table>
<script>alert('x')</script></body></html>"""


# --- 正規化 -------------------------------------------------------------------


def test_normalize_collapses_spacing():
    assert normalize("a  b\r\n\n\n\nc") == "a b\n\nc"


def test_normalize_collapses_nbsp_and_ideographic_space():
    assert normalize("領収書 の　提出") == "領収書 の 提出"


def test_normalize_accepts_empty_input():
    assert normalize(None) == ""


# --- HTML ---------------------------------------------------------------------


def test_html_extracts_body_text():
    text = html_to_text(HTML)
    assert "領収書は 必要です。" in text
    assert "第1章" in text


def test_html_drops_scripts_styles_and_chrome():
    text = html_to_text(HTML)
    assert "alert" not in text
    assert "color:red" not in text
    assert "ホーム" not in text
    assert "社内メモ" not in text


def test_html_joins_table_cells():
    assert "区分 | 上限" in html_to_text(HTML)


def test_html_survives_unclosed_tags():
    assert "本文" in html_to_text("<div><p>本文")


def test_decode_honours_the_declared_charset():
    assert "領収書" in decode('<meta charset="shift_jis"><p>領収書</p>'.encode("shift_jis"))


def test_decode_falls_back_for_unknown_charset():
    assert "領収書" in decode('<meta charset="bogus"><p>領収書</p>'.encode("utf-8"))


def test_escape_for_display_neutralises_markup():
    assert escape_for_display("<script>x</script>") == "&lt;script&gt;x&lt;/script&gt;"


# --- アップロード検証 ---------------------------------------------------------


def test_read_bytes_accepts_markdown():
    assert read_bytes("faq.md", "本文です".encode("utf-8")).text == "本文です"


def test_read_bytes_routes_html_through_the_parser():
    assert "alert" not in read_bytes("page.html", HTML.encode("utf-8")).text


def test_read_bytes_rejects_an_unsupported_extension():
    with pytest.raises(ValueError, match="対応していない形式"):
        read_bytes("data.exe", b"MZ")


def test_read_bytes_rejects_an_oversized_file(monkeypatch):
    from src import config

    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    config.settings.cache_clear()
    with pytest.raises(ValueError, match="大きすぎます"):
        read_bytes("big.md", b"x" * (2 * 1024 * 1024))


def test_read_bytes_rejects_an_empty_document():
    with pytest.raises(ValueError, match="本文がありません"):
        read_bytes("empty.md", b"   ")


def test_read_bytes_strips_directory_traversal_from_the_name():
    assert read_bytes("../../etc/passwd.md", b"body").name == "passwd.md"


def test_read_file_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path / "missing.md")


def test_load_knowledge_reads_the_bundled_documents():
    names = {document.name for document in load_knowledge()}
    assert {"public_faq.md", "support_manual.md"} <= names


def test_load_knowledge_reports_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_knowledge(tmp_path)


def test_load_json_reports_broken_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_json(path)


def test_load_json_reads_valid_json(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_json(path) == {"a": 1}


# --- 分割 ---------------------------------------------------------------------


def test_tokenize_splits_cjk_into_bigrams():
    assert tokenize("経費精算") == ["経費", "費精", "精算"]


def test_tokenize_keeps_latin_words_whole():
    assert tokenize("RAG evaluation") == ["rag", "evaluation"]


def test_split_keeps_short_text_intact():
    assert split_text("短い本文") == ["短い本文"]


def test_split_returns_nothing_for_blank_text():
    assert split_text("   \n ") == []


def test_split_respects_the_size_limit():
    chunks = split_text("あ" * 2000, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_split_covers_the_whole_document():
    body = "".join(f"文{i}。" for i in range(300))
    assert set(body) <= set("".join(split_text(body, chunk_size=250, overlap=50)))


def test_split_prefers_a_sentence_boundary():
    body = "あ" * 200 + "。" + "い" * 200
    assert split_text(body, chunk_size=260, overlap=40)[0].endswith("。")


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, 100), (100, 150)])
def test_split_rejects_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("本文", chunk_size=size, overlap=overlap)


def test_split_documents_labels_each_chunk_with_its_source(documents):
    chunks = split_documents(documents, chunk_size=100, overlap=0)
    assert {chunk.source for chunk in chunks} == {"faq.md", "manual.md"}
    assert all(chunk.position >= 0 for chunk in chunks)
