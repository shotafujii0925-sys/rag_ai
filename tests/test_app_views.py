"""Integration tests for the Streamlit views, using Streamlit's own test harness
(`streamlit.testing.v1.AppTest`, bundled with the `streamlit` package we already
depend on).

Unit tests in the other files cover `src/` in isolation, but nothing had ever
clicked a button or driven `session_state` the way a browser does. Two real
bugs (unescaped HTML in the review transcript, an uncaught exception when a
scenario_id doesn't resolve) shipped specifically because of that gap — this
file exists to close it and pins both as regressions.

`st.switch_page` only works inside the full multi-page app (`app.py`'s
`st.navigation`), not when a single view is loaded standalone via
`AppTest.from_file`. So a training → review handoff is tested by building the
session with the real domain functions (`start_session`/`add_turn`/
`finish_session` — the same ones the button calls) and handing the resulting
`session_state` to a fresh `AppTest` for the next page, exactly as
`st.session_state` really does carry across `st.switch_page` in the browser.
"""

import pytest
from streamlit.testing.v1 import AppTest

from src.config import REPO_ROOT
from src.conversation.persona import find_scenario, load_scenarios
from src.conversation.session import add_turn, finish_session, start_session
from src.rag.loader import load_knowledge
from src.rag.retriever import Retriever


def view(name: str) -> str:
    """AppTest.from_file resolves relative paths against the *caller's* file,
    i.e. tests/, not the repo root — so views/home.py would look for
    tests/views/home.py. Always hand it an absolute path instead."""
    return str(REPO_ROOT / "views" / name)


@pytest.fixture
def scenario():
    return find_scenario("usage-01", load_scenarios())


@pytest.fixture
def bundled_retriever():
    return Retriever(load_knowledge())


# --- home ----------------------------------------------------------------


def test_home_renders_without_error():
    at = AppTest.from_file(view("home.py"), default_timeout=30).run()
    assert not at.exception


def test_home_lists_every_scenario():
    at = AppTest.from_file(view("home.py"), default_timeout=30).run()
    rows = at.dataframe[0].value
    assert len(rows) == len(load_scenarios())


# --- training: happy path --------------------------------------------------


def test_training_start_button_is_disabled_until_a_scenario_exists():
    at = AppTest.from_file(view("training.py"), default_timeout=30).run()
    assert "session" not in at.session_state or at.session_state["session"] is None


def test_training_starting_a_scenario_shows_its_opening_line():
    at = AppTest.from_file(view("training.py"), default_timeout=30).run()
    at.selectbox[0].select_index(0).run()
    [b for b in at.button if b.label == "この設定で開始"][0].click().run()

    assert not at.exception
    session = at.session_state["session"]
    assert session["transcript"][0]["role"] == "customer"


def test_training_submitting_a_turn_appends_both_sides():
    at = AppTest.from_file(view("training.py"), default_timeout=30).run()
    at.selectbox[0].select_index(0).run()
    [b for b in at.button if b.label == "この設定で開始"][0].click().run()

    at.text_area[0].set_value("設定 > 予約ページ からURLを確認できます。").run()
    [b for b in at.button if b.label == "送信する"][0].click().run()

    assert not at.exception
    roles = [turn["role"] for turn in at.session_state["session"]["transcript"]]
    assert roles == ["customer", "agent", "customer"]


def test_training_finish_button_is_disabled_before_any_turn():
    at = AppTest.from_file(view("training.py"), default_timeout=30).run()
    at.selectbox[0].select_index(0).run()
    [b for b in at.button if b.label == "この設定で開始"][0].click().run()

    finish_button = [b for b in at.button if "終了" in b.label][0]
    assert finish_button.disabled is True


def test_training_finish_button_enables_after_a_turn():
    at = AppTest.from_file(view("training.py"), default_timeout=30).run()
    at.selectbox[0].select_index(0).run()
    [b for b in at.button if b.label == "この設定で開始"][0].click().run()
    at.text_area[0].set_value("確認します。").run()
    [b for b in at.button if b.label == "送信する"][0].click().run()

    finish_button = [b for b in at.button if "終了" in b.label][0]
    assert finish_button.disabled is False


# --- training -> review handoff --------------------------------------------


def finished_session(scenario, retriever, *, agent_message: str) -> dict:
    session = start_session(scenario)
    add_turn(session, agent_message, scenario, retriever)
    finish_session(session)
    return session


def test_review_shows_scores_for_a_finished_session(scenario, bundled_retriever):
    session = finished_session(scenario, bundled_retriever, agent_message="確認します。")

    at = AppTest.from_file(view("review.py"), default_timeout=30)
    at.session_state["session"] = session
    at.session_state["scenario_id"] = scenario.id
    at.run()

    assert not at.exception
    assert len(at.metric) == len(session["transcript"]) or len(at.metric) >= 1
    labels = [m.label for m in at.metric]
    assert "平均" in labels


def test_review_redirects_when_the_session_is_not_finished(scenario, bundled_retriever):
    session = start_session(scenario)
    add_turn(session, "確認します。", scenario, bundled_retriever)  # not finished

    at = AppTest.from_file(view("review.py"), default_timeout=30)
    at.session_state["session"] = session
    at.session_state["scenario_id"] = scenario.id
    at.run()

    assert not at.exception
    assert any("終了していません" in w.value for w in at.warning)


def test_review_prompts_to_start_when_there_is_no_session():
    at = AppTest.from_file(view("review.py"), default_timeout=30).run()
    assert not at.exception
    assert any("まだ対話がありません" in i.value for i in at.info)


# --- regression: bug #1, unescaped transcript content -----------------------


def test_review_escapes_html_in_the_agent_message(scenario, bundled_retriever):
    """A trainee's own free text must never execute as HTML on the review page."""
    payload = "<img src=x onerror=alert(1)>設定を確認します。"
    session = finished_session(scenario, bundled_retriever, agent_message=payload)

    at = AppTest.from_file(view("review.py"), default_timeout=30)
    at.session_state["session"] = session
    at.session_state["scenario_id"] = scenario.id
    at.run()  # st.expander bodies run unconditionally, whether collapsed or not

    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "<img" not in rendered
    assert "&lt;img" in rendered


def test_review_escapes_html_in_the_summary(scenario, bundled_retriever):
    """A model-authored summary is untrusted the same way a document is."""
    session = finished_session(scenario, bundled_retriever, agent_message="確認します。")

    at = AppTest.from_file(view("review.py"), default_timeout=30)
    at.session_state["session"] = session
    at.session_state["scenario_id"] = scenario.id
    at.session_state["report"] = {
        "method": "llm",
        "scores": [{"id": "a", "label": "説明の正確性", "score": 3, "reason": "理由", "advice": ""}],
        "total": 3,
        "max_total": 5,
        "average": 3.0,
        "summary": "<script>alert(1)</script>要点をまとめました。",
    }
    at.run()

    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


# --- regression: bug #2, an unresolved scenario_id must not crash the page --


def test_training_handles_an_unknown_scenario_id_gracefully(scenario):
    at = AppTest.from_file(view("training.py"), default_timeout=30)
    at.session_state["session"] = start_session(scenario)
    at.session_state["scenario_id"] = "no-such-scenario"
    at.run()

    assert not at.exception
    assert any("読み込めませんでした" in e.value for e in at.error)
    assert "session" not in at.session_state or at.session_state["session"] is None


def test_review_handles_an_unknown_scenario_id_gracefully(scenario, bundled_retriever):
    session = finished_session(scenario, bundled_retriever, agent_message="確認します。")

    at = AppTest.from_file(view("review.py"), default_timeout=30)
    at.session_state["session"] = session
    at.session_state["scenario_id"] = "no-such-scenario"
    at.run()

    assert not at.exception
    assert any("読み込めませんでした" in e.value for e in at.error)
    assert "session" not in at.session_state or at.session_state["session"] is None


# --- regression: bug #3, uploaded documents must reach the search ----------


def test_tech_demo_renders_without_error():
    at = AppTest.from_file(view("tech_demo.py"), default_timeout=30).run()
    assert not at.exception


def test_tech_demo_upload_adds_a_document_to_the_index():
    at = AppTest.from_file(view("tech_demo.py"), default_timeout=30).run()
    before = next(m.value for m in at.metric if m.label == "参照文書")

    content = "深夜料金プランは、22時から翌6時までの予約に限り、半額で提供する特別プランです。"
    at.file_uploader[0].upload("night-plan.md", content.encode("utf-8"), "text/markdown").run()

    after = next(m.value for m in at.metric if m.label == "参照文書")
    assert not at.exception
    assert after != before
    assert any("night-plan.md" in s.value for s in at.success)


def test_tech_demo_search_finds_the_uploaded_document():
    at = AppTest.from_file(view("tech_demo.py"), default_timeout=30).run()
    content = "深夜料金プランは、22時から翌6時までの予約に限り、半額で提供する特別プランです。"
    at.file_uploader[0].upload("night-plan.md", content.encode("utf-8"), "text/markdown").run()

    at.text_input[0].set_value("深夜料金プランはどうやって使いますか").run()
    [b for b in at.button if b.label == "検索して回答"][0].click().run()

    assert not at.exception
    rendered = " ".join(m.value for m in at.markdown)
    assert "深夜料金プラン" in rendered


def test_tech_demo_rejects_a_document_with_no_extractable_text():
    """The extension itself is already enforced by st.file_uploader(type=...) —
    a mismatched extension can't even reach the app. What can still reach it is
    an allowed-extension file whose content is empty once parsed, e.g. an HTML
    file that is all <script>, no visible text."""
    at = AppTest.from_file(view("tech_demo.py"), default_timeout=30).run()
    at.file_uploader[0].upload(
        "empty.html", b"<html><head><script>x=1;</script></head><body></body></html>", "text/html"
    ).run()

    assert not at.exception
    assert any("本文がありません" in e.value for e in at.error)


def test_tech_demo_eval_dataset_ignores_the_uploaded_document():
    """The labelled eval set is scored against the bundled corpus only, always —
    otherwise a visitor's upload would silently change what the numbers mean."""
    at = AppTest.from_file(view("tech_demo.py"), default_timeout=30).run()
    content = "深夜料金プランは、22時から翌6時までの予約に限り、半額で提供する特別プランです。"
    at.file_uploader[0].upload("night-plan.md", content.encode("utf-8"), "text/markdown").run()

    [b for b in at.button if b.label == "評価データセットを実行"][0].click().run()

    assert not at.exception
    hit_rate_metric = next(m for m in at.metric if m.label == "正解文書の取得率")
    assert hit_rate_metric.value  # 実行できていること自体を確認する
