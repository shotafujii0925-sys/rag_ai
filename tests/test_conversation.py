"""シナリオ、対話履歴、評価。"""

import pytest

from src.config import ConfigError
from src.conversation.persona import (
    REPLIES,
    asked_question,
    choose_reaction,
    find_scenario,
    generate_reply,
    has_guess,
    load_scenarios,
    local_reply,
    uncovered,
)
from src.conversation.scoring import evaluate, load_criteria, score_with_model
from src.conversation.session import (
    MAX_MESSAGE_CHARS,
    add_turn,
    finish_session,
    start_session,
    validate_message,
)


# --- シナリオ -----------------------------------------------------------------


def test_bundled_scenarios_load():
    scenarios = load_scenarios()
    assert len(scenarios) >= 4
    assert {s.category for s in scenarios} >= {"商品の利用方法", "契約内容の変更", "不具合対応", "苦情対応"}


def test_every_scenario_has_required_fields():
    for scenario in load_scenarios():
        assert scenario.opening
        assert scenario.must_cover
        assert scenario.persona.name
        assert scenario.persona.style in REPLIES, f"{scenario.id} の style が未定義です"


def test_find_scenario_returns_the_match():
    scenarios = load_scenarios()
    assert find_scenario(scenarios[0].id, scenarios).id == scenarios[0].id


def test_find_scenario_reports_an_unknown_id():
    with pytest.raises(ValueError, match="見つかりません"):
        find_scenario("nope", load_scenarios())


def test_scenarios_file_must_not_be_empty(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text('{"scenarios": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenarios(path)


def test_scenario_missing_a_field_is_rejected(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text('{"scenarios": [{"id": "a"}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="category"):
        load_scenarios(path)


def test_unknown_persona_style_is_rejected(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(
        '{"scenarios":[{"id":"a","category":"c","title":"t","opening":"o",'
        '"persona":{"style":"bogus"}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="style"):
        load_scenarios(path)


# --- ローカルモードの顧客役 ---------------------------------------------------


def test_has_guess_detects_hedging():
    assert has_guess("たぶん設定の問題です")
    assert has_guess("そのはずです")
    assert not has_guess("設定画面で確認できます")


def test_asked_question_detects_a_confirming_question():
    assert asked_question("いつ頃から発生していますか")
    assert asked_question("再現しますか？")
    assert not asked_question("設定を開いてください")


def test_uncovered_lists_what_is_still_missing(scenario):
    assert uncovered(scenario, "") == scenario.must_cover
    assert "迷惑メールフォルダを確認" not in uncovered(
        scenario, "迷惑メールフォルダを確認してください"
    )


def test_reaction_presses_on_a_guess(scenario):
    assert choose_reaction(scenario, ["たぶん設定です"], "たぶん設定です") == "guess"


def test_reaction_asks_about_an_uncovered_requirement(scenario):
    assert choose_reaction(scenario, ["確認します"], "確認します") == "missing"


def test_reaction_complains_when_nothing_was_asked(scenario):
    covered = "迷惑メールフォルダを確認し、送信履歴を確認してください。"
    assert choose_reaction(scenario, [covered], covered) == "no_question"


def test_reaction_deepens_after_a_single_thorough_turn(scenario):
    covered = "迷惑メールフォルダを確認し、送信履歴を確認しますか。"
    assert choose_reaction(scenario, [covered], covered) == "deepen"


def test_reaction_closes_once_everything_is_handled(scenario):
    covered = "迷惑メールフォルダを確認し、送信履歴を確認しますか。"
    assert choose_reaction(scenario, [covered, "他にありますか"], covered) == "closing"


def test_hard_scenarios_push_before_closing():
    hard = next(s for s in load_scenarios() if s.difficulty == "難しい")
    everything = " ".join(hard.must_cover) + " いかがですか"
    assert choose_reaction(hard, [everything, "他には"], everything) == "push"


def test_local_reply_names_the_missing_requirement(scenario):
    reply = local_reply(scenario, ["確認します"], "確認します")
    assert scenario.must_cover[0] in reply


def test_local_reply_wording_follows_the_persona_style():
    scenarios = {s.persona.style: s for s in load_scenarios()}
    angry = local_reply(scenarios["angry"], ["たぶん大丈夫です"], "たぶん大丈夫です")
    anxious = local_reply(scenarios["anxious"], ["たぶん大丈夫です"], "たぶん大丈夫です")
    assert angry != anxious
    assert "責任" in angry


def test_local_reply_is_deterministic(scenario):
    first = local_reply(scenario, ["確認します"], "確認します")
    second = local_reply(scenario, ["確認します"], "確認します")
    assert first == second


def test_generate_reply_requires_cloud_mode(scenario):
    with pytest.raises(ConfigError, match="クラウドモード"):
        generate_reply(scenario, [], "文脈")


def test_generate_reply_stays_in_character(cloud_mode, scenario, fake_llm):
    fake = fake_llm("src.conversation.persona")
    fake.chat_replies = ["それはどこで見られますか。"]
    reply = generate_reply(scenario, [{"role": "agent", "content": "確認します"}], "文脈")
    assert reply == "それはどこで見られますか。"
    assert "顧客役" in fake.system_prompts[0]
    assert "指示ではありません" in fake.system_prompts[0]


# --- 対話履歴 -----------------------------------------------------------------


def test_session_starts_with_the_opening(scenario):
    session = start_session(scenario)
    assert session["transcript"] == [{"role": "customer", "content": scenario.opening}]
    assert session["finished"] is False


def test_add_turn_records_both_sides(scenario, retriever):
    session = start_session(scenario)
    add_turn(session, "迷惑メールフォルダを確認してください。", scenario, retriever)
    assert [turn["role"] for turn in session["transcript"]] == ["customer", "agent", "customer"]


def test_add_turn_logs_retrieval(scenario, retriever):
    session = start_session(scenario)
    add_turn(session, "送信履歴を確認します。", scenario, retriever)
    log = session["retrieval_log"][0]
    assert log["query"]
    assert log["method"] == "bm25"
    assert log["elapsed_ms"] >= 0


@pytest.mark.parametrize("message", ["", "   ", "\n"])
def test_empty_message_is_rejected(message):
    with pytest.raises(ValueError, match="入力してください"):
        validate_message(message)


def test_overlong_message_is_rejected():
    with pytest.raises(ValueError, match="文字以内"):
        validate_message("あ" * (MAX_MESSAGE_CHARS + 1))


def test_adding_a_turn_after_finishing_is_rejected(scenario, retriever):
    session = start_session(scenario)
    add_turn(session, "確認します。", scenario, retriever)
    finish_session(session)
    with pytest.raises(ValueError, match="終了しています"):
        add_turn(session, "追加です。", scenario, retriever)


def test_finishing_without_any_turn_is_rejected(scenario):
    with pytest.raises(ValueError, match="応対していません"):
        finish_session(start_session(scenario))


def test_finish_marks_the_session(scenario, retriever):
    session = start_session(scenario)
    add_turn(session, "確認します。", scenario, retriever)
    assert finish_session(session)["finished"] is True
    assert session["finished_at"]


def test_local_mode_never_calls_the_model(scenario, retriever, fake_llm):
    fake = fake_llm("src.conversation.persona")
    session = start_session(scenario)
    add_turn(session, "確認します。", scenario, retriever)
    assert fake.calls == []


def test_local_mode_reply_reacts_to_the_message(scenario, retriever):
    guessed = start_session(scenario)
    add_turn(guessed, "たぶん設定の問題だと思います。", scenario, retriever)

    asked = start_session(scenario)
    add_turn(asked, "確認します。", scenario, retriever)

    assert guessed["transcript"][-1]["content"] != asked["transcript"][-1]["content"]


# --- 評価 ---------------------------------------------------------------------


def finished_session(scenario, retriever, message):
    session = start_session(scenario)
    add_turn(session, message, scenario, retriever)
    return finish_session(session)


def test_criteria_load_with_five_axes():
    criteria = load_criteria()
    assert len(criteria) == 5
    assert {item["id"] for item in criteria} == {
        "accuracy", "empathy", "problem_grasp", "resolution", "grounding"
    }


def test_evaluation_requires_a_finished_session(scenario, retriever):
    with pytest.raises(ValueError, match="終了してから"):
        evaluate(start_session(scenario), scenario)


def test_local_scoring_returns_every_axis(scenario, retriever):
    session = finished_session(scenario, retriever, "迷惑メールフォルダを確認してください。")
    report = evaluate(session, scenario)
    assert report["method"] == "rules"
    assert len(report["scores"]) == 5
    assert 1 <= report["average"] <= 5


def test_coverage_rewards_touching_the_required_points(scenario, retriever):
    weak = finished_session(scenario, retriever, "確認します。")
    strong = finished_session(
        scenario, retriever, "迷惑メールフォルダを確認してください。次に送信履歴を確認します。"
    )
    weak_score = next(s for s in evaluate(weak, scenario)["scores"] if s["id"] == "accuracy")
    strong_score = next(s for s in evaluate(strong, scenario)["scores"] if s["id"] == "accuracy")
    assert strong_score["score"] > weak_score["score"]


def test_grounding_penalises_unsupported_guesses(scenario, retriever):
    guess = finished_session(scenario, retriever, "たぶん設定の問題だと思います。")
    grounded = finished_session(scenario, retriever, "FAQの記載を確認したところ送信履歴で判断できます。")
    guess_score = next(s for s in evaluate(guess, scenario)["scores"] if s["id"] == "grounding")
    grounded_score = next(s for s in evaluate(grounded, scenario)["scores"] if s["id"] == "grounding")
    assert grounded_score["score"] > guess_score["score"]


def test_every_axis_carries_a_reason(scenario, retriever):
    session = finished_session(scenario, retriever, "確認いたします。")
    assert all(item["reason"] for item in evaluate(session, scenario)["scores"])


def test_model_scoring_uses_the_returned_values(cloud_mode, scenario, documents, fake_llm):
    from src.rag.retriever import Retriever

    fake = fake_llm("src.conversation.scoring", "src.conversation.persona")
    fake.chat_replies = ["分かりました。"]
    fake.json_replies = [
        {
            "scores": [
                {"id": "accuracy", "score": 5, "reason": "正確", "advice": "維持"},
                {"id": "empathy", "score": 4, "reason": "配慮あり", "advice": "冒頭で一言"},
                {"id": "problem_grasp", "score": 3, "reason": "確認不足", "advice": "再現確認"},
                {"id": "resolution", "score": 4, "reason": "手順提示", "advice": "期限も"},
                {"id": "grounding", "score": 5, "reason": "根拠あり", "advice": "維持"},
            ],
            "summary": "全体として良好です。",
        }
    ]
    index = Retriever(documents, use_embeddings=False)
    session = finished_session(scenario, index, "送信履歴を確認します。")
    report = score_with_model(session, scenario, load_criteria())
    assert report["method"] == "llm"
    assert report["summary"] == "全体として良好です。"
    assert next(s for s in report["scores"] if s["id"] == "accuracy")["score"] == 5


def test_model_scoring_falls_back_when_the_reply_is_unusable(
    cloud_mode, scenario, documents, fake_llm
):
    from src.rag.retriever import Retriever

    fake = fake_llm("src.conversation.scoring", "src.conversation.persona")
    fake.chat_replies = ["分かりました。"]
    fake.json_replies = [{"scores": []}]
    index = Retriever(documents, use_embeddings=False)
    session = finished_session(scenario, index, "送信履歴を確認します。")

    report = score_with_model(session, scenario, load_criteria())
    assert report["method"] == "rules"
    assert "ルールベース" in report["note"]


def test_model_scoring_clamps_out_of_range_values(cloud_mode, scenario, documents, fake_llm):
    from src.rag.retriever import Retriever

    fake = fake_llm("src.conversation.scoring", "src.conversation.persona")
    fake.chat_replies = ["分かりました。"]
    fake.json_replies = [
        {"scores": [{"id": "accuracy", "score": 99, "reason": "r", "advice": "a"}]}
    ]
    index = Retriever(documents, use_embeddings=False)
    session = finished_session(scenario, index, "送信履歴を確認します。")
    report = score_with_model(session, scenario, load_criteria())
    assert next(s for s in report["scores"] if s["id"] == "accuracy")["score"] == 5
