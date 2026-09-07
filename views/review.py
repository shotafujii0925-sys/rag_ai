import json

import streamlit as st

from views._shared import load_resources, setup, show_error
from src.conversation.persona import find_scenario
from src.conversation.scoring import evaluate

setup(
    "評価",
    "対応の評価",
    "5つの観点で採点します。点数そのものより、理由と改善アドバイスを読んで次の対応に反映してください。",
)

retriever, scenarios, criteria = load_resources()
session = st.session_state.get("session")

if not session:
    st.info("まだ対話がありません。「模擬対応」から始めてください。")
    if st.button("模擬対応へ"):
        st.switch_page("views/training.py")
    st.stop()

if not session.get("finished"):
    st.warning("対話がまだ終了していません。「模擬対応」画面で終了してから評価してください。")
    if st.button("模擬対応へ戻る"):
        st.switch_page("views/training.py")
    st.stop()

scenario = find_scenario(st.session_state["scenario_id"], scenarios)
report = st.session_state.get("report")

if report is None:
    try:
        with st.spinner("応対を評価しています…"):
            report = evaluate(session, scenario, criteria)
        st.session_state["report"] = report
    except Exception as error:
        show_error("評価を実行できませんでした。", error)
        st.stop()

st.caption(f"シナリオ: {scenario.label}｜採点方法: {'AI採点' if report['method'] == 'llm' else 'ルールベース採点'}")
if report.get("note"):
    st.warning(report["note"])

columns = st.columns(len(report["scores"]) + 1)
columns[0].metric("平均", f"{report['average']} / 5")
for column, item in zip(columns[1:], report["scores"]):
    column.metric(item["label"], f"{item['score']} / 5")

st.markdown(f'<div class="card">{report["summary"]}</div>', unsafe_allow_html=True)

st.markdown("## 観点ごとの講評")
for item in report["scores"]:
    with st.expander(f"{item['label']} — {item['score']} / 5", expanded=item["score"] <= 3):
        st.markdown(f"**点数の理由**\n\n{item['reason']}")
        if item.get("advice"):
            st.markdown(f"**改善アドバイス**\n\n{item['advice']}")

st.markdown("## 必須説明項目")
transcript = " ".join(turn["content"] for turn in session["transcript"] if turn["role"] == "agent")
st.dataframe(
    [
        {"必須説明項目": item, "触れた": any(part in transcript for part in item.split("、"))}
        for item in scenario.must_cover
    ]
    or [{"必須説明項目": "設定なし", "触れた": False}],
    hide_index=True,
    use_container_width=True,
)

st.markdown("## 対話記録")
with st.expander("全文を表示"):
    for turn in session["transcript"]:
        role = "担当者" if turn["role"] == "agent" else "顧客"
        st.markdown(f"**{role}**")
        st.markdown(f'<div class="plain">{turn["content"]}</div>', unsafe_allow_html=True)

st.divider()
left, right = st.columns(2)
left.download_button(
    "結果をJSONで保存",
    data=json.dumps(
        {"scenario": scenario.label, "report": report, "transcript": session["transcript"]},
        ensure_ascii=False,
        indent=2,
    ),
    file_name="training-review.json",
    mime="application/json",
    use_container_width=True,
)
if right.button("別のシナリオで練習する", type="primary", use_container_width=True):
    for key in ("session", "scenario_id", "report", "agent_message"):
        st.session_state.pop(key, None)
    st.switch_page("views/training.py")
