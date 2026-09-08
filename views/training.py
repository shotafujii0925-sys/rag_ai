import streamlit as st

from views._shared import load_resources, setup, show_error
from src.config import settings
from src.conversation.persona import find_scenario
from src.conversation.session import add_turn, agent_turns, finish_session, start_session

setup(
    "模擬対応",
    "顧客役AIとの模擬対応",
    "シナリオを選んで対応を始めます。相手の話を最後まで聞き、事実を確認してから説明してください。",
)

retriever, scenarios, criteria = load_resources()
config = settings()
session = st.session_state.get("session")

# --- シナリオ選択 -------------------------------------------------------------

with st.expander("シナリオを選ぶ", expanded=session is None):
    labels = {scenario.label: scenario for scenario in scenarios}
    chosen_label = st.selectbox("シナリオ", list(labels))
    scenario = labels[chosen_label]

    st.markdown(
        f'<div class="card"><b>{scenario.title}</b>（難易度: {scenario.difficulty}）<br>'
        f"<b>顧客像:</b> {scenario.persona.name} — {scenario.persona.description}<br>"
        f"<b>話し方:</b> {scenario.persona.tone}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"参照の目安: {scenario.reference_hint}")
    st.caption("※ 顧客が持っている前提は、こちらから質問しないと出てきません。")

    if st.button("この設定で開始", type="primary"):
        st.session_state["session"] = start_session(scenario)
        st.session_state["scenario_id"] = scenario.id
        st.session_state.pop("report", None)
        st.session_state.pop("agent_message", None)
        st.rerun()

session = st.session_state.get("session")
if not session:
    st.stop()

try:
    scenario = find_scenario(st.session_state["scenario_id"], scenarios)
except ValueError as error:
    show_error("このセッションのシナリオを読み込めませんでした。もう一度選び直してください。", error)
    for key in ("session", "scenario_id", "report", "agent_message"):
        st.session_state.pop(key, None)
    st.stop()

# --- 対話 ---------------------------------------------------------------------

st.markdown(f"## {scenario.label}")
if session.get("finished"):
    st.success("この対話は終了しています。評価画面で結果を確認してください。")

for turn in session["transcript"]:
    with st.chat_message("user" if turn["role"] == "agent" else "assistant"):
        st.write(turn["content"])

if not session.get("finished"):
    if config.is_cloud:
        with st.expander("音声で入力する"):
            recording = st.audio_input("応対内容を録音", key="agent_audio")
            if recording and st.button("文字起こしして入力欄へ"):
                try:
                    from src.llm import transcribe

                    with st.spinner("文字起こししています…"):
                        st.session_state["agent_message"] = transcribe(recording.getvalue())
                    st.rerun()
                except Exception as error:
                    show_error("音声を文字起こしできませんでした。", error)
    else:
        st.caption("音声入力はクラウドモードでのみ利用できます。")

    with st.form("agent_turn", clear_on_submit=True):
        message = st.text_area("顧客への応対", key="agent_message", height=120)
        sent = st.form_submit_button("送信する", type="primary", use_container_width=True)

    if sent:
        try:
            with st.spinner("顧客の反応を確認しています…"):
                add_turn(session, message, scenario, retriever)
            st.session_state["session"] = session
            st.rerun()
        except ValueError as error:
            st.warning(str(error))
        except Exception as error:
            show_error("顧客の応答を生成できませんでした。", error)

    if st.button("対話を終了して評価へ", disabled=not agent_turns(session)):
        try:
            finish_session(session)
            st.session_state["session"] = session
            st.switch_page("views/review.py")
        except ValueError as error:
            st.warning(str(error))
else:
    left, right = st.columns(2)
    if left.button("評価を見る", type="primary", use_container_width=True):
        st.switch_page("views/review.py")
    if right.button("最初からやり直す", use_container_width=True):
        for key in ("session", "scenario_id", "report", "agent_message"):
            st.session_state.pop(key, None)
        st.rerun()
