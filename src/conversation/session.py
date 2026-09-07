"""Conversation history.

A session is a plain dict so Streamlit can keep it in session_state and the whole
thing can be exported as JSON. Only the transcript is retained — no identifiers,
no timestamps beyond the session start, nothing that would need scrubbing before
a trainee shares their result.
"""

from datetime import datetime, timezone
from typing import Any

from ..config import settings
from ..rag.generator import build_context
from ..rag.retriever import Retriever
from .persona import Scenario, generate_reply, scripted_reply

AGENT = "agent"
CUSTOMER = "customer"
MAX_MESSAGE_CHARS = 2000
MAX_TURNS = 40


def start_session(scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": settings().mode,
        "opening": scenario.opening,
        "transcript": [{"role": CUSTOMER, "content": scenario.opening}],
        "retrieval_log": [],
        "finished": False,
    }


def agent_turns(session: dict[str, Any]) -> list[str]:
    return [turn["content"] for turn in session["transcript"] if turn["role"] == AGENT]


def customer_turns(session: dict[str, Any]) -> list[str]:
    return [turn["content"] for turn in session["transcript"] if turn["role"] == CUSTOMER]


def validate_message(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        raise ValueError("説明の内容を入力してください。")
    if len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"1回の発言は{MAX_MESSAGE_CHARS}文字以内にしてください。")
    return text


def add_turn(
    session: dict[str, Any], message: str, scenario: Scenario, retriever: Retriever
) -> str:
    """Record the trainee's message and produce the customer's reply."""
    if session.get("finished"):
        raise ValueError("この対話はすでに終了しています。")
    if len(session["transcript"]) >= MAX_TURNS:
        raise ValueError("対話が長くなりすぎました。一度終了して評価に進んでください。")

    text = validate_message(message)
    session["transcript"].append({"role": AGENT, "content": text})

    result = retriever.retrieve(f"{scenario.title} {text}")
    session["retrieval_log"].append(
        {
            "query": result.query,
            "method": result.method,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "passages": [passage.as_dict() for passage in result.passages],
        }
    )

    if settings().is_cloud:
        reply = generate_reply(scenario, session["transcript"], build_context(result.passages))
    else:
        reply = scripted_reply(scenario, len(agent_turns(session)) - 1)

    session["transcript"].append({"role": CUSTOMER, "content": reply})
    return reply


def finish_session(session: dict[str, Any]) -> dict[str, Any]:
    if not agent_turns(session):
        raise ValueError("まだ一度も応対していません。対話を進めてから終了してください。")
    session["finished"] = True
    session["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return session
