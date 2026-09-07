"""Scoring a finished conversation against the five criteria.

Local mode scores with the deterministic rules stored beside each criterion, so
a reviewer can reproduce any number by hand. Cloud mode sends the same criteria
as a rubric to the model, and falls back to the rules if the reply is unusable.
"""

import re
from pathlib import Path
from typing import Any

from ..config import settings
from ..rag.loader import load_json
from ..rag.splitter import tokenize
from .persona import Scenario
from .session import agent_turns

SENTENCE_SPLIT = re.compile(r"[。！？\n]")
QUESTION_MARK = re.compile(r"[?？]|ますか|でしょうか|ですか")

SCORING_SYSTEM = (
    "あなたはカスタマーサポート研修の評価者です。"
    "担当者の応対を、与えられた観点ごとに1〜5で採点してください。"
    "5は十分、3は最低限、1はできていないことを表します。"
    "各観点に、点数の理由と、次に何を変えれば1点上がるかの助言を必ず添えてください。"
    "根拠のない加点はせず、記録に現れていないことは評価しないでください。"
    '出力は次のJSONのみ: {"scores":[{"id":"観点ID","score":3,"reason":"理由","advice":"助言"}],'
    '"summary":"全体の総評"}'
)


def load_criteria(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or settings().data_dir / "evaluation_criteria.json"
    payload = load_json(source)
    criteria = [item for item in payload.get("criteria", []) if isinstance(item, dict)]
    if not criteria:
        raise ValueError("評価基準が読み込めませんでした。")
    return criteria


def _clamp(value: float) -> int:
    return max(1, min(5, round(value)))


def _covers(text: str, item: str) -> bool:
    terms = {term for term in tokenize(item) if len(term) > 1}
    if not terms:
        return item in text
    return len(terms & set(tokenize(text))) / len(terms) >= 0.6


def _score_coverage(transcript: str, scenario: Scenario) -> tuple[int, str]:
    items = scenario.must_cover
    if not items:
        return 3, "このシナリオには必須説明項目が設定されていません。"
    covered = [item for item in items if _covers(transcript, item)]
    missing = [item for item in items if item not in covered]
    score = _clamp(1 + 4 * len(covered) / len(items))
    if not missing:
        return score, f"必須説明項目{len(items)}件すべてに触れています。"
    return score, f"触れていない項目: {'、'.join(missing)}。"


def _score_phrase(transcript: str, rules: dict[str, Any]) -> tuple[int, str]:
    hits = [word for word in rules.get("any_of", []) if word in transcript]
    score = _clamp(1 + min(4, len(hits) * 2))
    if hits:
        return score, f"該当する表現を{len(hits)}種類確認しました（例: {hits[0]}）。"
    return score, "該当する表現が見つかりませんでした。"


def _score_question(session: dict[str, Any]) -> tuple[int, str]:
    turns = agent_turns(session)
    if not turns:
        return 1, "担当者の発話が記録されていません。"
    asked = sum(1 for turn in turns if QUESTION_MARK.search(turn))
    ratio = asked / len(turns)
    score = _clamp(1 + 4 * min(1.0, ratio / 0.5))
    return score, f"{len(turns)}回の発話のうち{asked}回で確認の質問をしています。"


def _score_grounding(transcript: str, rules: dict[str, Any]) -> tuple[int, str]:
    penalties = [word for word in rules.get("penalize", []) if word in transcript]
    rewards = [word for word in rules.get("reward", []) if word in transcript]
    score = _clamp(3 + len(rewards) - 2 * len(penalties))
    if penalties:
        return score, f"根拠を示さない推測表現があります（例: {penalties[0]}）。"
    if rewards:
        return score, f"根拠を示す表現を確認しました（例: {rewards[0]}）。"
    return score, "根拠を示す表現も推測表現も見つかりませんでした。"


def score_locally(
    session: dict[str, Any], scenario: Scenario, criteria: list[dict[str, Any]]
) -> dict[str, Any]:
    transcript = " ".join(agent_turns(session))
    scores = []
    for criterion in criteria:
        rules = criterion.get("rules", {})
        kind = rules.get("type")
        if kind == "coverage":
            score, reason = _score_coverage(transcript, scenario)
        elif kind == "question":
            score, reason = _score_question(session)
        elif kind == "penalty_phrase":
            score, reason = _score_grounding(transcript, rules)
        else:
            score, reason = _score_phrase(transcript, rules)
        scores.append(
            {
                "id": criterion["id"],
                "label": criterion["label"],
                "score": score,
                "reason": reason,
                "advice": criterion.get("description", ""),
            }
        )
    return _finalize(scores, method="rules")


def score_with_model(
    session: dict[str, Any], scenario: Scenario, criteria: list[dict[str, Any]]
) -> dict[str, Any]:
    settings().require_cloud()
    from ..llm import chat_json

    rubric = "\n".join(
        f"- {item['id']} ({item['label']}): {item.get('rubric', item.get('description', ''))}"
        for item in criteria
    )
    history = "\n".join(
        f"{'担当者' if turn['role'] == 'agent' else '顧客'}: {turn['content']}"
        for turn in session["transcript"]
    )
    payload = chat_json(
        SCORING_SYSTEM,
        f"【シナリオ】{scenario.label}\n"
        f"【必須説明項目】{'、'.join(scenario.must_cover) or 'なし'}\n\n"
        f"【評価観点】\n{rubric}\n\n【対話記録】\n{history}",
        temperature=0,
    )

    by_id = {item["id"]: item for item in criteria}
    returned = {
        str(row.get("id")): row
        for row in payload.get("scores", [])
        if isinstance(row, dict) and str(row.get("id")) in by_id
    }
    if not returned:
        # モデルが観点を1つも返さなかったときは黙って0点にせずルールへ落とす。
        fallback = score_locally(session, scenario, criteria)
        fallback["note"] = "AIの採点結果を解釈できなかったため、ルールベースの採点を表示しています。"
        return fallback

    scores = []
    for criterion in criteria:
        row = returned.get(criterion["id"], {})
        raw = row.get("score")
        scores.append(
            {
                "id": criterion["id"],
                "label": criterion["label"],
                "score": _clamp(raw) if isinstance(raw, (int, float)) else 3,
                "reason": str(row.get("reason", "")) or "理由が返されませんでした。",
                "advice": str(row.get("advice", "")) or criterion.get("description", ""),
            }
        )
    result = _finalize(scores, method="llm")
    result["summary"] = str(payload.get("summary", "")) or result["summary"]
    return result


def _finalize(scores: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    total = sum(item["score"] for item in scores)
    average = total / len(scores) if scores else 0
    weakest = min(scores, key=lambda item: item["score"]) if scores else None
    return {
        "method": method,
        "scores": scores,
        "total": total,
        "max_total": 5 * len(scores),
        "average": round(average, 2),
        "summary": (
            f"5観点の平均は {average:.1f} 点です。"
            + (f"まず「{weakest['label']}」から見直すと効果的です。" if weakest else "")
        ),
    }


def evaluate(
    session: dict[str, Any], scenario: Scenario, criteria: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if not session.get("finished"):
        raise ValueError("対話を終了してから評価してください。")
    rules = criteria if criteria is not None else load_criteria()
    if settings().is_cloud:
        return score_with_model(session, scenario, rules)
    return score_locally(session, scenario, rules)
