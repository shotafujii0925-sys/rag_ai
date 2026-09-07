"""RAG quality metrics.

The four metrics follow the RAGAS definitions, implemented here directly rather
than through the library. Two reasons: the judging prompts stay readable and
reviewable in this repository, and the retrieval-side metrics can then be
computed from labels with no API key and no cost.

Each result carries `method` so a report can never be mistaken for something it
was not: "llm" means a model scored it, "label" means it came from the labelled
evaluation set.
"""

import re
from typing import Any

from ..config import settings
from .retriever import Passage
from .splitter import tokenize

SENTENCE_SPLIT = re.compile(r"[。！？\n]")

FAITHFULNESS_SYSTEM = (
    "あなたは回答の忠実性を検証する評価者です。"
    "回答を検証可能な最小単位の主張に分解し、それぞれが参照文書だけから確認できるかを判定してください。"
    "参照文書に書かれていない主張は、一般常識として正しくても supported=false としてください。"
    "挨拶や言い換えなど事実を主張していない文は無視します。"
    '出力は次のJSONのみ: {"claims":[{"claim":"主張","supported":true,"reason":"根拠または不足点"}]}'
)

RELEVANCY_SYSTEM = (
    "あなたは回答から質問を復元する評価者です。"
    "与えられた回答が答えになるような質問を3件、日本語で作ってください。"
    "回答が「わからない」「確認できません」など実質的な情報を含まない場合は noncommittal を true にします。"
    '出力は次のJSONのみ: {"questions":["質問1","質問2","質問3"],"noncommittal":false}'
)

RECALL_SYSTEM = (
    "あなたは想定回答の各文が、参照文書から導けるかを判定する評価者です。"
    "想定回答を文に分け、それぞれが参照文書の記載から確認できるかを判定してください。"
    '出力は次のJSONのみ: {"sentences":[{"sentence":"文","attributed":true}]}'
)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = (sum(a * a for a in left) ** 0.5) * (sum(b * b for b in right) ** 0.5)
    return dot / norm if norm else 0.0


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT.split(str(text or "")) if part.strip()]


# --- generation side (needs a model) -----------------------------------------


def faithfulness(answer: str, contexts: list[str]) -> dict[str, Any]:
    """Share of the answer's claims that the retrieved contexts support."""
    settings().require_cloud()
    body = str(answer or "").strip()
    if not body:
        raise ValueError("評価する回答がありません。")
    if not contexts:
        raise ValueError("評価に使う参照文書がありません。")

    from ..llm import chat_json

    joined = "\n\n".join(f"[参照文書 {i}]\n{text}" for i, text in enumerate(contexts, 1))
    result = chat_json(
        FAITHFULNESS_SYSTEM, f"回答:\n{body}\n\n参照文書:\n{joined[:12000]}", temperature=0
    )
    claims = [claim for claim in result.get("claims", []) if isinstance(claim, dict)]
    if not claims:
        return {"score": 0.0, "supported": 0, "total": 0, "claims": [], "method": "llm"}

    supported = sum(1 for claim in claims if claim.get("supported") is True)
    return {
        "score": supported / len(claims),
        "supported": supported,
        "total": len(claims),
        "claims": claims,
        "method": "llm",
    }


def answer_relevancy(question: str, answer: str) -> dict[str, Any]:
    """How directly the answer addresses the question.

    Generates the questions the answer would answer, then compares them to the
    real one in embedding space. An evasive answer scores 0 regardless.
    """
    settings().require_cloud()
    asked = str(question or "").strip()
    body = str(answer or "").strip()
    if not asked:
        raise ValueError("評価する質問がありません。")
    if not body:
        raise ValueError("評価する回答がありません。")

    from ..llm import chat_json, embed

    result = chat_json(RELEVANCY_SYSTEM, f"回答:\n{body}", temperature=0)
    generated = [str(item).strip() for item in result.get("questions", []) if str(item).strip()]
    noncommittal = bool(result.get("noncommittal"))
    if not generated:
        return {"score": 0.0, "noncommittal": noncommittal, "generated": [], "method": "llm"}

    vectors = embed([asked, *generated])
    similarities = [_cosine(vectors[0], vector) for vector in vectors[1:]]
    return {
        "score": 0.0 if noncommittal else sum(similarities) / len(similarities),
        "noncommittal": noncommittal,
        "generated": generated,
        "method": "llm",
    }


# --- retrieval side (labels, no model needed) --------------------------------


def context_precision(passages: list[Passage], expected_source: str) -> dict[str, Any]:
    """Share of retrieved passages that came from the document holding the answer.

    RAGAS judges each context with a model; here the labelled evaluation set
    supplies the same judgement, which keeps the sweep free to run.
    """
    if not passages:
        return {"score": 0.0, "relevant": 0, "retrieved": 0, "method": "label"}
    relevant = sum(1 for passage in passages if passage.source == expected_source)
    return {
        "score": relevant / len(passages),
        "relevant": relevant,
        "retrieved": len(passages),
        "method": "label",
    }


def context_recall(ground_truth: str, contexts: list[str], *, threshold: float = 0.6) -> dict[str, Any]:
    """Share of the expected answer's sentences that the contexts can account for.

    Offline approximation: a sentence counts as attributed when most of its terms
    appear somewhere in the retrieved context. Use `context_recall_llm` when a
    model is available and a stricter judgement is wanted.
    """
    sentences = _sentences(ground_truth)
    if not sentences:
        raise ValueError("想定回答が空です。")
    if not contexts:
        return {"score": 0.0, "attributed": 0, "total": len(sentences), "method": "label"}

    context_tokens = set(tokenize(" ".join(contexts)))
    attributed = 0
    details = []
    for sentence in sentences:
        terms = {term for term in tokenize(sentence) if len(term) > 1}
        ratio = len(terms & context_tokens) / len(terms) if terms else 0.0
        hit = ratio >= threshold
        attributed += int(hit)
        details.append({"sentence": sentence, "attributed": hit, "ratio": round(ratio, 3)})

    return {
        "score": attributed / len(sentences),
        "attributed": attributed,
        "total": len(sentences),
        "sentences": details,
        "method": "label",
    }


def context_recall_llm(ground_truth: str, contexts: list[str]) -> dict[str, Any]:
    settings().require_cloud()
    sentences = _sentences(ground_truth)
    if not sentences:
        raise ValueError("想定回答が空です。")
    if not contexts:
        return {"score": 0.0, "attributed": 0, "total": len(sentences), "method": "llm"}

    from ..llm import chat_json

    joined = "\n\n".join(f"[参照文書 {i}]\n{text}" for i, text in enumerate(contexts, 1))
    result = chat_json(
        RECALL_SYSTEM,
        f"想定回答:\n{ground_truth}\n\n参照文書:\n{joined[:12000]}",
        temperature=0,
    )
    judged = [item for item in result.get("sentences", []) if isinstance(item, dict)]
    if not judged:
        return {"score": 0.0, "attributed": 0, "total": len(sentences), "method": "llm"}

    attributed = sum(1 for item in judged if item.get("attributed") is True)
    return {
        "score": attributed / len(judged),
        "attributed": attributed,
        "total": len(judged),
        "sentences": judged,
        "method": "llm",
    }
