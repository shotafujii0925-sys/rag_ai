"""Thin OpenAI-compatible client used only in cloud mode.

Local mode never reaches this module: callers check `settings().is_cloud` first
and take a deterministic path instead. That keeps the whole app runnable, and
testable, with no key and no network.
"""

import json
from functools import lru_cache
from typing import Any

from .config import ConfigError, settings


@lru_cache(maxsize=8)
def _client_for(api_key: str, base_url: str | None, timeout: int, max_retries: int) -> Any:
    """Cached per (key, endpoint, ...), not globally.

    A UI-entered key can differ between concurrent sessions in the same process;
    caching by its value keeps one visitor's client from being reused — and
    billed — for another visitor's requests.
    """
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)


def _client() -> Any:
    config = settings()
    config.require_cloud()
    return _client_for(config.api_key, config.base_url, config.timeout_seconds, config.max_retries)


def chat(system: str, user: str, *, json_mode: bool = False, temperature: float | None = None) -> str:
    config = settings()
    request: dict[str, Any] = {
        "model": config.chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature if temperature is None else temperature,
    }
    if json_mode:
        request["response_format"] = {"type": "json_object"}
    response = _client().chat.completions.create(**request)
    return response.choices[0].message.content or ""


def chat_json(system: str, user: str, **kwargs: Any) -> dict[str, Any]:
    raw = chat(system, user, json_mode=True, **kwargs).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as error:
        raise ConfigError("AIの応答を解釈できませんでした。もう一度お試しください。") from error
    return parsed if isinstance(parsed, dict) else {}


def embed(texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    model = settings().embedding_model
    for start in range(0, len(texts), batch_size):
        response = _client().embeddings.create(model=model, input=texts[start : start + batch_size])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def transcribe(audio: bytes, *, filename: str = "recording.wav") -> str:
    if not audio:
        raise ValueError("音声データが空です。もう一度録音してください。")
    response = _client().audio.transcriptions.create(
        model=settings().transcription_model,
        file=(filename, audio),
    )
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise ValueError("音声を認識できませんでした。静かな場所で録音し直してください。")
    return text
