"""Settings. Everything tunable comes from the environment, nothing is hardcoded."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

LOCAL = "local"
CLOUD = "cloud"


class ConfigError(ValueError):
    """A setting is present but unusable. Message is safe to show a user."""


def _int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} は整数で指定してください。") from None
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} は {minimum}〜{maximum} の範囲で指定してください。")
    return value


def _float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} は数値で指定してください。") from None
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} は {minimum}〜{maximum} の範囲で指定してください。")
    return value


@dataclass(frozen=True)
class Settings:
    mode: str
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    transcription_model: str
    temperature: float
    timeout_seconds: int
    max_retries: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    max_upload_mb: int

    @property
    def is_cloud(self) -> bool:
        return self.mode == CLOUD

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    def require_cloud(self) -> None:
        """Raise a user-safe error when a cloud-only feature is reached in local mode."""
        if not self.is_cloud:
            raise ConfigError(
                "この機能はクラウドモードでのみ利用できます。"
                "APP_MODE=cloud と OPENAI_API_KEY を .env に設定してください。"
            )


@lru_cache(maxsize=1)
def settings() -> Settings:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    requested = (os.getenv("APP_MODE") or LOCAL).strip().lower()
    if requested not in {LOCAL, CLOUD}:
        raise ConfigError(f"APP_MODE は {LOCAL} か {CLOUD} を指定してください。")
    # 鍵が無いのに cloud を指定されたら、黙って落とさず local に倒す。
    mode = CLOUD if (requested == CLOUD and api_key) else LOCAL

    chunk_size = _int("CHUNK_SIZE", 600, minimum=50, maximum=8000)
    chunk_overlap = _int("CHUNK_OVERLAP", 100, minimum=0, maximum=4000)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP は CHUNK_SIZE より小さい値にしてください。")

    return Settings(
        mode=mode,
        api_key=api_key,
        base_url=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
        temperature=_float("TEMPERATURE", 0.3, minimum=0.0, maximum=2.0),
        timeout_seconds=_int("TIMEOUT_SECONDS", 30, minimum=1, maximum=600),
        max_retries=_int("MAX_RETRIES", 2, minimum=0, maximum=10),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_int("TOP_K", 4, minimum=1, maximum=50),
        score_threshold=_float("SCORE_THRESHOLD", 0.15, minimum=0.0, maximum=1.0),
        max_upload_mb=_int("MAX_UPLOAD_MB", 5, minimum=1, maximum=50),
    )


def cloud_requested_but_unavailable() -> bool:
    """True when the user asked for cloud mode but no key was found."""
    return (os.getenv("APP_MODE") or LOCAL).strip().lower() == CLOUD and not (
        os.getenv("OPENAI_API_KEY") or ""
    ).strip()
