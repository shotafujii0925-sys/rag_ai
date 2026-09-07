"""Settings. Everything tunable comes from the environment, nothing is hardcoded.

The one exception is a key entered in the UI for the current browser session
(see `set_session_api_key`) — it never touches `os.environ`, `.env`, or disk.
"""

import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
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
def _env_settings() -> Settings:
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


# UI から入力されたキーは、ブラウザセッション単位の contextvar にだけ置く。
# Streamlit は1プロセスで複数人のセッションを同時に処理できるため、これを
# os.environ やモジュール変数のような共有領域に置くと、ある訪問者のキーで
# 別の訪問者のリクエストが動いてしまう。contextvar ならセッションごとの
# 実行系列に閉じるので、その混線が起きない。
_session_api_key: ContextVar[str | None] = ContextVar("session_api_key", default=None)


def set_session_api_key(key: str | None) -> None:
    """Register a UI-entered key for the current session only.

    Call this once near the top of every script run (see
    `views/_shared.render_sidebar`), before any code reads `settings()`.
    Never written to `.env`, disk, or a log — held in memory for this session only
    and discarded when the browser tab closes.
    """
    _session_api_key.set((key or "").strip() or None)


def settings() -> Settings:
    base = _env_settings()
    override = _session_api_key.get()
    if override is None:
        return base
    # UI にキーが入っている間は、.env の APP_MODE に関わらずクラウド機能を有効にする。
    return replace(base, api_key=override, mode=CLOUD)


# 既存の呼び出し側・テストが `settings.cache_clear()` で環境変数の再読み込みを
# 促せるように、内部でキャッシュしている関数のクリアをそのまま委譲する。
settings.cache_clear = _env_settings.cache_clear


def cloud_requested_but_unavailable() -> bool:
    """True when the user asked for cloud mode but no key was found."""
    return (os.getenv("APP_MODE") or LOCAL).strip().lower() == CLOUD and not (
        os.getenv("OPENAI_API_KEY") or ""
    ).strip()
