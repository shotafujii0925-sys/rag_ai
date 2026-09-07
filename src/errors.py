"""User-facing error handling.

Every message shown in the UI goes through `safe_message`. Internal detail —
API keys, endpoints, file paths, stack traces — must never reach the screen or
the log, because a screenshot of this app could otherwise leak configuration.
"""

import logging
import re

from .config import ConfigError

logger = logging.getLogger("support_trainer")

# ログに出す前に潰すパターン。値そのものを消し、残っていることだけ分かるようにする。
_REDACTIONS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "[redacted-key]"),
    (re.compile(r"https?://[^\s\"']+"), "[redacted-url]"),
    (re.compile(r"[A-Za-z]:\\[^\s\"']+"), "[redacted-path]"),
    (re.compile(r"/(?:home|Users)/[^\s\"']+"), "[redacted-path]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[redacted-email]"),
)

FALLBACK = "処理中に問題が発生しました。入力内容を確認して、もう一度お試しください。"

# 例外の型ごとの利用者向け文言。内部の理由は出さない。
_BY_TYPE = {
    "APITimeoutError": "応答に時間がかかりすぎたため中断しました。しばらく待って再度お試しください。",
    "APIConnectionError": "外部サービスへ接続できませんでした。ネットワーク接続を確認してください。",
    "AuthenticationError": "APIキーが受け付けられませんでした。.env の設定を確認してください。",
    "PermissionDeniedError": "この操作を行う権限がありません。APIキーの権限設定を確認してください。",
    "RateLimitError": "リクエストが集中しています。少し時間をおいて再度お試しください。",
    "BadRequestError": "リクエストの内容が受け付けられませんでした。入力を短くして再度お試しください。",
    "JSONDecodeError": "AIの応答を解釈できませんでした。もう一度お試しください。",
    "TimeoutError": "応答に時間がかかりすぎたため中断しました。しばらく待って再度お試しください。",
    "ConnectionError": "外部サービスへ接続できませんでした。ネットワーク接続を確認してください。",
    "FileNotFoundError": "必要なファイルが見つかりませんでした。data ディレクトリを確認してください。",
    "UnicodeDecodeError": "ファイルの文字コードを判別できませんでした。UTF-8で保存し直してください。",
}


def redact(text: str) -> str:
    cleaned = str(text or "")
    for pattern, replacement in _REDACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def safe_message(error: Exception) -> str:
    """Return a message that is safe to show a user."""
    # 型ごとの文言を先に見る。JSONDecodeError と UnicodeDecodeError は ValueError の
    # 派生で、素通しすると内部のパース位置やバイト列が画面に出てしまう。
    known = _BY_TYPE.get(type(error).__name__)
    if known:
        return known
    # 残る ValueError / ConfigError は、こちらが書いた検証メッセージなのでそのまま出す。
    if isinstance(error, (ConfigError, ValueError)):
        return redact(str(error)) or FALLBACK
    return FALLBACK


def log_error(context: str, error: Exception) -> None:
    """Log the type and a redacted message. Never log the payload or a traceback."""
    logger.error("%s: %s: %s", context, type(error).__name__, redact(str(error)))
