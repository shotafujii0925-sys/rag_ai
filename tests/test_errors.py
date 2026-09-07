"""利用者向けエラー表示とログから秘密情報が漏れないこと。"""

import json
import logging

import pytest

from src.config import ConfigError
from src.errors import FALLBACK, log_error, redact, safe_message


# キー形式の文字列をソースに直接書くと、tests/test_no_secrets.py の
# 秘密情報スキャナが正しく反応してしまう。組み立てて使う。
FAKE_KEY = "sk-" + "abcdefghijklmnopqrstuvwxyz012345"


@pytest.mark.parametrize(
    "text",
    [
        "key=" + FAKE_KEY,
        "failed to reach https://internal.example.com/v1/chat",
        "file C:\\Users\\someone\\secret\\config.py missing",
        "path /home/someone/app/.env not found",
        "contact person@example.co.jp for access",
    ],
)
def test_redact_removes_sensitive_fragments(text):
    cleaned = redact(text)
    assert FAKE_KEY not in cleaned
    assert "internal.example.com" not in cleaned
    assert "someone" not in cleaned
    assert "person@example.co.jp" not in cleaned
    assert "redacted" in cleaned


def test_redact_leaves_plain_text_alone():
    assert redact("入力が空です。") == "入力が空です。"


def test_config_errors_are_shown_verbatim():
    assert safe_message(ConfigError("TOP_K は 1〜50 の範囲で指定してください。")).startswith("TOP_K")


def test_value_errors_are_shown_verbatim():
    assert safe_message(ValueError("質問を入力してください。")) == "質問を入力してください。"


def test_value_error_text_is_still_redacted():
    assert "sk-" not in safe_message(ValueError("key " + FAKE_KEY))


@pytest.mark.parametrize(
    "error,expected",
    [
        (TimeoutError("boom"), "時間がかかりすぎた"),
        (ConnectionError("boom"), "接続できませんでした"),
        (FileNotFoundError("boom"), "見つかりませんでした"),
        (json.JSONDecodeError("boom", "{}", 0), "解釈できませんでした"),
    ],
)
def test_known_error_types_get_a_specific_message(error, expected):
    assert expected in safe_message(error)


def test_unknown_errors_fall_back_to_a_generic_message():
    class Weird(Exception):
        pass

    assert safe_message(Weird("internal detail " + FAKE_KEY)) == FALLBACK


def test_messages_never_contain_a_traceback():
    try:
        raise RuntimeError("deep internal failure at line 42")
    except RuntimeError as error:
        message = safe_message(error)
    assert "Traceback" not in message
    assert "line 42" not in message


def test_log_records_the_type_but_redacts_the_message(caplog):
    with caplog.at_level(logging.ERROR, logger="support_trainer"):
        log_error("生成", RuntimeError("token " + FAKE_KEY + " rejected"))
    logged = caplog.text
    assert "RuntimeError" in logged
    assert FAKE_KEY not in logged
    assert "redacted-key" in logged


def test_log_does_not_record_urls(caplog):
    with caplog.at_level(logging.ERROR, logger="support_trainer"):
        log_error("接続", ConnectionError("https://api.internal.example/v1 unreachable"))
    assert "api.internal.example" not in caplog.text
